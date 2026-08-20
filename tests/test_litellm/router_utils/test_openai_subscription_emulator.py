from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import os
import time
import traceback
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio

import litellm
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.exceptions import (
    InternalServerError,
    NotFoundError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.llms.chatgpt.authenticator import Authenticator
from litellm.router_utils.openai_subscription_affinity import (
    OpenAIProfileState,
    OpenAISubscriptionAffinityStore,
)
from litellm.router_utils.pre_call_checks.openai_subscription_affinity_check import (
    OpenAISubscriptionAffinityCheck,
)
from tests.test_litellm.router_utils.openai_subscription_emulator import (
    EmulatedOpenAIResponse,
    OpenAISubscriptionEmulator,
)

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
pytestmark = [
    pytest.mark.skipif(REDIS_HOST is None, reason="requires a disposable Redis instance"),
    pytest.mark.asyncio(loop_scope="module"),
]

PROFILE_A = "subscription-a"
PROFILE_B = "subscription-b"
PROFILES = (PROFILE_A, PROFILE_B)


def _user_hash(index: int) -> str:
    return hashlib.sha256(f"emulated-virtual-key-{index}".encode()).hexdigest()


def _make_test_jwt(profile: str) -> str:
    def encode(value: dict[str, object]) -> str:
        raw = json.dumps(value, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return (
        f"{encode({'alg': 'none', 'typ': 'JWT'})}.{encode({'exp': int(time.time()) + 3600, 'test_profile': profile})}."
    )


def _write_auth_file(path: Path, token: str, profile: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "access_token": token,
                "account_id": f"test-account-{profile}",
                "expires_at": time.time() + 3600,
            }
        ),
        encoding="utf-8",
    )


async def _wait_for(predicate: Callable[[], Any], *, timeout: float = 3) -> Any:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("timed out waiting for an asynchronous callback")


@dataclass(slots=True)
class EmulatedPool:
    emulator: OpenAISubscriptionEmulator
    store: OpenAISubscriptionAffinityStore
    auth_files: dict[str, Path]
    routers: list[litellm.Router]
    device_login_attempts: list[str]

    def make_router(
        self,
        *,
        models: tuple[str, ...] = ("gpt-5.4", "gpt-5.3-codex"),
        num_retries: int = 2,
        timeout: float = 5,
        ttl_seconds: int = 60,
    ) -> tuple[litellm.Router, OpenAISubscriptionAffinityCheck]:
        model_list = []
        for model in models:
            for profile in PROFILES:
                model_list.append(
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": f"chatgpt/{model}",
                            "chatgpt_auth_file": str(self.auth_files[profile]),
                        },
                        "model_info": {
                            "id": f"{model}-{profile}",
                            "openai_oauth_profile": profile,
                        },
                    }
                )
        router = litellm.Router(
            model_list=model_list,
            optional_pre_call_checks=["openai_subscription_affinity"],
            redis_host=REDIS_HOST,
            redis_port=REDIS_PORT,
            num_retries=num_retries,
            timeout=timeout,
            retry_after=0,
            disable_cooldowns=True,
            deployment_affinity_ttl_seconds=ttl_seconds,
        )
        self.routers.append(router)
        callback = next(
            item for item in (router.optional_callbacks or []) if isinstance(item, OpenAISubscriptionAffinityCheck)
        )
        return router, callback


@pytest_asyncio.fixture(scope="module", loop_scope="module", autouse=True)
async def stop_logging_worker_after_module():
    yield
    await asyncio.sleep(0.2)
    await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
    await GLOBAL_LOGGING_WORKER.stop()


@pytest_asyncio.fixture(loop_scope="module")
async def emulated_pool(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    tokens = {profile: _make_test_jwt(profile) for profile in PROFILES}
    auth_files = {profile: tmp_path / profile / "auth.json" for profile in PROFILES}
    for profile in PROFILES:
        _write_auth_file(auth_files[profile], tokens[profile], profile)

    emulator = OpenAISubscriptionEmulator(tokens).start()
    monkeypatch.setenv("CHATGPT_API_BASE", emulator.base_url)
    monkeypatch.setenv("OPENAI_CHATGPT_API_BASE", emulator.base_url)

    device_login_attempts: list[str] = []

    def reject_device_login(_self):
        device_login_attempts.append("".join(traceback.format_stack(limit=12)))
        raise AssertionError("per-profile routing must not use interactive device login")

    monkeypatch.setattr(Authenticator, "_login_device_code", reject_device_login)

    redis_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT)
    redis_client = redis_cache.init_async_client()
    await redis_client.flushdb()
    store = OpenAISubscriptionAffinityStore(DualCache(redis_cache=redis_cache))
    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = []
    pool = EmulatedPool(
        emulator=emulator,
        store=store,
        auth_files=auth_files,
        routers=[],
        device_login_attempts=device_login_attempts,
    )
    try:
        yield pool
    finally:
        await asyncio.sleep(0.05)
        await asyncio.wait_for(GLOBAL_LOGGING_WORKER.flush(), timeout=5)
        for router in reversed(pool.routers):
            router.discard()
        await litellm.close_litellm_async_clients()
        litellm.callbacks = original_callbacks
        emulator.stop()
        await redis_client.flushdb()
        assert not device_login_attempts, device_login_attempts[0]


async def test_real_router_reaches_emulator_and_persists_affinity(emulated_pool: EmulatedPool) -> None:
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0)
    user_hash = _user_hash(1)

    response = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )

    assert response.choices[0].message.content == "emulated response"
    assert await _wait_for(lambda: len(emulated_pool.emulator.requests)) == 1
    request = emulated_pool.emulator.requests[0]
    assert request.path == "/responses"
    assert request.profile == PROFILE_A
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_A


async def test_empty_completed_response_is_successful_without_retry_or_device_login(
    emulated_pool: EmulatedPool,
) -> None:
    emulated_pool.emulator.enqueue(
        PROFILE_A,
        EmulatedOpenAIResponse(empty_completed_output=True),
    )
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=2)
    user_hash = _user_hash(2)

    response = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )

    assert response.choices[0].message.content == ""
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A]
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_A
    assert not emulated_pool.device_login_attempts


async def test_incomplete_response_is_not_retried_on_another_profile(
    emulated_pool: EmulatedPool,
) -> None:
    emulated_pool.emulator.enqueue(
        PROFILE_A,
        EmulatedOpenAIResponse(
            responses_status="incomplete",
            incomplete_reason="max_output_tokens",
        ),
    )
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=2)

    with pytest.raises(litellm.BadRequestError, match="max_output_tokens"):
        await router.acompletion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_hash": _user_hash(3)},
        )

    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A]
    assert (await emulated_pool.store.get_profile_state(PROFILE_A)).state is OpenAIProfileState.AVAILABLE
    assert not emulated_pool.device_login_attempts


async def test_missing_auth_file_disables_only_that_profile_and_fails_over(
    emulated_pool: EmulatedPool,
) -> None:
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=2)
    emulated_pool.auth_files[PROFILE_A].unlink()
    user_hash = _user_hash(4)

    response = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )

    assert response.choices[0].message.content == "emulated response"
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_B]
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_B
    assert (await emulated_pool.store.get_profile_state(PROFILE_A)).state is OpenAIProfileState.DISABLED
    assert not emulated_pool.device_login_attempts


async def test_parallel_real_requests_are_evenly_distributed(emulated_pool: EmulatedPool) -> None:
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0)

    responses = await asyncio.gather(
        *(
            router.acompletion(
                model="gpt-5.4",
                messages=[{"role": "user", "content": "hello"}],
                metadata={"user_api_key_hash": _user_hash(index)},
            )
            for index in range(6)
        )
    )

    assert all(response.choices[0].message.content == "emulated response" for response in responses)
    assert Counter(request.profile for request in emulated_pool.emulator.requests) == {
        PROFILE_A: 3,
        PROFILE_B: 3,
    }


async def test_completion_and_responses_share_binding_and_observe_limit_headers(
    emulated_pool: EmulatedPool,
) -> None:
    headers = {
        "x-codex-primary-used-percent": "25",
        "x-codex-primary-window-minutes": "300",
        "x-codex-primary-reset-at": "1700000300",
    }
    emulated_pool.emulator.enqueue(
        PROFILE_A,
        EmulatedOpenAIResponse(headers=headers),
        EmulatedOpenAIResponse(headers=headers),
    )
    router, callback = emulated_pool.make_router(num_retries=0, ttl_seconds=5)
    metrics_logger = MagicMock()
    callback.metrics_logger = metrics_logger
    user_hash = _user_hash(10)

    completion = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )
    responses_api = await router.aresponses(
        model="gpt-5.3-codex",
        input="hello",
        metadata={"user_api_key_hash": user_hash},
    )
    response_events = [event async for event in responses_api]

    assert completion.choices[0].message.content == "emulated response"
    assert response_events[-1].type == "response.completed"
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A, PROFILE_A]
    assert [request.path for request in emulated_pool.emulator.requests] == ["/responses", "/responses"]
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_A
    assert await emulated_pool.store.get_remaining_ttl(user_hash) in range(4, 6)
    await _wait_for(lambda: metrics_logger.observe_openai_subscription_limit_window.call_count == 2)
    for metric_call in metrics_logger.observe_openai_subscription_limit_window.call_args_list:
        assert metric_call.kwargs["profile"] == PROFILE_A
        assert metric_call.kwargs["window"] == "primary"
        assert metric_call.kwargs["used_ratio"] == 0.25


async def test_rate_limit_retries_on_another_profile_and_moves_binding(emulated_pool: EmulatedPool) -> None:
    emulated_pool.emulator.enqueue(
        PROFILE_A,
        EmulatedOpenAIResponse(
            status_code=429,
            headers={
                "Retry-After": "120",
                "x-codex-primary-used-percent": "100",
            },
        ),
    )
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=2)
    user_hash = _user_hash(20)

    response = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )

    assert response.choices[0].message.content == "emulated response"
    await _wait_for(lambda: len(emulated_pool.emulator.requests) == 2)
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A, PROFILE_B]
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_B
    snapshot = await emulated_pool.store.get_profile_state(PROFILE_A)
    assert snapshot.state is OpenAIProfileState.COOLDOWN
    assert snapshot.reset_at is not None
    assert snapshot.reset_at > int(time.time())


async def test_streaming_completion_uses_real_router_and_refreshes_binding(emulated_pool: EmulatedPool) -> None:
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0, ttl_seconds=5)
    user_hash = _user_hash(30)

    stream = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        stream=True,
        metadata={"user_api_key_hash": user_hash},
    )
    chunks = [chunk async for chunk in stream]

    streamed_text = "".join(
        chunk.choices[0].delta.content or ""
        for chunk in chunks
        if chunk.choices and chunk.choices[0].delta.content is not None
    )
    assert emulated_pool.emulator.requests[0].streaming is True
    assert streamed_text == "emulated stream"
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_A
    assert await emulated_pool.store.get_remaining_ttl(user_hash) in range(4, 6)


async def test_all_rate_limited_profiles_return_neutral_429_with_nearest_retry_after(
    emulated_pool: EmulatedPool,
) -> None:
    emulated_pool.emulator.enqueue(
        PROFILE_A,
        EmulatedOpenAIResponse(status_code=429, headers={"Retry-After": "120"}),
    )
    emulated_pool.emulator.enqueue(
        PROFILE_B,
        EmulatedOpenAIResponse(status_code=429, headers={"Retry-After": "60"}),
    )
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=3)

    with pytest.raises(RateLimitError) as exc_info:
        await router.acompletion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_hash": _user_hash(40)},
        )

    error = exc_info.value
    assert error.status_code == 429
    assert PROFILE_A not in str(error)
    assert PROFILE_B not in str(error)
    retry_after = int(error.headers["retry-after"])
    assert 1 <= retry_after <= 60
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A, PROFILE_B]
    assert (await emulated_pool.store.get_profile_state(PROFILE_A)).state is OpenAIProfileState.COOLDOWN
    assert (await emulated_pool.store.get_profile_state(PROFILE_B)).state is OpenAIProfileState.COOLDOWN


async def test_all_invalid_oauth_profiles_return_neutral_503(emulated_pool: EmulatedPool) -> None:
    emulated_pool.emulator.enqueue(PROFILE_A, EmulatedOpenAIResponse(status_code=401))
    emulated_pool.emulator.enqueue(PROFILE_B, EmulatedOpenAIResponse(status_code=403))
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=3)

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await router.acompletion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_hash": _user_hash(50)},
        )

    error = exc_info.value
    assert error.status_code == 503
    assert PROFILE_A not in str(error)
    assert PROFILE_B not in str(error)
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A, PROFILE_B]
    assert (await emulated_pool.store.get_profile_state(PROFILE_A)).state is OpenAIProfileState.DISABLED
    assert (await emulated_pool.store.get_profile_state(PROFILE_B)).state is OpenAIProfileState.DISABLED


@pytest.mark.parametrize(
    "status_code,expected_exception",
    [(404, NotFoundError), (500, InternalServerError)],
)
async def test_non_profile_http_errors_do_not_poison_subscription(
    emulated_pool: EmulatedPool,
    status_code: int,
    expected_exception: type[Exception],
) -> None:
    emulated_pool.emulator.enqueue(PROFILE_A, EmulatedOpenAIResponse(status_code=status_code))
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0)

    with pytest.raises(expected_exception):
        await router.acompletion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_hash": _user_hash(60 + status_code)},
        )

    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A]
    snapshot = await emulated_pool.store.get_profile_state(PROFILE_A)
    assert snapshot.state is OpenAIProfileState.AVAILABLE
    assert snapshot.reason_code is None


@pytest.mark.parametrize(
    "response,timeout,expected_exception",
    [
        (EmulatedOpenAIResponse(close_connection=True), 1, InternalServerError),
        (EmulatedOpenAIResponse(delay_seconds=0.15), 0.03, Timeout),
    ],
)
async def test_transport_errors_do_not_poison_subscription(
    emulated_pool: EmulatedPool,
    response: EmulatedOpenAIResponse,
    timeout: float,
    expected_exception: type[Exception],
) -> None:
    emulated_pool.emulator.enqueue(PROFILE_A, response)
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0, timeout=timeout)

    with pytest.raises(expected_exception):
        await router.acompletion(
            model="gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            metadata={"user_api_key_hash": _user_hash(70)},
        )

    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A]
    snapshot = await emulated_pool.store.get_profile_state(PROFILE_A)
    assert snapshot.state is OpenAIProfileState.AVAILABLE
    assert snapshot.reason_code is None


async def test_expired_cooldown_recovers_through_real_half_open_probe(emulated_pool: EmulatedPool) -> None:
    now = int(time.time())
    await emulated_pool.store.mark_profile_cooldown(PROFILE_A, reset_at=now - 1, reason_code="rate_limit")
    await emulated_pool.store.mark_profile_cooldown(PROFILE_B, reset_at=now + 300, reason_code="rate_limit")
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0)
    user_hash = _user_hash(80)

    response = await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "hello"}],
        metadata={"user_api_key_hash": user_hash},
    )

    assert response.choices[0].message.content == "emulated response"
    assert [request.profile for request in emulated_pool.emulator.requests] == [PROFILE_A]
    snapshot = await emulated_pool.store.get_profile_state(PROFILE_A)
    deadline = time.monotonic() + 3
    while snapshot.state is not OpenAIProfileState.AVAILABLE and time.monotonic() < deadline:
        await asyncio.sleep(0.01)
        snapshot = await emulated_pool.store.get_profile_state(PROFILE_A)
    assert snapshot.state is OpenAIProfileState.AVAILABLE
    assert await emulated_pool.store.get_profile(user_hash) == PROFILE_A


async def test_emulator_records_and_logs_do_not_expose_oauth_secrets(
    emulated_pool: EmulatedPool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO)
    router, _ = emulated_pool.make_router(models=("gpt-5.4",), num_retries=0)

    await router.acompletion(
        model="gpt-5.4",
        messages=[{"role": "user", "content": "confidential prompt"}],
        metadata={"user_api_key_hash": _user_hash(90)},
    )

    safe_record = f"{emulated_pool.emulator.requests!r}\n{caplog.text}"
    assert PROFILE_A in safe_record
    assert "confidential prompt" not in safe_record
    for auth_file in emulated_pool.auth_files.values():
        auth = json.loads(auth_file.read_text(encoding="utf-8"))
        assert auth["access_token"] not in safe_record
        assert auth["account_id"] not in safe_record
