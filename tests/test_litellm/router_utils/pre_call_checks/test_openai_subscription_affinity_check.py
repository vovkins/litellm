from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
)
from litellm.router_utils.pre_call_checks.openai_subscription_affinity_check import (
    OpenAISubscriptionAffinityCheck,
)

USER_KEY_HASH = "a" * 64
PROFILE_ID = "openai-oauth-1"


def make_callback() -> tuple[OpenAISubscriptionAffinityCheck, AsyncMock]:
    store = MagicMock(spec=OpenAISubscriptionAffinityStore)
    refresh = store.refresh_profile_if_current
    callback = OpenAISubscriptionAffinityCheck(store=store)
    return callback, refresh


def make_success_kwargs(
    *,
    user_api_key_hash: object = USER_KEY_HASH,
    profile_id: object = PROFILE_ID,
) -> dict:
    return {
        "standard_logging_object": {
            "metadata": {"user_api_key_hash": user_api_key_hash},
        },
        "litellm_params": {
            "model_info": {"openai_oauth_profile": profile_id},
        },
    }


@pytest.mark.asyncio
async def test_success_refreshes_selected_profile_binding() -> None:
    callback, refresh = make_callback()
    refresh.return_value = True

    await callback.async_log_success_event(make_success_kwargs(), {}, 0, 1)

    refresh.assert_awaited_once_with(
        user_api_key_hash=USER_KEY_HASH,
        profile_id=PROFILE_ID,
        ttl_seconds=OpenAISubscriptionAffinityStore.DEFAULT_TTL_SECONDS,
    )


@pytest.mark.asyncio
async def test_non_openai_deployment_is_ignored() -> None:
    callback, refresh = make_callback()
    kwargs = make_success_kwargs()
    kwargs["litellm_params"]["model_info"] = {"id": "glm-deployment"}

    await callback.async_log_success_event(kwargs, {}, 0, 1)

    refresh.assert_not_awaited()


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"standard_logging_object": {}, "litellm_params": {}},
        make_success_kwargs(user_api_key_hash=None),
        make_success_kwargs(profile_id=None),
        make_success_kwargs(user_api_key_hash=123),
        make_success_kwargs(profile_id=123),
    ],
)
@pytest.mark.asyncio
async def test_incomplete_or_untrusted_success_metadata_is_ignored(kwargs: dict) -> None:
    callback, refresh = make_callback()

    await callback.async_log_success_event(kwargs, {}, 0, 1)

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_request_metadata_cannot_spoof_selected_profile() -> None:
    callback, refresh = make_callback()
    kwargs = make_success_kwargs()
    kwargs["litellm_params"] = {
        "metadata": {"openai_oauth_profile": "attacker-profile"},
    }

    await callback.async_log_success_event(kwargs, {}, 0, 1)

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_storage_failure_after_success_is_logged_without_sensitive_values(
    caplog: pytest.LogCaptureFixture,
) -> None:
    callback, refresh = make_callback()
    refresh.side_effect = OpenAISubscriptionAffinityStoreError(f"backend failed for {USER_KEY_HASH} and {PROFILE_ID}")

    with caplog.at_level(logging.ERROR):
        await callback.async_log_success_event(make_success_kwargs(), {}, 0, 1)

    assert "TTL refresh failed after a successful request" in caplog.text
    assert USER_KEY_HASH not in caplog.text
    assert PROFILE_ID not in caplog.text


@pytest.mark.asyncio
async def test_failure_event_does_not_refresh_binding() -> None:
    callback, refresh = make_callback()

    await callback.async_log_failure_event(make_success_kwargs(), {}, 0, 1)

    refresh.assert_not_awaited()


@pytest.mark.asyncio
async def test_stream_refreshes_only_after_final_success_event() -> None:
    callback, refresh = make_callback()
    kwargs = make_success_kwargs()

    await callback.async_log_stream_event(kwargs, {"delta": "partial"}, 0, 1)
    refresh.assert_not_awaited()

    await callback.async_log_success_event(kwargs, {"complete": True}, 0, 1)
    refresh.assert_awaited_once()
