from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.exceptions import ServiceUnavailableError
from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
)
from litellm.router_utils.pre_call_checks.openai_subscription_affinity_check import (
    OpenAISubscriptionAffinityCheck,
)

USER_KEY_HASH = "a" * 64
PROFILE_ID = "openai-oauth-1"
SECOND_PROFILE_ID = "openai-oauth-2"


def make_callback() -> tuple[OpenAISubscriptionAffinityCheck, AsyncMock]:
    store = MagicMock(spec=OpenAISubscriptionAffinityStore)
    refresh = store.refresh_profile_if_current
    callback = OpenAISubscriptionAffinityCheck(store=store)
    return callback, refresh


def make_routing_callback() -> tuple[OpenAISubscriptionAffinityCheck, MagicMock]:
    store = MagicMock(spec=OpenAISubscriptionAffinityStore)
    callback = OpenAISubscriptionAffinityCheck(store=store)
    return callback, store


def make_deployment(model: str, deployment_id: str, profile_id: object = None) -> dict:
    model_info = {"id": deployment_id}
    if profile_id is not None:
        model_info["openai_oauth_profile"] = profile_id
    return {
        "model_name": model,
        "litellm_params": {"model": f"chatgpt/{model}"},
        "model_info": model_info,
    }


def make_request_kwargs(user_api_key_hash: object = USER_KEY_HASH) -> dict:
    return {"metadata": {"user_api_key_hash": user_api_key_hash}}


@pytest.mark.asyncio
async def test_routing_filters_oauth_deployments_to_assigned_profile() -> None:
    callback, store = make_routing_callback()
    store.get_or_assign_profile.return_value = SECOND_PROFILE_ID
    deployments = [
        make_deployment("gpt-5.4", "deployment-a", PROFILE_ID),
        make_deployment("gpt-5.4", "deployment-b", SECOND_PROFILE_ID),
    ]

    filtered = await callback.async_filter_deployments(
        model="gpt-5.4",
        healthy_deployments=deployments,
        messages=None,
        request_kwargs=make_request_kwargs(),
    )

    assert filtered == [deployments[1]]
    store.get_or_assign_profile.assert_awaited_once_with(
        user_api_key_hash=USER_KEY_HASH,
        available_profile_ids=[PROFILE_ID, SECOND_PROFILE_ID],
        ttl_seconds=OpenAISubscriptionAffinityStore.DEFAULT_TTL_SECONDS,
    )


@pytest.mark.asyncio
async def test_one_binding_is_used_across_openai_model_groups() -> None:
    callback, store = make_routing_callback()
    store.get_or_assign_profile.return_value = PROFILE_ID
    gpt_deployments = [
        make_deployment("gpt-5.4", "gpt-a", PROFILE_ID),
        make_deployment("gpt-5.4", "gpt-b", SECOND_PROFILE_ID),
    ]
    codex_deployments = [
        make_deployment("gpt-5.3-codex", "codex-a", PROFILE_ID),
        make_deployment("gpt-5.3-codex", "codex-b", SECOND_PROFILE_ID),
    ]

    first = await callback.async_filter_deployments("gpt-5.4", gpt_deployments, None, make_request_kwargs())
    second = await callback.async_filter_deployments("gpt-5.3-codex", codex_deployments, None, make_request_kwargs())

    assert first == [gpt_deployments[0]]
    assert second == [codex_deployments[0]]
    assert [call.kwargs["user_api_key_hash"] for call in store.get_or_assign_profile.await_args_list] == [
        USER_KEY_HASH,
        USER_KEY_HASH,
    ]


@pytest.mark.asyncio
async def test_glm_deployments_are_not_processed_by_openai_affinity() -> None:
    callback, store = make_routing_callback()
    deployments = [
        make_deployment("glm-5.2", "glm-primary"),
        make_deployment("glm-5.2", "glm-secondary"),
    ]

    filtered = await callback.async_filter_deployments("glm-5.2", deployments, None, make_request_kwargs())

    assert filtered == deployments
    store.get_or_assign_profile.assert_not_awaited()


@pytest.mark.parametrize(
    "request_kwargs",
    [None, {}, {"metadata": {}}, make_request_kwargs(None)],
)
@pytest.mark.asyncio
async def test_oauth_routing_requires_virtual_key_hash(request_kwargs: dict | None) -> None:
    callback, store = make_routing_callback()
    deployments = [make_deployment("gpt-5.4", "deployment-a", PROFILE_ID)]

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await callback.async_filter_deployments("gpt-5.4", deployments, None, request_kwargs)

    assert exc_info.value.status_code == 503
    assert PROFILE_ID not in str(exc_info.value)
    store.get_or_assign_profile.assert_not_awaited()


@pytest.mark.asyncio
async def test_responses_metadata_can_supply_virtual_key_hash() -> None:
    callback, store = make_routing_callback()
    store.get_or_assign_profile.return_value = PROFILE_ID
    deployments = [make_deployment("gpt-5.4", "deployment-a", PROFILE_ID)]

    filtered = await callback.async_filter_deployments(
        "gpt-5.4",
        deployments,
        None,
        {"litellm_metadata": {"user_api_key_hash": USER_KEY_HASH}},
    )

    assert filtered == deployments


@pytest.mark.parametrize(
    "deployments",
    [
        [
            make_deployment("gpt-5.4", "oauth", PROFILE_ID),
            make_deployment("gpt-5.4", "unmarked"),
        ],
        [make_deployment("gpt-5.4", "invalid", 123)],
    ],
)
@pytest.mark.asyncio
async def test_invalid_oauth_model_group_fails_closed(deployments: list[dict]) -> None:
    callback, store = make_routing_callback()

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await callback.async_filter_deployments("gpt-5.4", deployments, None, make_request_kwargs())

    assert exc_info.value.status_code == 503
    store.get_or_assign_profile.assert_not_awaited()


@pytest.mark.parametrize(
    "side_effect",
    [
        OpenAISubscriptionAffinityStoreError("redis unavailable"),
        ValueError("invalid hash"),
    ],
)
@pytest.mark.asyncio
async def test_shared_state_failure_returns_neutral_503(side_effect: Exception) -> None:
    callback, store = make_routing_callback()
    store.get_or_assign_profile.side_effect = side_effect
    deployments = [make_deployment("gpt-5.4", "deployment-a", PROFILE_ID)]

    with pytest.raises(ServiceUnavailableError) as exc_info:
        await callback.async_filter_deployments("gpt-5.4", deployments, None, make_request_kwargs())

    message = str(exc_info.value)
    assert exc_info.value.status_code == 503
    assert USER_KEY_HASH not in message
    assert PROFILE_ID not in message
    assert "redis" not in message


@pytest.mark.asyncio
async def test_unhealthy_bound_profile_does_not_fall_back_randomly() -> None:
    callback, store = make_routing_callback()
    store.get_or_assign_profile.return_value = SECOND_PROFILE_ID
    deployments = [make_deployment("gpt-5.4", "deployment-a", PROFILE_ID)]

    with pytest.raises(ServiceUnavailableError):
        await callback.async_filter_deployments("gpt-5.4", deployments, None, make_request_kwargs())


@pytest.mark.asyncio
async def test_router_registers_and_separates_openai_and_glm_affinity() -> None:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    original_callbacks = list(litellm.callbacks)
    litellm.callbacks = []
    router = None
    try:
        glm_deployment = make_deployment("glm-5.2", "glm-primary")
        glm_deployment["litellm_params"] = {
            "model": "openai/gpt-4",
            "api_key": "mock-key",
        }
        router = litellm.Router(
            model_list=[glm_deployment],
            optional_pre_call_checks=["openai_subscription_affinity"],
            model_group_affinity_config={"glm-5.2": ["deployment_affinity"]},
            deployment_affinity_ttl_seconds=86400,
        )
        router.add_optional_pre_call_checks(["openai_subscription_affinity"])

        callbacks = router.optional_callbacks or []
        openai_callbacks = [callback for callback in callbacks if isinstance(callback, OpenAISubscriptionAffinityCheck)]
        deployment_callback = next(callback for callback in callbacks if isinstance(callback, DeploymentAffinityCheck))

        assert len(openai_callbacks) == 1
        assert callbacks.index(openai_callbacks[0]) < callbacks.index(deployment_callback)
        assert litellm.callbacks.index(openai_callbacks[0]) < litellm.callbacks.index(deployment_callback)
        assert deployment_callback.enable_user_key_affinity is False
        assert deployment_callback._get_effective_flags("glm-5.2") == (True, False, False)
        assert deployment_callback._get_effective_flags("gpt-5.4") == (False, False, False)

        openai_callback = openai_callbacks[0]
        openai_callback.store.get_or_assign_profile = AsyncMock(return_value=SECOND_PROFILE_ID)
        oauth_deployments = [
            make_deployment("gpt-5.4", "oauth-a", PROFILE_ID),
            make_deployment("gpt-5.4", "oauth-b", SECOND_PROFILE_ID),
        ]
        glm_deployments = [
            make_deployment("glm-5.2", "glm-a"),
            make_deployment("glm-5.2", "glm-b"),
        ]

        filtered_oauth = await router.async_callback_filter_deployments(
            model="gpt-5.4",
            healthy_deployments=oauth_deployments,
            messages=None,
            parent_otel_span=None,
            request_kwargs=make_request_kwargs(),
        )
        filtered_glm = await router.async_callback_filter_deployments(
            model="glm-5.2",
            healthy_deployments=glm_deployments,
            messages=None,
            parent_otel_span=None,
            request_kwargs=make_request_kwargs(),
        )

        assert filtered_oauth == [oauth_deployments[1]]
        assert filtered_glm == glm_deployments
        openai_callback.store.get_or_assign_profile.assert_awaited_once()
    finally:
        if router is not None:
            router.discard()
        litellm.callbacks = original_callbacks


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
