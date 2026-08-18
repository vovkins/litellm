from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
)

USER_KEY_HASH = "a" * 64
PROFILE_ID = "openai-oauth-1"


def make_store(response: object = None) -> tuple[OpenAISubscriptionAffinityStore, AsyncMock]:
    script = AsyncMock(return_value=response)
    redis_cache = MagicMock()
    redis_cache.async_register_script.return_value = script
    cache = DualCache(redis_cache=redis_cache)
    return OpenAISubscriptionAffinityStore(cache), script


def test_cache_key_is_global_and_contains_only_the_existing_hash() -> None:
    cache_key = OpenAISubscriptionAffinityStore.get_cache_key(USER_KEY_HASH.upper())

    assert cache_key == f"openai_subscription_affinity:v1:{USER_KEY_HASH}"
    assert "model" not in cache_key


@pytest.mark.parametrize("invalid_hash", ["", "sk-raw-secret", "a" * 63, "g" * 64])
def test_cache_key_rejects_values_that_are_not_sha256_hashes(invalid_hash: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        OpenAISubscriptionAffinityStore.get_cache_key(invalid_hash)


@pytest.mark.asyncio
async def test_get_profile_reads_from_shared_redis_script() -> None:
    store, script = make_store([b"found", PROFILE_ID.encode(), b"86399"])

    assert await store.get_profile(USER_KEY_HASH) == PROFILE_ID
    script.assert_awaited_once_with(
        keys=(f"openai_subscription_affinity:v1:{USER_KEY_HASH}",),
        args=("read",),
        client=None,
    )


@pytest.mark.asyncio
async def test_get_profile_returns_none_for_missing_binding() -> None:
    store, _ = make_store([b"missing"])

    assert await store.get_profile(USER_KEY_HASH) is None


@pytest.mark.asyncio
async def test_set_profile_writes_safe_identifier_with_explicit_ttl() -> None:
    store, script = make_store([b"written"])

    await store.set_profile(USER_KEY_HASH, PROFILE_ID, ttl_seconds=86400)

    script.assert_awaited_once_with(
        keys=(f"openai_subscription_affinity:v1:{USER_KEY_HASH}",),
        args=("write", PROFILE_ID, "86400"),
        client=None,
    )


@pytest.mark.parametrize("invalid_profile", ["", "UPPERCASE", "profile/id", "p" * 65])
@pytest.mark.asyncio
async def test_set_profile_rejects_unsafe_profile_identifiers(invalid_profile: str) -> None:
    store, script = make_store([b"written"])

    with pytest.raises(ValueError, match="profile_id"):
        await store.set_profile(USER_KEY_HASH, invalid_profile, ttl_seconds=86400)
    script.assert_not_awaited()


@pytest.mark.parametrize("invalid_ttl", [0, -1, True, 1.5])
@pytest.mark.asyncio
async def test_set_profile_rejects_invalid_ttl(invalid_ttl: Any) -> None:
    store, script = make_store([b"written"])

    with pytest.raises(ValueError, match="ttl_seconds"):
        await store.set_profile(USER_KEY_HASH, PROFILE_ID, ttl_seconds=invalid_ttl)
    script.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_profile_removes_shared_binding() -> None:
    store, script = make_store([b"deleted", b"1"])

    await store.delete_profile(USER_KEY_HASH)

    script.assert_awaited_once_with(
        keys=(f"openai_subscription_affinity:v1:{USER_KEY_HASH}",),
        args=("delete",),
        client=None,
    )


@pytest.mark.asyncio
async def test_get_remaining_ttl_uses_same_atomic_snapshot() -> None:
    store, _ = make_store([b"found", PROFILE_ID.encode(), b"321"])

    assert await store.get_remaining_ttl(USER_KEY_HASH) == 321


@pytest.mark.asyncio
async def test_get_remaining_ttl_returns_none_for_missing_binding() -> None:
    store, _ = make_store([b"missing"])

    assert await store.get_remaining_ttl(USER_KEY_HASH) is None


@pytest.mark.parametrize(
    "response",
    [
        None,
        [b"unexpected"],
        [b"found", b"UPPERCASE", b"60"],
        [b"found", PROFILE_ID.encode(), b"-1"],
        [b"found", PROFILE_ID.encode(), b"not-a-ttl"],
    ],
)
@pytest.mark.asyncio
async def test_malformed_redis_data_fails_closed(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.get_profile(USER_KEY_HASH)


@pytest.mark.asyncio
async def test_missing_redis_fails_instead_of_using_process_local_cache() -> None:
    store = OpenAISubscriptionAffinityStore(DualCache())

    with pytest.raises(OpenAISubscriptionAffinityStoreError, match="requires Redis"):
        await store.get_profile(USER_KEY_HASH)


@pytest.mark.asyncio
async def test_redis_failure_is_wrapped_without_sensitive_values() -> None:
    store, script = make_store()
    script.side_effect = ConnectionError("backend unavailable")

    with pytest.raises(OpenAISubscriptionAffinityStoreError) as exc_info:
        await store.set_profile(USER_KEY_HASH, PROFILE_ID, ttl_seconds=86400)

    message = str(exc_info.value)
    assert USER_KEY_HASH not in message
    assert PROFILE_ID not in message
    assert "backend unavailable" not in message


@pytest.mark.asyncio
async def test_redis_can_be_attached_after_store_construction() -> None:
    cache = DualCache()
    store = OpenAISubscriptionAffinityStore(cache)
    script = AsyncMock(return_value=[b"found", PROFILE_ID.encode(), b"42"])
    redis_cache = MagicMock()
    redis_cache.async_register_script.return_value = script
    cache.attach_redis_cache(redis_cache)

    assert await store.get_profile(USER_KEY_HASH) == PROFILE_ID


def test_store_does_not_change_existing_deployment_affinity_namespace() -> None:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    assert DeploymentAffinityCheck.CACHE_KEY_PREFIX == "deployment_affinity:v1"
    assert OpenAISubscriptionAffinityStore.CACHE_KEY_PREFIX != DeploymentAffinityCheck.CACHE_KEY_PREFIX
