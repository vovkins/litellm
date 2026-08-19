"""Redis integration tests for OpenAI subscription affinity.

Run explicitly against a disposable Redis instance:

    REDIS_HOST=127.0.0.1 REDIS_PORT=6379 pytest -q \
        tests/test_litellm/router_utils/test_openai_subscription_affinity_redis.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
from collections import Counter

import pytest
import pytest_asyncio

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
)
from litellm.router_utils.pre_call_checks.openai_subscription_affinity_check import (
    OpenAISubscriptionAffinityCheck,
)

REDIS_HOST = os.getenv("REDIS_HOST")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
pytestmark = pytest.mark.skipif(REDIS_HOST is None, reason="requires a disposable Redis instance")


def _user_hash(index: int) -> str:
    return hashlib.sha256(f"virtual-key-{index}".encode()).hexdigest()


@pytest_asyncio.fixture
async def affinity_store() -> OpenAISubscriptionAffinityStore:
    redis_cache = RedisCache(host=REDIS_HOST, port=REDIS_PORT)
    client = redis_cache.init_async_client()
    await client.flushdb()
    try:
        yield OpenAISubscriptionAffinityStore(DualCache(redis_cache=redis_cache))
    finally:
        await client.flushdb()


@pytest.mark.asyncio
async def test_assignments_are_evenly_distributed(affinity_store: OpenAISubscriptionAffinityStore) -> None:
    profiles = ["subscription-c", "subscription-a", "subscription-b"]

    assigned = [
        await affinity_store.get_or_assign_profile(_user_hash(index), profiles, ttl_seconds=60) for index in range(99)
    ]

    assert Counter(assigned) == {
        "subscription-a": 33,
        "subscription-b": 33,
        "subscription-c": 33,
    }


@pytest.mark.asyncio
async def test_parallel_first_requests_converge_on_one_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(1)

    assigned = await asyncio.gather(
        *(
            affinity_store.get_or_assign_profile(
                user_hash,
                ["subscription-a", "subscription-b"],
                ttl_seconds=60,
            )
            for _ in range(64)
        )
    )

    assert set(assigned) == {"subscription-a"}
    assert await affinity_store.get_remaining_ttl(user_hash) in range(1, 61)
    assert (
        await affinity_store.get_or_assign_profile(
            _user_hash(2),
            ["subscription-a", "subscription-b"],
            ttl_seconds=60,
        )
        == "subscription-b"
    )


@pytest.mark.asyncio
async def test_existing_assignment_and_ttl_are_not_changed(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(2)
    original = await affinity_store.get_or_assign_profile(
        user_hash,
        ["subscription-a", "subscription-b"],
        ttl_seconds=5,
    )
    ttl_before = await affinity_store.get_remaining_ttl(user_hash)

    existing = await affinity_store.get_or_assign_profile(
        user_hash,
        ["subscription-c", "subscription-d"],
        ttl_seconds=60,
    )
    ttl_after = await affinity_store.get_remaining_ttl(user_hash)

    assert existing == original
    assert ttl_before is not None
    assert ttl_after is not None
    assert ttl_after <= ttl_before
    assert ttl_after < 60


@pytest.mark.asyncio
async def test_expired_binding_gets_next_round_robin_assignment(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(3)

    first = await affinity_store.get_or_assign_profile(
        user_hash,
        ["subscription-a", "subscription-b"],
        ttl_seconds=1,
    )
    await asyncio.sleep(1.1)
    second = await affinity_store.get_or_assign_profile(
        user_hash,
        ["subscription-a", "subscription-b"],
        ttl_seconds=60,
    )

    assert first == "subscription-a"
    assert second == "subscription-b"


@pytest.mark.asyncio
async def test_adding_profile_preserves_active_bindings_and_gradually_rebalances(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    original_hashes = [_user_hash(index) for index in range(4)]
    original_profiles = {
        user_hash: await affinity_store.get_or_assign_profile(
            user_hash,
            ["subscription-a", "subscription-b"],
            ttl_seconds=300,
        )
        for user_hash in original_hashes
    }
    original_ttls = {user_hash: await affinity_store.get_remaining_ttl(user_hash) for user_hash in original_hashes}

    restarted_store = OpenAISubscriptionAffinityStore(
        DualCache(redis_cache=RedisCache(host=REDIS_HOST, port=REDIS_PORT))
    )
    preserved_profiles = {
        user_hash: await restarted_store.get_or_assign_profile(
            user_hash,
            ["subscription-c", "subscription-b", "subscription-a"],
            ttl_seconds=3600,
        )
        for user_hash in original_hashes
    }
    preserved_ttls = {user_hash: await restarted_store.get_remaining_ttl(user_hash) for user_hash in original_hashes}

    assert preserved_profiles == original_profiles
    for user_hash in original_hashes:
        assert original_ttls[user_hash] is not None
        assert preserved_ttls[user_hash] is not None
        assert preserved_ttls[user_hash] <= original_ttls[user_hash]
        assert preserved_ttls[user_hash] < 3600

    new_assignments = [
        await restarted_store.get_or_assign_profile(
            _user_hash(index),
            ["subscription-c", "subscription-a", "subscription-b"],
            ttl_seconds=300,
        )
        for index in range(4, 10)
    ]
    assert new_assignments == [
        "subscription-b",
        "subscription-c",
        "subscription-a",
        "subscription-b",
        "subscription-c",
        "subscription-a",
    ]


@pytest.mark.asyncio
async def test_expired_binding_can_move_to_newly_added_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    expiring_hash = _user_hash(20)
    assert (
        await affinity_store.get_or_assign_profile(
            expiring_hash,
            ["subscription-a", "subscription-b"],
            ttl_seconds=1,
        )
        == "subscription-a"
    )
    assert (
        await affinity_store.get_or_assign_profile(
            _user_hash(21),
            ["subscription-a", "subscription-b"],
            ttl_seconds=60,
        )
        == "subscription-b"
    )

    await asyncio.sleep(1.1)

    assert (
        await affinity_store.get_or_assign_profile(
            expiring_hash,
            ["subscription-c", "subscription-b", "subscription-a"],
            ttl_seconds=60,
        )
        == "subscription-c"
    )


@pytest.mark.asyncio
async def test_success_refresh_extends_only_matching_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(30)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=5)

    assert (
        await affinity_store.refresh_profile_if_current(
            user_hash,
            "subscription-a",
            ttl_seconds=60,
        )
        is True
    )
    assert await affinity_store.get_profile(user_hash) == "subscription-a"
    assert await affinity_store.get_remaining_ttl(user_hash) in range(59, 61)


@pytest.mark.asyncio
async def test_success_refresh_does_not_change_mismatched_profile_or_ttl(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(31)
    await affinity_store.set_profile(user_hash, "subscription-b", ttl_seconds=5)
    ttl_before = await affinity_store.get_remaining_ttl(user_hash)

    assert (
        await affinity_store.refresh_profile_if_current(
            user_hash,
            "subscription-a",
            ttl_seconds=60,
        )
        is False
    )
    ttl_after = await affinity_store.get_remaining_ttl(user_hash)

    assert await affinity_store.get_profile(user_hash) == "subscription-b"
    assert ttl_before is not None
    assert ttl_after is not None
    assert ttl_after <= ttl_before
    assert ttl_after < 60


@pytest.mark.asyncio
async def test_success_refresh_does_not_recreate_missing_binding(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(32)

    assert (
        await affinity_store.refresh_profile_if_current(
            user_hash,
            "subscription-a",
            ttl_seconds=60,
        )
        is False
    )
    assert await affinity_store.get_profile(user_hash) is None


@pytest.mark.asyncio
async def test_delayed_success_cannot_extend_newer_profile_binding(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(33)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=60)

    await affinity_store.set_profile(user_hash, "subscription-b", ttl_seconds=5)
    assert (
        await affinity_store.refresh_profile_if_current(
            user_hash,
            "subscription-a",
            ttl_seconds=300,
        )
        is False
    )

    assert await affinity_store.get_profile(user_hash) == "subscription-b"
    assert await affinity_store.get_remaining_ttl(user_hash) in range(1, 6)


@pytest.mark.asyncio
async def test_profile_binding_is_shared_across_model_groups(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    callback = OpenAISubscriptionAffinityCheck(store=affinity_store, ttl_seconds=60)
    user_hash = _user_hash(40)
    gpt_deployments = [
        {"model_info": {"id": "gpt-a", "openai_oauth_profile": "subscription-a"}},
        {"model_info": {"id": "gpt-b", "openai_oauth_profile": "subscription-b"}},
    ]
    codex_deployments = [
        {"model_info": {"id": "codex-a", "openai_oauth_profile": "subscription-a"}},
        {"model_info": {"id": "codex-b", "openai_oauth_profile": "subscription-b"}},
    ]
    request_kwargs = {"metadata": {"user_api_key_hash": user_hash}}

    first = await callback.async_filter_deployments("gpt-5.4", gpt_deployments, None, request_kwargs)
    second = await callback.async_filter_deployments("gpt-5.3-codex", codex_deployments, None, request_kwargs)

    assert first == [gpt_deployments[0]]
    assert second == [codex_deployments[0]]
    assert await affinity_store.get_profile(user_hash) == "subscription-a"
