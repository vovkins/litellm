"""Redis integration tests for OpenAI subscription affinity.

Run explicitly against a disposable Redis instance:

    REDIS_HOST=127.0.0.1 REDIS_PORT=6379 pytest -q \
        tests/test_litellm/router_utils/test_openai_subscription_affinity_redis.py
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections import Counter

import pytest
import pytest_asyncio

from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.router_utils.openai_subscription_affinity import (
    OpenAIProfileFailureUpdateStatus,
    OpenAIProfileSelectionStatus,
    OpenAIProfileState,
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
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
        ["subscription-a", "subscription-c", "subscription-d"],
        ttl_seconds=60,
    )
    ttl_after = await affinity_store.get_remaining_ttl(user_hash)

    assert existing == original
    assert ttl_before is not None
    assert ttl_after is not None
    assert ttl_after <= ttl_before
    assert ttl_after < 60


@pytest.mark.asyncio
async def test_binding_to_removed_profile_is_reassigned(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(4)
    await affinity_store.set_profile(user_hash, "subscription-removed", ttl_seconds=300)

    selection = await affinity_store.select_available_profile(
        user_hash,
        ["subscription-a", "subscription-b"],
        ttl_seconds=60,
    )

    assert selection.status is OpenAIProfileSelectionStatus.REASSIGNED
    assert selection.profile_id in {"subscription-a", "subscription-b"}
    assert await affinity_store.get_profile(user_hash) == selection.profile_id


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
async def test_assignment_excludes_unavailable_states_and_replaces_bound_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(22)
    await affinity_store.set_profile(user_hash, "subscription-b", ttl_seconds=300)
    await affinity_store.mark_profile_cooldown(
        "subscription-b",
        reset_at=int(time.time()) + 300,
        reason_code="rate_limit",
    )
    await affinity_store.disable_profile("subscription-c", "oauth_error")

    selection = await affinity_store.select_available_profile(
        user_hash,
        ["subscription-a", "subscription-b", "subscription-c"],
        ttl_seconds=60,
    )

    assert selection.status is OpenAIProfileSelectionStatus.REASSIGNED
    assert selection.profile_id == "subscription-a"
    assert await affinity_store.get_profile(user_hash) == "subscription-a"
    assert await affinity_store.get_remaining_ttl(user_hash) in range(1, 61)


@pytest.mark.asyncio
async def test_assignment_round_robin_is_even_across_only_available_profiles(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    await affinity_store.disable_profile("subscription-b", "oauth_error")

    assigned = [
        await affinity_store.get_or_assign_profile(
            _user_hash(index),
            ["subscription-a", "subscription-b", "subscription-c"],
            ttl_seconds=60,
        )
        for index in range(100, 160)
    ]

    assert Counter(assigned) == {"subscription-a": 30, "subscription-c": 30}


@pytest.mark.asyncio
async def test_no_available_profile_preserves_binding_and_counter(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(23)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=300)
    reset_at = int(time.time()) + 300
    await affinity_store.mark_profile_cooldown("subscription-a", reset_at, "rate_limit")
    await affinity_store.disable_profile("subscription-b", "oauth_error")
    redis_cache = affinity_store.cache.redis_cache
    assert redis_cache is not None
    raw_redis_client = redis_cache.init_async_client()

    selection = await affinity_store.select_available_profile(
        user_hash,
        ["subscription-a", "subscription-b"],
        ttl_seconds=60,
    )

    assert selection.status is OpenAIProfileSelectionStatus.UNAVAILABLE
    assert selection.profile_id is None
    assert selection.cooldown_profiles == 1
    assert selection.disabled_profiles == 1
    assert selection.next_recovery_at == reset_at
    assert await affinity_store.get_profile(user_hash) == "subscription-a"
    assert await raw_redis_client.exists(affinity_store.COUNTER_CACHE_KEY) == 0


@pytest.mark.asyncio
async def test_rate_limit_failure_atomically_updates_state_and_releases_binding(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(24)
    reset_at = int(time.time()) + 300
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=300)

    update = await affinity_store.fail_profile_if_current_binding(
        user_hash,
        "subscription-a",
        OpenAIProfileState.COOLDOWN,
        reset_at=reset_at,
        reason_code="rate_limit",
    )

    assert update.status is OpenAIProfileFailureUpdateStatus.APPLIED
    assert update.effective_state is OpenAIProfileState.COOLDOWN
    assert await affinity_store.get_profile(user_hash) is None
    snapshot = await affinity_store.get_profile_state("subscription-a")
    assert snapshot.state is OpenAIProfileState.COOLDOWN
    assert snapshot.reset_at == reset_at
    assert (
        await affinity_store.get_or_assign_profile(
            user_hash,
            ["subscription-a", "subscription-b"],
            ttl_seconds=60,
        )
        == "subscription-b"
    )


@pytest.mark.asyncio
async def test_delayed_failure_cannot_remove_or_poison_new_binding(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(25)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=300)
    await affinity_store.set_profile(user_hash, "subscription-b", ttl_seconds=300)

    update = await affinity_store.fail_profile_if_current_binding(
        user_hash,
        "subscription-a",
        OpenAIProfileState.DISABLED,
        reason_code="oauth_error",
    )

    assert update.status is OpenAIProfileFailureUpdateStatus.STALE
    assert await affinity_store.get_profile(user_hash) == "subscription-b"
    assert (await affinity_store.get_profile_state("subscription-a")).persisted is False


@pytest.mark.asyncio
async def test_parallel_failures_for_one_key_have_one_cas_winner(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(26)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=300)
    reset_at = int(time.time()) + 300

    updates = await asyncio.gather(
        *(
            affinity_store.fail_profile_if_current_binding(
                user_hash,
                "subscription-a",
                OpenAIProfileState.COOLDOWN,
                reset_at=reset_at,
                reason_code="rate_limit",
            )
            for _ in range(64)
        )
    )

    assert Counter(update.status for update in updates) == {
        OpenAIProfileFailureUpdateStatus.APPLIED: 1,
        OpenAIProfileFailureUpdateStatus.STALE: 63,
    }
    assert await affinity_store.get_profile(user_hash) is None
    assert (await affinity_store.get_profile_state("subscription-a")).state is OpenAIProfileState.COOLDOWN


@pytest.mark.asyncio
async def test_many_keys_fail_over_evenly_after_shared_profile_failure(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hashes = [_user_hash(index) for index in range(200, 260)]
    await asyncio.gather(
        *(affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=300) for user_hash in user_hashes)
    )
    reset_at = int(time.time()) + 300

    updates = await asyncio.gather(
        *(
            affinity_store.fail_profile_if_current_binding(
                user_hash,
                "subscription-a",
                OpenAIProfileState.COOLDOWN,
                reset_at=reset_at,
                reason_code="rate_limit",
            )
            for user_hash in user_hashes
        )
    )
    reassigned = await asyncio.gather(
        *(
            affinity_store.get_or_assign_profile(
                user_hash,
                ["subscription-a", "subscription-b", "subscription-c"],
                ttl_seconds=60,
            )
            for user_hash in user_hashes
        )
    )

    assert all(update.status is OpenAIProfileFailureUpdateStatus.APPLIED for update in updates)
    assert Counter(reassigned) == {"subscription-b": 30, "subscription-c": 30}


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


@pytest.mark.asyncio
async def test_failure_callback_makes_next_routing_attempt_use_another_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    callback = OpenAISubscriptionAffinityCheck(store=affinity_store, ttl_seconds=60)
    user_hash = _user_hash(41)
    deployments = [
        {"model_info": {"id": "gpt-a", "openai_oauth_profile": "subscription-a"}},
        {"model_info": {"id": "gpt-b", "openai_oauth_profile": "subscription-b"}},
    ]
    request_kwargs = {"metadata": {"user_api_key_hash": user_hash}}
    first = await callback.async_filter_deployments("gpt-5.4", deployments, None, request_kwargs)
    failed_profile = first[0]["model_info"]["openai_oauth_profile"]
    error = RuntimeError("provider response must not be inspected")
    error.status_code = 429  # type: ignore[attr-defined]

    await callback.async_log_failure_event(
        {
            "exception": error,
            "standard_logging_object": {"metadata": {"user_api_key_hash": user_hash}},
            "litellm_params": {"model_info": {"openai_oauth_profile": failed_profile}},
        },
        None,
        0,
        1,
    )
    second = await callback.async_filter_deployments("gpt-5.4", deployments, None, request_kwargs)

    assert second[0]["model_info"]["openai_oauth_profile"] != failed_profile
    assert (await affinity_store.get_profile_state(failed_profile)).state is OpenAIProfileState.COOLDOWN


@pytest.mark.asyncio
async def test_unknown_profile_state_is_available_without_redis_write(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    state_key = affinity_store.get_profile_state_cache_key(profile_id)
    redis_cache = affinity_store.cache.redis_cache
    assert redis_cache is not None
    raw_redis_client = redis_cache.init_async_client()

    snapshot = await affinity_store.get_profile_state(profile_id)

    assert snapshot.state is OpenAIProfileState.AVAILABLE
    assert snapshot.persisted is False
    assert await raw_redis_client.exists(state_key) == 0


@pytest.mark.asyncio
async def test_profile_state_survives_store_restart(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    await affinity_store.disable_profile("subscription-a", "auth_error")

    restarted_store = OpenAISubscriptionAffinityStore(
        DualCache(redis_cache=RedisCache(host=REDIS_HOST, port=REDIS_PORT))
    )
    snapshot = await restarted_store.get_profile_state("subscription-a")

    assert snapshot.state is OpenAIProfileState.DISABLED
    assert snapshot.reason_code == "auth_error"
    assert snapshot.persisted is True


@pytest.mark.asyncio
async def test_profile_state_transitions_clear_obsolete_fields(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    await affinity_store.mark_profile_cooldown(profile_id, reset_at=1, reason_code="rate_limit")
    cooldown = await affinity_store.get_profile_state(profile_id)

    lease = await affinity_store.acquire_profile_half_open_lease(profile_id, lease_seconds=30)
    half_open = await affinity_store.get_profile_state(profile_id)
    assert lease is not None
    assert cooldown.state is OpenAIProfileState.COOLDOWN
    assert half_open.state is OpenAIProfileState.HALF_OPEN
    assert half_open.reset_at == 1
    assert half_open.reason_code == "rate_limit"
    assert half_open.lease_expires_at == lease.expires_at

    assert (
        await affinity_store.complete_profile_half_open(
            profile_id,
            lease.token,
            OpenAIProfileState.AVAILABLE,
        )
        is True
    )
    available = await affinity_store.get_profile_state(profile_id)
    assert available.state is OpenAIProfileState.AVAILABLE
    assert available.reset_at is None
    assert available.lease_expires_at is None
    assert available.reason_code is None


@pytest.mark.asyncio
async def test_cooldown_cannot_be_probed_before_reset(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    await affinity_store.mark_profile_cooldown(
        "subscription-a",
        reset_at=int(time.time()) + 60,
        reason_code="rate_limit",
    )

    assert (
        await affinity_store.acquire_profile_half_open_lease(
            "subscription-a",
            lease_seconds=30,
        )
        is None
    )
    assert (await affinity_store.get_profile_state("subscription-a")).state is OpenAIProfileState.COOLDOWN


@pytest.mark.asyncio
async def test_parallel_half_open_acquisition_has_one_winner(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    await affinity_store.mark_profile_cooldown(profile_id, reset_at=1, reason_code="rate_limit")

    leases = await asyncio.gather(
        *(affinity_store.acquire_profile_half_open_lease(profile_id, lease_seconds=30) for _ in range(64))
    )

    winners = [lease for lease in leases if lease is not None]
    assert len(winners) == 1
    snapshot = await affinity_store.get_profile_state(profile_id)
    assert snapshot.state is OpenAIProfileState.HALF_OPEN
    assert snapshot.lease_expires_at == winners[0].expires_at


@pytest.mark.asyncio
async def test_expired_half_open_lease_is_replaced_and_old_owner_is_rejected(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    await affinity_store.mark_profile_cooldown(profile_id, reset_at=1, reason_code="rate_limit")
    old_lease = await affinity_store.acquire_profile_half_open_lease(profile_id, lease_seconds=1)
    assert old_lease is not None

    await asyncio.sleep(1.1)

    new_lease = await affinity_store.acquire_profile_half_open_lease(profile_id, lease_seconds=30)
    assert new_lease is not None
    assert new_lease.token != old_lease.token
    assert (
        await affinity_store.complete_profile_half_open(
            profile_id,
            old_lease.token,
            OpenAIProfileState.AVAILABLE,
        )
        is False
    )
    assert (
        await affinity_store.complete_profile_half_open(
            profile_id,
            new_lease.token,
            OpenAIProfileState.AVAILABLE,
        )
        is True
    )


@pytest.mark.asyncio
async def test_half_open_completion_can_return_profile_to_cooldown(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    await affinity_store.mark_profile_cooldown(profile_id, reset_at=1, reason_code="rate_limit")
    lease = await affinity_store.acquire_profile_half_open_lease(profile_id)
    assert lease is not None
    next_reset = int(time.time()) + 120

    assert (
        await affinity_store.complete_profile_half_open(
            profile_id,
            lease.token,
            OpenAIProfileState.COOLDOWN,
            reset_at=next_reset,
            reason_code="rate_limit",
        )
        is True
    )
    snapshot = await affinity_store.get_profile_state(profile_id)
    assert snapshot.state is OpenAIProfileState.COOLDOWN
    assert snapshot.reset_at == next_reset
    assert snapshot.reason_code == "rate_limit"
    assert snapshot.lease_expires_at is None


@pytest.mark.asyncio
async def test_half_open_completion_can_disable_profile(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    await affinity_store.mark_profile_cooldown(profile_id, reset_at=1, reason_code="rate_limit")
    lease = await affinity_store.acquire_profile_half_open_lease(profile_id)
    assert lease is not None

    assert (
        await affinity_store.complete_profile_half_open(
            profile_id,
            lease.token,
            OpenAIProfileState.DISABLED,
            reason_code="auth_error",
        )
        is True
    )
    snapshot = await affinity_store.get_profile_state(profile_id)
    assert snapshot.state is OpenAIProfileState.DISABLED
    assert snapshot.reason_code == "auth_error"
    assert snapshot.reset_at is None
    assert snapshot.lease_expires_at is None


@pytest.mark.asyncio
async def test_corrupted_profile_hash_fails_closed_and_is_not_overwritten(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    state_key = affinity_store.get_profile_state_cache_key(profile_id)
    redis_cache = affinity_store.cache.redis_cache
    assert redis_cache is not None
    raw_redis_client = redis_cache.init_async_client()
    await raw_redis_client.hset(
        state_key,
        mapping={
            "state": "cooldown",
            "reset_at": "1700000000",
            "updated_at": "1700000000",
        },
    )

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await affinity_store.get_profile_state(profile_id)
    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await affinity_store.mark_profile_available(profile_id)
    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await affinity_store.select_available_profile(
            _user_hash(300),
            [profile_id, "subscription-b"],
            ttl_seconds=60,
        )

    assert await raw_redis_client.hget(state_key, "state") == b"cooldown"
    assert await raw_redis_client.hget(state_key, "reason") is None


@pytest.mark.asyncio
async def test_profile_hash_with_unknown_field_fails_closed(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    profile_id = "subscription-a"
    state_key = affinity_store.get_profile_state_cache_key(profile_id)
    redis_cache = affinity_store.cache.redis_cache
    assert redis_cache is not None
    raw_redis_client = redis_cache.init_async_client()
    await raw_redis_client.hset(
        state_key,
        mapping={
            "state": "disabled",
            "reason": "auth_error",
            "updated_at": "1700000000",
            "unexpected": "must-not-be-accepted",
        },
    )

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await affinity_store.get_profile_state(profile_id)
    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await affinity_store.disable_profile(profile_id, "auth_error")

    assert await raw_redis_client.hget(state_key, "unexpected") == b"must-not-be-accepted"


@pytest.mark.asyncio
async def test_profile_availability_state_does_not_change_virtual_key_binding(
    affinity_store: OpenAISubscriptionAffinityStore,
) -> None:
    user_hash = _user_hash(50)
    await affinity_store.set_profile(user_hash, "subscription-a", ttl_seconds=60)

    await affinity_store.disable_profile("subscription-a", "auth_error")

    assert await affinity_store.get_profile(user_hash) == "subscription-a"
    assert await affinity_store.get_remaining_ttl(user_hash) in range(1, 61)
