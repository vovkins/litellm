from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.caching.dual_cache import DualCache
from litellm.router_utils.openai_subscription_affinity import (
    OpenAIProfileAvailability,
    OpenAIProfileFailureUpdateStatus,
    OpenAIProfileHalfOpenLease,
    OpenAIProfileProbeBindingAction,
    OpenAIProfileSelectionStatus,
    OpenAIProfileState,
    OpenAIProfileStateSnapshot,
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
    OpenAISubscriptionNoAvailableProfilesError,
)

USER_KEY_HASH = "a" * 64
PROFILE_ID = "openai-oauth-1"
AFFINITY_KEY = f"openai_subscription_affinity:{{v1}}:binding:{USER_KEY_HASH}"
COUNTER_KEY = "openai_subscription_affinity:{v1}:counter"
PROFILE_STATE_KEY = "openai_subscription_affinity:{v1}:profile:openai-oauth-1"
SUBSCRIPTION_A_STATE_KEY = "openai_subscription_affinity:{v1}:profile:subscription-a"
SUBSCRIPTION_B_STATE_KEY = "openai_subscription_affinity:{v1}:profile:subscription-b"
SUBSCRIPTION_C_STATE_KEY = "openai_subscription_affinity:{v1}:profile:subscription-c"


def make_store(response: object = None) -> tuple[OpenAISubscriptionAffinityStore, AsyncMock]:
    script = AsyncMock(return_value=response)
    redis_cache = MagicMock()
    redis_cache.async_register_script.return_value = script
    cache = DualCache(redis_cache=redis_cache)
    return OpenAISubscriptionAffinityStore(cache), script


def test_cache_key_is_global_and_contains_only_the_existing_hash() -> None:
    cache_key = OpenAISubscriptionAffinityStore.get_cache_key(USER_KEY_HASH.upper())

    assert cache_key == AFFINITY_KEY
    assert "model" not in cache_key


def test_assignment_keys_share_one_redis_cluster_hash_tag() -> None:
    affinity_key = OpenAISubscriptionAffinityStore.get_cache_key(USER_KEY_HASH)
    counter_key = OpenAISubscriptionAffinityStore.COUNTER_CACHE_KEY

    assert "{v1}" in affinity_key
    assert "{v1}" in counter_key


@pytest.mark.parametrize("invalid_hash", ["", "sk-raw-secret", "a" * 63, "g" * 64])
def test_cache_key_rejects_values_that_are_not_sha256_hashes(invalid_hash: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        OpenAISubscriptionAffinityStore.get_cache_key(invalid_hash)


@pytest.mark.asyncio
async def test_get_profile_reads_from_shared_redis_script() -> None:
    store, script = make_store([b"found", PROFILE_ID.encode(), b"86399"])

    assert await store.get_profile(USER_KEY_HASH) == PROFILE_ID
    script.assert_awaited_once_with(
        keys=(AFFINITY_KEY,),
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
        keys=(AFFINITY_KEY,),
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
        keys=(AFFINITY_KEY,),
        args=("delete",),
        client=None,
    )


@pytest.mark.asyncio
async def test_refresh_profile_if_current_extends_matching_binding() -> None:
    store, script = make_store([b"refreshed"])

    assert await store.refresh_profile_if_current(USER_KEY_HASH, PROFILE_ID, ttl_seconds=86400) is True
    script.assert_awaited_once_with(
        keys=(AFFINITY_KEY,),
        args=("refresh_if_current", PROFILE_ID, "86400"),
        client=None,
    )


@pytest.mark.parametrize("response", [[b"missing"], [b"mismatch"]])
@pytest.mark.asyncio
async def test_refresh_profile_if_current_does_not_recreate_or_replace_binding(response: object) -> None:
    store, _ = make_store(response)

    assert await store.refresh_profile_if_current(USER_KEY_HASH, PROFILE_ID, ttl_seconds=86400) is False


@pytest.mark.parametrize("response", [None, [b"unexpected"], [b"refreshed", b"extra"]])
@pytest.mark.asyncio
async def test_refresh_profile_if_current_fails_closed_on_malformed_result(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.refresh_profile_if_current(USER_KEY_HASH, PROFILE_ID, ttl_seconds=86400)


@pytest.mark.parametrize(
    ("user_hash", "profile_id", "ttl_seconds", "error"),
    [
        ("not-a-hash", PROFILE_ID, 60, "SHA-256"),
        (USER_KEY_HASH, "UNSAFE", 60, "profile_id"),
        (USER_KEY_HASH, PROFILE_ID, 0, "ttl_seconds"),
    ],
)
@pytest.mark.asyncio
async def test_refresh_profile_if_current_validates_inputs(
    user_hash: str,
    profile_id: str,
    ttl_seconds: int,
    error: str,
) -> None:
    store, script = make_store([b"refreshed"])

    with pytest.raises(ValueError, match=error):
        await store.refresh_profile_if_current(user_hash, profile_id, ttl_seconds=ttl_seconds)
    script.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_remaining_ttl_uses_same_atomic_snapshot() -> None:
    store, _ = make_store([b"found", PROFILE_ID.encode(), b"321"])

    assert await store.get_remaining_ttl(USER_KEY_HASH) == 321


@pytest.mark.asyncio
async def test_get_remaining_ttl_returns_none_for_missing_binding() -> None:
    store, _ = make_store([b"missing"])

    assert await store.get_remaining_ttl(USER_KEY_HASH) is None


@pytest.mark.asyncio
async def test_get_or_assign_profile_normalizes_profiles_and_uses_shared_counter() -> None:
    store, script = make_store([b"assigned", b"subscription-a", b"86400"])

    profile_id = await store.get_or_assign_profile(
        USER_KEY_HASH,
        ["subscription-b", "subscription-a", "subscription-b"],
    )

    assert profile_id == "subscription-a"
    script.assert_awaited_once_with(
        keys=(AFFINITY_KEY, COUNTER_KEY, SUBSCRIPTION_A_STATE_KEY, SUBSCRIPTION_B_STATE_KEY),
        args=("86400", "2", "0", "", "30", "subscription-a", "subscription-b"),
        client=None,
    )


@pytest.mark.asyncio
async def test_adding_profile_preserves_existing_binding() -> None:
    store, script = make_store([b"existing", b"subscription-a", b"321"])

    assert (
        await store.get_or_assign_profile(
            USER_KEY_HASH,
            ["subscription-c", "subscription-a", "subscription-b"],
            ttl_seconds=86400,
        )
        == "subscription-a"
    )
    script.assert_awaited_once_with(
        keys=(
            AFFINITY_KEY,
            COUNTER_KEY,
            SUBSCRIPTION_A_STATE_KEY,
            SUBSCRIPTION_B_STATE_KEY,
            SUBSCRIPTION_C_STATE_KEY,
        ),
        args=(
            "86400",
            "3",
            "0",
            "",
            "30",
            "subscription-a",
            "subscription-b",
            "subscription-c",
        ),
        client=None,
    )


@pytest.mark.parametrize("profiles", [[], "subscription-a"])
@pytest.mark.asyncio
async def test_get_or_assign_profile_rejects_invalid_profile_collection(profiles: Any) -> None:
    store, script = make_store([b"assigned", b"subscription-a", b"60"])

    with pytest.raises(ValueError, match="available_profile_ids"):
        await store.get_or_assign_profile(USER_KEY_HASH, profiles, ttl_seconds=60)
    script.assert_not_awaited()


@pytest.mark.asyncio
async def test_get_or_assign_profile_rejects_invalid_profile_in_collection() -> None:
    store, script = make_store([b"assigned", b"subscription-a", b"60"])

    with pytest.raises(ValueError, match="profile_id"):
        await store.get_or_assign_profile(USER_KEY_HASH, ["subscription-a", "UNSAFE"], ttl_seconds=60)
    script.assert_not_awaited()


@pytest.mark.asyncio
async def test_select_available_profile_returns_typed_unavailable_summary() -> None:
    store, _ = make_store(
        [
            b"unavailable",
            b"subscription-a",
            b"321",
            b"1",
            b"0",
            b"1",
            b"1700000100",
            b"100",
        ]
    )

    selection = await store.select_available_profile(
        USER_KEY_HASH,
        ["subscription-a", "subscription-b"],
    )

    assert selection.status is OpenAIProfileSelectionStatus.UNAVAILABLE
    assert selection.profile_id is None
    assert selection.remaining_ttl_seconds == 321
    assert selection.cooldown_profiles == 1
    assert selection.half_open_profiles == 0
    assert selection.disabled_profiles == 1
    assert selection.next_recovery_at == 1700000100
    assert selection.retry_after_seconds == 100


@pytest.mark.asyncio
async def test_inspect_profile_availability_returns_typed_full_pool_summary() -> None:
    store, script = make_store([b"0", b"1", b"0", b"1", b"1700000100", b"100"])

    availability = await store.inspect_profile_availability(
        ["subscription-b", "subscription-a"],
    )

    assert availability == OpenAIProfileAvailability(
        available_profiles=0,
        cooldown_profiles=1,
        half_open_profiles=0,
        disabled_profiles=1,
        next_recovery_at=1700000100,
        retry_after_seconds=100,
    )
    script.assert_awaited_once_with(
        keys=(SUBSCRIPTION_A_STATE_KEY, SUBSCRIPTION_B_STATE_KEY),
        args=("2", "subscription-a", "subscription-b"),
        client=None,
    )


@pytest.mark.parametrize(
    "response",
    [
        [b"invalid_state"],
        [b"0", b"1", b"0", b"0", b"1700000100", b"100"],
        [b"0", b"1", b"0", b"1", b"", b""],
        [b"0", b"1", b"0", b"1", b"1700000100", b"0"],
    ],
)
@pytest.mark.asyncio
async def test_inspect_profile_availability_rejects_invalid_result(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.inspect_profile_availability(["subscription-a", "subscription-b"])


@pytest.mark.asyncio
async def test_get_or_assign_profile_raises_typed_error_when_all_profiles_are_unavailable() -> None:
    store, _ = make_store([b"unavailable", b"", b"-2", b"0", b"0", b"2", b"", b""])

    with pytest.raises(OpenAISubscriptionNoAvailableProfilesError) as exc_info:
        await store.get_or_assign_profile(
            USER_KEY_HASH,
            ["subscription-a", "subscription-b"],
        )

    assert exc_info.value.selection.disabled_profiles == 2
    assert USER_KEY_HASH not in str(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        [b"invalid_state"],
        [b"invalid_binding"],
        [b"unavailable", b"", b"-2", b"1", b"0", b"0", b"", b""],
        [b"unavailable", b"", b"60", b"0", b"0", b"2", b"", b""],
        [b"unavailable", b"subscription-a", b"60", b"1", b"0", b"1", b"", b""],
        [b"unavailable", b"", b"-2", b"1", b"0", b"0", b"1700000100", b"0"],
        [b"unavailable", b"", b"-2", b"1", b"0", b"0", b"1700000100", b"604801"],
        [b"assigned", b"not-in-pool", b"60"],
    ],
)
@pytest.mark.asyncio
async def test_state_aware_assignment_fails_closed_on_invalid_result(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.select_available_profile(
            USER_KEY_HASH,
            ["subscription-a", "subscription-b"],
            ttl_seconds=60,
        )


@pytest.mark.asyncio
async def test_recovery_assignment_returns_private_half_open_lease(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, script = make_store([b"probe_assigned", b"subscription-a", b"60", b"1700000030"])
    monkeypatch.setattr(
        "litellm.router_utils.openai_subscription_affinity.secrets.token_hex",
        lambda _: "d" * 32,
    )

    selection = await store.select_available_profile(
        USER_KEY_HASH,
        ["subscription-b", "subscription-a"],
        ttl_seconds=60,
        allow_recovery_probe=True,
        half_open_lease_seconds=30,
    )

    assert selection.status is OpenAIProfileSelectionStatus.PROBE_ASSIGNED
    assert selection.profile_id == "subscription-a"
    assert selection.half_open_lease == OpenAIProfileHalfOpenLease(
        profile_id="subscription-a",
        token="d" * 32,
        expires_at=1700000030,
    )
    assert "d" * 32 not in repr(selection)
    script.assert_awaited_once_with(
        keys=(AFFINITY_KEY, COUNTER_KEY, SUBSCRIPTION_A_STATE_KEY, SUBSCRIPTION_B_STATE_KEY),
        args=("60", "2", "1", "d" * 32, "30", "subscription-a", "subscription-b"),
        client=None,
    )


@pytest.mark.parametrize(
    ("allow_recovery_probe", "lease_seconds", "message"),
    [
        ("yes", 30, "allow_recovery_probe"),
        (True, 0, "lease_seconds"),
        (True, 3601, "lease_seconds"),
    ],
)
@pytest.mark.asyncio
async def test_recovery_assignment_rejects_invalid_contract(
    allow_recovery_probe: Any,
    lease_seconds: int,
    message: str,
) -> None:
    store, script = make_store([b"assigned", b"subscription-a", b"60"])

    with pytest.raises(ValueError, match=message):
        await store.select_available_profile(
            USER_KEY_HASH,
            ["subscription-a"],
            ttl_seconds=60,
            allow_recovery_probe=allow_recovery_probe,
            half_open_lease_seconds=lease_seconds,
        )
    script.assert_not_awaited()


@pytest.mark.parametrize(
    ("target_state", "reset_at", "reason_code", "response", "expected_action"),
    [
        (
            OpenAIProfileState.AVAILABLE,
            None,
            None,
            [b"applied", b"refreshed"],
            OpenAIProfileProbeBindingAction.REFRESHED,
        ),
        (
            OpenAIProfileState.COOLDOWN,
            1700000200,
            "rate_limit",
            [b"applied", b"released"],
            OpenAIProfileProbeBindingAction.RELEASED,
        ),
        (
            OpenAIProfileState.DISABLED,
            None,
            "oauth_error",
            [b"applied", b"unchanged"],
            OpenAIProfileProbeBindingAction.UNCHANGED,
        ),
    ],
)
@pytest.mark.asyncio
async def test_probe_completion_updates_state_and_binding_atomically(
    target_state: OpenAIProfileState,
    reset_at: int | None,
    reason_code: str | None,
    response: object,
    expected_action: OpenAIProfileProbeBindingAction,
) -> None:
    store, script = make_store(response)

    completion = await store.complete_profile_probe_for_binding(
        USER_KEY_HASH,
        PROFILE_ID,
        "e" * 32,
        target_state,
        ttl_seconds=60,
        reset_at=reset_at,
        reason_code=reason_code,
    )

    assert completion.applied is True
    assert completion.binding_action is expected_action
    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY, AFFINITY_KEY),
        args=(
            PROFILE_ID,
            "e" * 32,
            target_state.value,
            str(reset_at) if reset_at is not None else "",
            reason_code or "",
            "60",
        ),
        client=None,
    )


@pytest.mark.asyncio
async def test_probe_completion_rejects_stale_lease() -> None:
    store, _ = make_store([b"stale"])

    completion = await store.complete_profile_probe_for_binding(
        USER_KEY_HASH,
        PROFILE_ID,
        "e" * 32,
        OpenAIProfileState.AVAILABLE,
    )

    assert completion.applied is False
    assert completion.binding_action is None


@pytest.mark.parametrize(
    ("target_state", "reset_at", "reason_code", "response", "expected_state"),
    [
        (
            OpenAIProfileState.COOLDOWN,
            1700000100,
            "rate_limit",
            [b"applied", b"cooldown"],
            OpenAIProfileState.COOLDOWN,
        ),
        (
            OpenAIProfileState.DISABLED,
            None,
            "oauth_error",
            [b"applied", b"disabled"],
            OpenAIProfileState.DISABLED,
        ),
    ],
)
@pytest.mark.asyncio
async def test_profile_failure_updates_state_and_binding_in_one_script(
    target_state: OpenAIProfileState,
    reset_at: int | None,
    reason_code: str,
    response: object,
    expected_state: OpenAIProfileState,
) -> None:
    store, script = make_store(response)

    update = await store.fail_profile_if_current_binding(
        USER_KEY_HASH,
        PROFILE_ID,
        target_state,
        reset_at=reset_at,
        reason_code=reason_code,
    )

    assert update.status is OpenAIProfileFailureUpdateStatus.APPLIED
    assert update.effective_state is expected_state
    script.assert_awaited_once_with(
        keys=(AFFINITY_KEY, PROFILE_STATE_KEY),
        args=(
            PROFILE_ID,
            target_state.value,
            str(reset_at) if reset_at is not None else "",
            reason_code,
        ),
        client=None,
    )


@pytest.mark.parametrize("reason", [b"missing", b"mismatch"])
@pytest.mark.asyncio
async def test_profile_failure_reports_stale_compare_and_swap(reason: bytes) -> None:
    store, _ = make_store([b"stale", reason])

    update = await store.fail_profile_if_current_binding(
        USER_KEY_HASH,
        PROFILE_ID,
        OpenAIProfileState.DISABLED,
        reason_code="oauth_error",
    )

    assert update.status is OpenAIProfileFailureUpdateStatus.STALE
    assert update.effective_state is None


@pytest.mark.parametrize(
    ("target_state", "reset_at", "reason_code", "message"),
    [
        (OpenAIProfileState.AVAILABLE, None, "oauth_error", "target_state"),
        (OpenAIProfileState.COOLDOWN, None, "rate_limit", "requires"),
        (OpenAIProfileState.DISABLED, 1700000100, "oauth_error", "does not accept"),
        (OpenAIProfileState.DISABLED, None, "unsafe reason", "reason_code"),
    ],
)
@pytest.mark.asyncio
async def test_profile_failure_rejects_invalid_transition_contract(
    target_state: OpenAIProfileState,
    reset_at: int | None,
    reason_code: str,
    message: str,
) -> None:
    store, script = make_store([b"applied", b"disabled"])

    with pytest.raises(ValueError, match=message):
        await store.fail_profile_if_current_binding(
            USER_KEY_HASH,
            PROFILE_ID,
            target_state,
            reset_at=reset_at,
            reason_code=reason_code,
        )
    script.assert_not_awaited()


@pytest.mark.parametrize(
    "response",
    [
        None,
        [b"unexpected"],
        [b"assigned", b"UNSAFE", b"60"],
        [b"assigned", b"subscription-a", b"-1"],
        [b"existing", b"subscription-a", b"not-a-ttl"],
    ],
)
@pytest.mark.asyncio
async def test_get_or_assign_profile_fails_closed_on_malformed_result(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.get_or_assign_profile(USER_KEY_HASH, ["subscription-a"], ttl_seconds=60)


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


def test_profile_state_key_contains_only_safe_profile_identifier() -> None:
    state_key = OpenAISubscriptionAffinityStore.get_profile_state_cache_key(PROFILE_ID)

    assert state_key == PROFILE_STATE_KEY
    assert "{v1}" in state_key
    assert USER_KEY_HASH not in state_key


@pytest.mark.asyncio
async def test_unknown_profile_defaults_to_available_without_persisted_record() -> None:
    store, script = make_store([b"missing"])

    snapshot = await store.get_profile_state(PROFILE_ID)

    assert snapshot == OpenAIProfileStateSnapshot(
        state=OpenAIProfileState.AVAILABLE,
        reset_at=None,
        lease_expires_at=None,
        reason_code=None,
        updated_at=None,
        persisted=False,
    )
    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=("read",),
        client=None,
    )


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        (
            [b"found", b"available", b"", b"", b"", b"1700000000"],
            OpenAIProfileStateSnapshot(
                state=OpenAIProfileState.AVAILABLE,
                reset_at=None,
                lease_expires_at=None,
                reason_code=None,
                updated_at=1700000000,
                persisted=True,
            ),
        ),
        (
            [b"found", b"cooldown", b"1700000100", b"", b"rate_limit", b"1700000000"],
            OpenAIProfileStateSnapshot(
                state=OpenAIProfileState.COOLDOWN,
                reset_at=1700000100,
                lease_expires_at=None,
                reason_code="rate_limit",
                updated_at=1700000000,
                persisted=True,
            ),
        ),
        (
            [
                b"found",
                b"half_open",
                b"1700000100",
                b"1700000200",
                b"rate_limit",
                b"1700000150",
            ],
            OpenAIProfileStateSnapshot(
                state=OpenAIProfileState.HALF_OPEN,
                reset_at=1700000100,
                lease_expires_at=1700000200,
                reason_code="rate_limit",
                updated_at=1700000150,
                persisted=True,
            ),
        ),
        (
            [b"found", b"disabled", b"", b"", b"auth_error", b"1700000000"],
            OpenAIProfileStateSnapshot(
                state=OpenAIProfileState.DISABLED,
                reset_at=None,
                lease_expires_at=None,
                reason_code="auth_error",
                updated_at=1700000000,
                persisted=True,
            ),
        ),
    ],
)
@pytest.mark.asyncio
async def test_profile_state_reads_valid_persisted_shapes(
    response: object,
    expected: OpenAIProfileStateSnapshot,
) -> None:
    store, _ = make_store(response)

    assert await store.get_profile_state(PROFILE_ID) == expected


@pytest.mark.parametrize(
    "response",
    [
        None,
        [b"invalid"],
        [b"found", b"unknown", b"", b"", b"", b"1700000000"],
        [b"found", b"available", b"1700000100", b"", b"", b"1700000000"],
        [b"found", b"cooldown", b"1700000100", b"", b"", b"1700000000"],
        [b"found", b"half_open", b"1700000100", b"", b"rate_limit", b"1700000000"],
        [b"found", b"disabled", b"1700000100", b"", b"auth_error", b"1700000000"],
        [b"found", b"disabled", b"", b"", b"Unsafe Value", b"1700000000"],
        [b"found", b"disabled", b"", b"", b"auth_error", b"not-a-time"],
    ],
)
@pytest.mark.asyncio
async def test_profile_state_fails_closed_on_malformed_data(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.get_profile_state(PROFILE_ID)


@pytest.mark.asyncio
async def test_mark_profile_available_uses_atomic_state_script() -> None:
    store, script = make_store([b"updated"])

    await store.mark_profile_available(PROFILE_ID)

    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=("mark_available",),
        client=None,
    )


@pytest.mark.asyncio
async def test_mark_profile_cooldown_uses_safe_reason_and_absolute_reset() -> None:
    store, script = make_store([b"updated"])

    await store.mark_profile_cooldown(PROFILE_ID, 1700000100, "rate_limit")

    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=("mark_cooldown", "1700000100", "rate_limit"),
        client=None,
    )


@pytest.mark.asyncio
async def test_disable_profile_uses_only_safe_reason_code() -> None:
    store, script = make_store([b"updated"])

    await store.disable_profile(PROFILE_ID, "auth_error")

    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=("disable", "auth_error"),
        client=None,
    )


@pytest.mark.parametrize("reset_at", [0, -1, True, 253402300800])
@pytest.mark.asyncio
async def test_cooldown_rejects_invalid_reset_timestamp(reset_at: Any) -> None:
    store, script = make_store([b"updated"])

    with pytest.raises(ValueError, match="reset_at"):
        await store.mark_profile_cooldown(PROFILE_ID, reset_at, "rate_limit")
    script.assert_not_awaited()


@pytest.mark.parametrize("reason_code", ["", "UPPERCASE", "raw provider error", "x" * 65])
@pytest.mark.asyncio
async def test_profile_state_rejects_unsafe_reason_codes(reason_code: str) -> None:
    store, script = make_store([b"updated"])

    with pytest.raises(ValueError, match="reason_code"):
        await store.disable_profile(PROFILE_ID, reason_code)
    script.assert_not_awaited()


@pytest.mark.asyncio
async def test_half_open_acquisition_returns_private_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    store, script = make_store([b"acquired", b"1700000030"])
    monkeypatch.setattr(
        "litellm.router_utils.openai_subscription_affinity.secrets.token_hex",
        lambda _: "b" * 32,
    )

    lease = await store.acquire_profile_half_open_lease(PROFILE_ID, lease_seconds=30)

    assert lease == OpenAIProfileHalfOpenLease(
        profile_id=PROFILE_ID,
        token="b" * 32,
        expires_at=1700000030,
    )
    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=("acquire_half_open", "b" * 32, "30"),
        client=None,
    )
    assert "b" * 32 not in repr(lease)


@pytest.mark.parametrize("state", list(OpenAIProfileState))
@pytest.mark.asyncio
async def test_half_open_acquisition_returns_none_when_lease_is_not_available(
    state: OpenAIProfileState,
) -> None:
    store, _ = make_store([b"not_acquired", state.value.encode()])

    assert await store.acquire_profile_half_open_lease(PROFILE_ID) is None


@pytest.mark.parametrize("lease_seconds", [0, -1, True, 3601])
@pytest.mark.asyncio
async def test_half_open_acquisition_rejects_invalid_duration(lease_seconds: Any) -> None:
    store, script = make_store([b"acquired", b"1700000030"])

    with pytest.raises(ValueError, match="lease_seconds"):
        await store.acquire_profile_half_open_lease(PROFILE_ID, lease_seconds)
    script.assert_not_awaited()


@pytest.mark.parametrize(
    "response",
    [None, [b"invalid"], [b"not_acquired", b"unknown"], [b"acquired", b"not-a-time"]],
)
@pytest.mark.asyncio
async def test_half_open_acquisition_fails_closed_on_invalid_response(response: object) -> None:
    store, _ = make_store(response)

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        await store.acquire_profile_half_open_lease(PROFILE_ID)


@pytest.mark.parametrize(
    ("target_state", "reset_at", "reason_code", "expected_args"),
    [
        (
            OpenAIProfileState.AVAILABLE,
            None,
            None,
            ("complete_half_open", "c" * 32, "available", "", ""),
        ),
        (
            OpenAIProfileState.COOLDOWN,
            1700000200,
            "rate_limit",
            ("complete_half_open", "c" * 32, "cooldown", "1700000200", "rate_limit"),
        ),
        (
            OpenAIProfileState.DISABLED,
            None,
            "auth_error",
            ("complete_half_open", "c" * 32, "disabled", "", "auth_error"),
        ),
    ],
)
@pytest.mark.asyncio
async def test_half_open_completion_applies_supported_target(
    target_state: OpenAIProfileState,
    reset_at: int | None,
    reason_code: str | None,
    expected_args: tuple[str, ...],
) -> None:
    store, script = make_store([b"applied"])

    assert (
        await store.complete_profile_half_open(
            PROFILE_ID,
            "c" * 32,
            target_state,
            reset_at=reset_at,
            reason_code=reason_code,
        )
        is True
    )
    script.assert_awaited_once_with(
        keys=(PROFILE_STATE_KEY,),
        args=expected_args,
        client=None,
    )


@pytest.mark.asyncio
async def test_half_open_completion_does_not_apply_stale_lease() -> None:
    store, _ = make_store([b"stale"])

    assert (
        await store.complete_profile_half_open(
            PROFILE_ID,
            "c" * 32,
            OpenAIProfileState.AVAILABLE,
        )
        is False
    )


@pytest.mark.parametrize(
    ("lease_token", "target_state", "reset_at", "reason_code", "message"),
    [
        ("invalid", OpenAIProfileState.AVAILABLE, None, None, "lease_token"),
        ("c" * 32, OpenAIProfileState.HALF_OPEN, None, None, "target_state"),
        ("c" * 32, "available", None, None, "target_state"),
        ("c" * 32, OpenAIProfileState.AVAILABLE, 1, None, "does not accept"),
        ("c" * 32, OpenAIProfileState.COOLDOWN, None, "rate_limit", "requires"),
        ("c" * 32, OpenAIProfileState.DISABLED, 1, "auth_error", "without reset_at"),
    ],
)
@pytest.mark.asyncio
async def test_half_open_completion_rejects_invalid_contract(
    lease_token: str,
    target_state: Any,
    reset_at: int | None,
    reason_code: str | None,
    message: str,
) -> None:
    store, script = make_store([b"applied"])

    with pytest.raises(ValueError, match=message):
        await store.complete_profile_half_open(
            PROFILE_ID,
            lease_token,
            target_state,
            reset_at=reset_at,
            reason_code=reason_code,
        )
    script.assert_not_awaited()


@pytest.mark.parametrize("operation", ["available", "cooldown", "disabled"])
@pytest.mark.asyncio
async def test_corrupt_profile_state_cannot_be_silently_overwritten(operation: str) -> None:
    store, _ = make_store([b"invalid"])

    with pytest.raises(OpenAISubscriptionAffinityStoreError):
        if operation == "available":
            await store.mark_profile_available(PROFILE_ID)
        elif operation == "cooldown":
            await store.mark_profile_cooldown(PROFILE_ID, 1700000100, "rate_limit")
        else:
            await store.disable_profile(PROFILE_ID, "auth_error")


@pytest.mark.asyncio
async def test_profile_state_storage_failure_does_not_expose_internal_values() -> None:
    store, script = make_store()
    script.side_effect = ConnectionError(f"failed profile={PROFILE_ID} reason=auth_error token={'d' * 32}")

    with pytest.raises(OpenAISubscriptionAffinityStoreError) as exc_info:
        await store.disable_profile(PROFILE_ID, "auth_error")

    message = str(exc_info.value)
    assert PROFILE_ID not in message
    assert "auth_error" not in message
    assert "d" * 32 not in message


def test_store_does_not_change_existing_deployment_affinity_namespace() -> None:
    from litellm.router_utils.pre_call_checks.deployment_affinity_check import (
        DeploymentAffinityCheck,
    )

    assert DeploymentAffinityCheck.CACHE_KEY_PREFIX == "deployment_affinity:v1"
    assert OpenAISubscriptionAffinityStore.CACHE_KEY_PREFIX != DeploymentAffinityCheck.CACHE_KEY_PREFIX
