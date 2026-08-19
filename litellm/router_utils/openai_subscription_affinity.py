"""Redis-backed affinity and availability for OpenAI OAuth profiles.

This module only owns persistence and atomic state transitions. Error
classification, failover, and routing policy are implemented by consumers.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Final, Protocol, cast

from litellm.caching.dual_cache import DualCache

_AFFINITY_STORE_SCRIPT: Final = """
local operation = ARGV[1]

if operation == 'read' then
  local value = redis.call('GET', KEYS[1])
  if value == false then
    return {'missing'}
  end
  return {'found', value, tostring(redis.call('TTL', KEYS[1]))}
end

if operation == 'write' then
  redis.call('SET', KEYS[1], ARGV[2], 'EX', ARGV[3])
  return {'written'}
end

if operation == 'delete' then
  return {'deleted', tostring(redis.call('DEL', KEYS[1]))}
end

if operation == 'refresh_if_current' then
  local current = redis.call('GET', KEYS[1])
  if current == false then
    return {'missing'}
  end
  if current ~= ARGV[2] then
    return {'mismatch'}
  end
  redis.call('EXPIRE', KEYS[1], ARGV[3])
  return {'refreshed'}
end

return redis.error_reply('unsupported affinity store operation')
"""

_PROFILE_STATE_VALIDATION_LUA: Final = """
local function is_positive_integer(value, maximum)
  if value == false or value == nil or string.match(value, '^%d+$') == nil then
    return false
  end
  local parsed = tonumber(value)
  return parsed ~= nil and parsed > 0 and parsed <= maximum
end

local function is_reason_code(value)
  return value ~= false
    and value ~= nil
    and string.len(value) <= 64
    and string.match(value, '^[a-z][a-z0-9_]*$') ~= nil
end

local function is_lease_token(value)
  return value ~= false
    and value ~= nil
    and string.len(value) == 32
    and string.match(value, '^[0-9a-f]+$') ~= nil
end

local function is_profile_id(value)
  return value ~= false
    and value ~= nil
    and string.len(value) <= 64
    and string.match(value, '^[a-z0-9][a-z0-9_-]*$') ~= nil
end

local function validate_profile_record(state_key)
  local state = redis.call('HGET', state_key, 'state')
  if state == false then
    if redis.call('EXISTS', state_key) == 0 then
      return 'missing'
    end
    return 'invalid'
  end

  local reset_at = redis.call('HGET', state_key, 'reset_at')
  local lease_token = redis.call('HGET', state_key, 'lease_token')
  local lease_until = redis.call('HGET', state_key, 'lease_until')
  local reason = redis.call('HGET', state_key, 'reason')
  local updated_at = redis.call('HGET', state_key, 'updated_at')
  local field_count = redis.call('HLEN', state_key)

  if not is_positive_integer(updated_at, 253402300799) then
    return 'invalid'
  end
  if state == 'available' then
    if field_count ~= 2
      or reset_at ~= false
      or lease_token ~= false
      or lease_until ~= false
      or reason ~= false then
      return 'invalid'
    end
  elseif state == 'cooldown' then
    if field_count ~= 4
      or not is_positive_integer(reset_at, 253402300799)
      or lease_token ~= false
      or lease_until ~= false
      or not is_reason_code(reason) then
      return 'invalid'
    end
  elseif state == 'half_open' then
    if field_count ~= 6
      or not is_positive_integer(reset_at, 253402300799)
      or not is_lease_token(lease_token)
      or not is_positive_integer(lease_until, 253402300799)
      or not is_reason_code(reason) then
      return 'invalid'
    end
  elseif state == 'disabled' then
    if field_count ~= 3
      or reset_at ~= false
      or lease_token ~= false
      or lease_until ~= false
      or not is_reason_code(reason) then
      return 'invalid'
    end
  else
    return 'invalid'
  end
  return state
end
"""

_ASSIGN_PROFILE_SCRIPT: Final = (
    _PROFILE_STATE_VALIDATION_LUA
    + """
local ttl = tonumber(ARGV[1])
local profile_count = tonumber(ARGV[2])
if ttl == nil or ttl <= 0 or profile_count == nil or profile_count <= 0 then
  return redis.error_reply('invalid affinity assignment arguments')
end
if #KEYS ~= profile_count + 2 or #ARGV ~= profile_count + 2 then
  return redis.error_reply('invalid affinity assignment shape')
end

local current = redis.call('GET', KEYS[1])
if current ~= false and not is_profile_id(current) then
  return {'invalid_binding'}
end
local current_ttl = -2
if current ~= false then
  current_ttl = redis.call('TTL', KEYS[1])
  if current_ttl < 0 then
    return {'invalid_binding'}
  end
end

local eligible = {}
local current_is_available = false
local cooldown_count = 0
local half_open_count = 0
local disabled_count = 0
local next_recovery_at = nil

for index = 1, profile_count do
  local profile_id = ARGV[index + 2]
  if not is_profile_id(profile_id) then
    return redis.error_reply('invalid affinity profile identifier')
  end
  local state_key = KEYS[index + 2]
  local status = validate_profile_record(state_key)
  if status == 'invalid' then
    return {'invalid_state'}
  end
  if status == 'missing' or status == 'available' then
    table.insert(eligible, profile_id)
    if current == profile_id then
      current_is_available = true
    end
  elseif status == 'cooldown' then
    cooldown_count = cooldown_count + 1
    local reset_at = tonumber(redis.call('HGET', state_key, 'reset_at'))
    if next_recovery_at == nil or reset_at < next_recovery_at then
      next_recovery_at = reset_at
    end
  elseif status == 'half_open' then
    half_open_count = half_open_count + 1
    local lease_until = tonumber(redis.call('HGET', state_key, 'lease_until'))
    if next_recovery_at == nil or lease_until < next_recovery_at then
      next_recovery_at = lease_until
    end
  elseif status == 'disabled' then
    disabled_count = disabled_count + 1
  end
end

if current_is_available then
  return {'existing', current, tostring(current_ttl)}
end

if #eligible == 0 then
  local current_value = current == false and '' or current
  return {
    'unavailable',
    current_value,
    tostring(current_ttl),
    tostring(cooldown_count),
    tostring(half_open_count),
    tostring(disabled_count),
    next_recovery_at == nil and '' or tostring(next_recovery_at)
  }
end

local sequence = redis.call('INCR', KEYS[2])
local selected_profile = eligible[((sequence - 1) % #eligible) + 1]
redis.call('SET', KEYS[1], selected_profile, 'EX', ttl)
return {current == false and 'assigned' or 'reassigned', selected_profile, tostring(ttl)}
"""
)

_FAIL_PROFILE_SCRIPT: Final = (
    _PROFILE_STATE_VALIDATION_LUA
    + """
local expected_profile = ARGV[1]
local target_state = ARGV[2]
local reset_at = ARGV[3]
local reason = ARGV[4]
if #KEYS ~= 2 or #ARGV ~= 4 then
  return redis.error_reply('invalid profile failure shape')
end
if not is_profile_id(expected_profile) or not is_reason_code(reason) then
  return redis.error_reply('invalid profile failure arguments')
end
if target_state ~= 'cooldown' and target_state ~= 'disabled' then
  return redis.error_reply('invalid profile failure target')
end
if target_state == 'cooldown' and not is_positive_integer(reset_at, 253402300799) then
  return redis.error_reply('invalid profile cooldown target')
end
if target_state == 'disabled' and reset_at ~= '' then
  return redis.error_reply('invalid profile disabled target')
end

local current = redis.call('GET', KEYS[1])
if current == false then
  return {'stale', 'missing'}
end
if not is_profile_id(current) then
  return {'invalid_binding'}
end
if current ~= expected_profile then
  return {'stale', 'mismatch'}
end

local status = validate_profile_record(KEYS[2])
if status == 'invalid' then
  return {'invalid_state'}
end

local now = tonumber(redis.call('TIME')[1])
local effective_state = status == 'missing' and 'available' or status
if status ~= 'half_open' then
  if target_state == 'disabled' then
    redis.call(
      'HSET', KEYS[2],
      'state', 'disabled',
      'reason', reason,
      'updated_at', tostring(now)
    )
    redis.call('HDEL', KEYS[2], 'reset_at', 'lease_token', 'lease_until')
    effective_state = 'disabled'
  elseif status ~= 'disabled' then
    local requested_reset_at = tonumber(reset_at)
    if status == 'cooldown' then
      local current_reset_at = tonumber(redis.call('HGET', KEYS[2], 'reset_at'))
      if current_reset_at > requested_reset_at then
        requested_reset_at = current_reset_at
      end
    end
    redis.call(
      'HSET', KEYS[2],
      'state', 'cooldown',
      'reset_at', tostring(requested_reset_at),
      'reason', reason,
      'updated_at', tostring(now)
    )
    redis.call('HDEL', KEYS[2], 'lease_token', 'lease_until')
    effective_state = 'cooldown'
  end
end

redis.call('DEL', KEYS[1])
return {'applied', effective_state}
"""
)

_PROFILE_STATE_SCRIPT: Final = (
    _PROFILE_STATE_VALIDATION_LUA
    + """
local operation = ARGV[1]

local function snapshot()
  local state = redis.call('HGET', KEYS[1], 'state')
  if state == false then
    return {'missing'}
  end
  return {
    'found',
    state,
    redis.call('HGET', KEYS[1], 'reset_at') or '',
    redis.call('HGET', KEYS[1], 'lease_until') or '',
    redis.call('HGET', KEYS[1], 'reason') or '',
    redis.call('HGET', KEYS[1], 'updated_at') or ''
  }
end

local status = validate_profile_record(KEYS[1])
if status == 'invalid' then
  return {'invalid'}
end
if operation == 'read' then
  return snapshot()
end

local now = tonumber(redis.call('TIME')[1])

if operation == 'mark_available' then
  redis.call('HSET', KEYS[1], 'state', 'available', 'updated_at', tostring(now))
  redis.call('HDEL', KEYS[1], 'reset_at', 'lease_token', 'lease_until', 'reason')
  return {'updated'}
end

if operation == 'mark_cooldown' then
  if not is_positive_integer(ARGV[2], 253402300799) or not is_reason_code(ARGV[3]) then
    return redis.error_reply('invalid profile cooldown arguments')
  end
  redis.call(
    'HSET', KEYS[1],
    'state', 'cooldown',
    'reset_at', ARGV[2],
    'reason', ARGV[3],
    'updated_at', tostring(now)
  )
  redis.call('HDEL', KEYS[1], 'lease_token', 'lease_until')
  return {'updated'}
end

if operation == 'disable' then
  if not is_reason_code(ARGV[2]) then
    return redis.error_reply('invalid profile disabled arguments')
  end
  redis.call(
    'HSET', KEYS[1],
    'state', 'disabled',
    'reason', ARGV[2],
    'updated_at', tostring(now)
  )
  redis.call('HDEL', KEYS[1], 'reset_at', 'lease_token', 'lease_until')
  return {'updated'}
end

if operation == 'acquire_half_open' then
  if not is_lease_token(ARGV[2]) or not is_positive_integer(ARGV[3], 3600) then
    return redis.error_reply('invalid profile half-open arguments')
  end
  if status == 'missing' or status == 'available' or status == 'disabled' then
    return {'not_acquired', status == 'missing' and 'available' or status}
  end
  if status == 'cooldown' then
    local reset_at = tonumber(redis.call('HGET', KEYS[1], 'reset_at'))
    if reset_at > now then
      return {'not_acquired', 'cooldown'}
    end
  elseif status == 'half_open' then
    local lease_until = tonumber(redis.call('HGET', KEYS[1], 'lease_until'))
    if lease_until > now then
      return {'not_acquired', 'half_open'}
    end
  end

  local lease_until = now + tonumber(ARGV[3])
  redis.call(
    'HSET', KEYS[1],
    'state', 'half_open',
    'lease_token', ARGV[2],
    'lease_until', tostring(lease_until),
    'updated_at', tostring(now)
  )
  return {'acquired', tostring(lease_until)}
end

if operation == 'complete_half_open' then
  if not is_lease_token(ARGV[2]) then
    return redis.error_reply('invalid profile half-open completion arguments')
  end
  if status ~= 'half_open' then
    return {'stale'}
  end
  local current_token = redis.call('HGET', KEYS[1], 'lease_token')
  local lease_until = tonumber(redis.call('HGET', KEYS[1], 'lease_until'))
  if current_token ~= ARGV[2] or lease_until <= now then
    return {'stale'}
  end

  local target = ARGV[3]
  if target == 'available' then
    redis.call('HSET', KEYS[1], 'state', target, 'updated_at', tostring(now))
    redis.call('HDEL', KEYS[1], 'reset_at', 'lease_token', 'lease_until', 'reason')
  elseif target == 'cooldown' then
    if not is_positive_integer(ARGV[4], 253402300799) or not is_reason_code(ARGV[5]) then
      return redis.error_reply('invalid profile cooldown completion arguments')
    end
    redis.call(
      'HSET', KEYS[1],
      'state', target,
      'reset_at', ARGV[4],
      'reason', ARGV[5],
      'updated_at', tostring(now)
    )
    redis.call('HDEL', KEYS[1], 'lease_token', 'lease_until')
  elseif target == 'disabled' then
    if not is_reason_code(ARGV[5]) then
      return redis.error_reply('invalid profile disabled completion arguments')
    end
    redis.call(
      'HSET', KEYS[1],
      'state', target,
      'reason', ARGV[5],
      'updated_at', tostring(now)
    )
    redis.call('HDEL', KEYS[1], 'reset_at', 'lease_token', 'lease_until')
  else
    return redis.error_reply('invalid profile half-open completion target')
  end
  return {'applied'}
end

return redis.error_reply('unsupported profile state operation')
"""
)

_SHA256_HEX_PATTERN: Final = re.compile(r"^[0-9a-fA-F]{64}$")
_PROFILE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_PROFILE_REASON_PATTERN: Final = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_HALF_OPEN_LEASE_TOKEN_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")


class _RedisScript(Protocol):
    async def __call__(
        self,
        *,
        keys: Sequence[str],
        args: Sequence[Any],
        client: Any | None = None,
    ) -> Any: ...


class _RedisScriptCache(Protocol):
    def async_register_script(self, script: str) -> _RedisScript: ...


class OpenAISubscriptionAffinityStoreError(RuntimeError):
    """The shared affinity state cannot be read or changed safely."""


class OpenAISubscriptionNoAvailableProfilesError(OpenAISubscriptionAffinityStoreError):
    """No configured OAuth profile is eligible for normal traffic."""

    def __init__(self, selection: OpenAIProfileSelection) -> None:
        super().__init__("No OpenAI subscription profile is currently available")
        self.selection = selection


@dataclass(frozen=True, slots=True)
class _AffinitySnapshot:
    profile_id: str
    ttl_seconds: int


class OpenAIProfileState(str, Enum):
    """Persisted availability state for one server-side OAuth profile."""

    AVAILABLE = "available"
    COOLDOWN = "cooldown"
    HALF_OPEN = "half_open"
    DISABLED = "disabled"


class OpenAIProfileSelectionStatus(str, Enum):
    """Outcome of one atomic state-aware affinity decision."""

    EXISTING = "existing"
    ASSIGNED = "assigned"
    REASSIGNED = "reassigned"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class OpenAIProfileSelection:
    status: OpenAIProfileSelectionStatus
    profile_id: str | None
    remaining_ttl_seconds: int | None
    cooldown_profiles: int = 0
    half_open_profiles: int = 0
    disabled_profiles: int = 0
    next_recovery_at: int | None = None


class OpenAIProfileFailureUpdateStatus(str, Enum):
    """CAS outcome for a profile-scoped provider failure."""

    APPLIED = "applied"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class OpenAIProfileFailureUpdate:
    status: OpenAIProfileFailureUpdateStatus
    effective_state: OpenAIProfileState | None


@dataclass(frozen=True, slots=True)
class OpenAIProfileStateSnapshot:
    state: OpenAIProfileState
    reset_at: int | None
    lease_expires_at: int | None
    reason_code: str | None
    updated_at: int | None
    persisted: bool


@dataclass(frozen=True, slots=True)
class OpenAIProfileHalfOpenLease:
    profile_id: str
    token: str = field(repr=False)
    expires_at: int


class OpenAISubscriptionAffinityStore:
    """Persist one global OpenAI OAuth profile binding per virtual-key hash.

    The store deliberately bypasses ``DualCache``'s in-memory tier. Affinity must
    be shared by every proxy replica, and a Redis failure must be visible to the
    future routing layer instead of producing conflicting pod-local decisions.
    ``DualCache`` is still the owner of the Redis connection and connection pool.
    """

    CACHE_KEY_PREFIX: Final = "openai_subscription_affinity:{v1}"
    COUNTER_CACHE_KEY: Final = f"{CACHE_KEY_PREFIX}:counter"
    PROFILE_STATE_CACHE_KEY_PREFIX: Final = f"{CACHE_KEY_PREFIX}:profile"
    DEFAULT_TTL_SECONDS: Final = 24 * 60 * 60
    DEFAULT_HALF_OPEN_LEASE_SECONDS: Final = 30
    MAX_HALF_OPEN_LEASE_SECONDS: Final = 60 * 60
    MAX_TIMESTAMP: Final = 253402300799

    def __init__(self, cache: DualCache) -> None:
        self.cache = cache

    @classmethod
    def get_cache_key(cls, user_api_key_hash: str) -> str:
        normalized_hash: Final = cls._validate_user_api_key_hash(user_api_key_hash)
        return f"{cls.CACHE_KEY_PREFIX}:binding:{normalized_hash}"

    @classmethod
    def get_profile_state_cache_key(cls, profile_id: str) -> str:
        validated_profile_id: Final = cls._validate_profile_id(profile_id)
        return f"{cls.PROFILE_STATE_CACHE_KEY_PREFIX}:{validated_profile_id}"

    async def get_profile(self, user_api_key_hash: str) -> str | None:
        snapshot: Final = await self._read_snapshot(user_api_key_hash)
        return snapshot.profile_id if snapshot is not None else None

    async def set_profile(self, user_api_key_hash: str, profile_id: str, ttl_seconds: int) -> None:
        cache_key: Final = self.get_cache_key(user_api_key_hash)
        validated_profile_id: Final = self._validate_profile_id(profile_id)
        validated_ttl: Final = self._validate_ttl(ttl_seconds)
        response: Final = await self._execute(
            _AFFINITY_STORE_SCRIPT,
            (cache_key,),
            ("write", validated_profile_id, str(validated_ttl)),
        )
        if response != ("written",):
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity write returned invalid data")

    async def delete_profile(self, user_api_key_hash: str) -> None:
        cache_key: Final = self.get_cache_key(user_api_key_hash)
        response: Final = await self._execute(_AFFINITY_STORE_SCRIPT, (cache_key,), ("delete",))
        if len(response) != 2 or response[0] != "deleted" or response[1] not in {"0", "1"}:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity delete returned invalid data")

    async def refresh_profile_if_current(
        self,
        user_api_key_hash: str,
        profile_id: str,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> bool:
        """Refresh a binding only if it still points to the successful profile.

        The comparison and expiry update run in one Redis script. A delayed
        success from an old profile therefore cannot overwrite or extend a newer
        binding, and an expired binding is never recreated by this operation.
        """

        cache_key: Final = self.get_cache_key(user_api_key_hash)
        validated_profile_id: Final = self._validate_profile_id(profile_id)
        validated_ttl: Final = self._validate_ttl(ttl_seconds)
        response: Final = await self._execute(
            _AFFINITY_STORE_SCRIPT,
            (cache_key,),
            ("refresh_if_current", validated_profile_id, str(validated_ttl)),
        )
        if response == ("refreshed",):
            return True
        if response in {("missing",), ("mismatch",)}:
            return False
        raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity refresh returned invalid data")

    async def get_remaining_ttl(self, user_api_key_hash: str) -> int | None:
        snapshot: Final = await self._read_snapshot(user_api_key_hash)
        return snapshot.ttl_seconds if snapshot is not None else None

    async def get_or_assign_profile(
        self,
        user_api_key_hash: str,
        available_profile_ids: Sequence[str],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> str:
        """Return an available binding or raise a typed availability error."""

        selection: Final = await self.select_available_profile(
            user_api_key_hash=user_api_key_hash,
            available_profile_ids=available_profile_ids,
            ttl_seconds=ttl_seconds,
        )
        if selection.profile_id is None:
            raise OpenAISubscriptionNoAvailableProfilesError(selection)
        return selection.profile_id

    async def select_available_profile(
        self,
        user_api_key_hash: str,
        available_profile_ids: Sequence[str],
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
    ) -> OpenAIProfileSelection:
        """Atomically preserve or assign a profile eligible for normal traffic.

        Profile order supplied by callers cannot affect distribution: identifiers
        are validated, de-duplicated, and sorted before the Redis script runs.
        Missing state records and explicit ``available`` records are eligible.
        ``cooldown``, ``half_open`` and ``disabled`` profiles are excluded. An
        eligible existing binding is kept without refreshing its TTL; an
        unavailable or removed binding is replaced through the same shared
        round-robin counter used for first assignments.
        """

        cache_key: Final = self.get_cache_key(user_api_key_hash)
        profiles: Final = self._normalize_profile_ids(available_profile_ids)
        validated_ttl: Final = self._validate_ttl(ttl_seconds)
        state_keys: Final = tuple(self.get_profile_state_cache_key(profile_id) for profile_id in profiles)
        response: Final = await self._execute(
            _ASSIGN_PROFILE_SCRIPT,
            (cache_key, self.COUNTER_CACHE_KEY, *state_keys),
            (str(validated_ttl), str(len(profiles)), *profiles),
        )
        if len(response) == 3 and response[0] in {
            OpenAIProfileSelectionStatus.EXISTING.value,
            OpenAIProfileSelectionStatus.ASSIGNED.value,
            OpenAIProfileSelectionStatus.REASSIGNED.value,
        }:
            try:
                status: Final = OpenAIProfileSelectionStatus(response[0])
                profile_id: Final = self._validate_profile_id(response[1])
                remaining_ttl: Final = int(response[2])
            except (TypeError, ValueError) as exc:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription affinity assignment returned invalid data"
                ) from exc
            if profile_id not in profiles or remaining_ttl < 0:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription affinity assignment returned invalid data"
                )
            return OpenAIProfileSelection(
                status=status,
                profile_id=profile_id,
                remaining_ttl_seconds=remaining_ttl,
            )

        if len(response) == 7 and response[0] == OpenAIProfileSelectionStatus.UNAVAILABLE.value:
            try:
                current_profile = self._validate_profile_id(response[1]) if response[1] else None
                remaining_ttl = int(response[2])
                cooldown_profiles = int(response[3])
                half_open_profiles = int(response[4])
                disabled_profiles = int(response[5])
                next_recovery_at = self._parse_optional_timestamp(response[6], "next_recovery_at")
            except (TypeError, ValueError) as exc:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription affinity availability returned invalid data"
                ) from exc
            if (
                remaining_ttl < -2
                or (current_profile is None and remaining_ttl != -2)
                or (current_profile is not None and remaining_ttl < 0)
                or min(cooldown_profiles, half_open_profiles, disabled_profiles) < 0
                or cooldown_profiles + half_open_profiles + disabled_profiles != len(profiles)
                or ((cooldown_profiles + half_open_profiles > 0) != (next_recovery_at is not None))
            ):
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription affinity availability returned invalid data"
                )
            return OpenAIProfileSelection(
                status=OpenAIProfileSelectionStatus.UNAVAILABLE,
                profile_id=None,
                remaining_ttl_seconds=remaining_ttl if current_profile is not None else None,
                cooldown_profiles=cooldown_profiles,
                half_open_profiles=half_open_profiles,
                disabled_profiles=disabled_profiles,
                next_recovery_at=next_recovery_at,
            )

        raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity assignment returned invalid data")

    async def fail_profile_if_current_binding(
        self,
        user_api_key_hash: str,
        profile_id: str,
        target_state: OpenAIProfileState,
        *,
        reason_code: str,
        reset_at: int | None = None,
    ) -> OpenAIProfileFailureUpdate:
        """Apply a profile failure and remove only its still-current binding.

        The state transition and compare-and-delete execute in one Redis script.
        A delayed failure from an old request therefore cannot replace or remove
        a newer binding. Existing ``disabled`` state is never downgraded to a
        cooldown, and an active ``half_open`` lease is owned by its probe and is
        not overwritten by an unrelated delayed request.
        """

        cache_key: Final = self.get_cache_key(user_api_key_hash)
        validated_profile_id: Final = self._validate_profile_id(profile_id)
        validated_reason: Final = self._validate_reason_code(reason_code)
        if target_state is OpenAIProfileState.COOLDOWN:
            if reset_at is None:
                raise ValueError("cooldown target_state requires reset_at")
            validated_reset_at = str(self._validate_timestamp(reset_at, "reset_at"))
        elif target_state is OpenAIProfileState.DISABLED:
            if reset_at is not None:
                raise ValueError("disabled target_state does not accept reset_at")
            validated_reset_at = ""
        else:
            raise ValueError("target_state must be cooldown or disabled")

        response: Final = await self._execute(
            _FAIL_PROFILE_SCRIPT,
            (cache_key, self.get_profile_state_cache_key(validated_profile_id)),
            (
                validated_profile_id,
                target_state.value,
                validated_reset_at,
                validated_reason,
            ),
        )
        if len(response) == 2 and response[0] == OpenAIProfileFailureUpdateStatus.STALE.value:
            if response[1] not in {"missing", "mismatch"}:
                raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile failure returned invalid data")
            return OpenAIProfileFailureUpdate(
                status=OpenAIProfileFailureUpdateStatus.STALE,
                effective_state=None,
            )
        if len(response) == 2 and response[0] == OpenAIProfileFailureUpdateStatus.APPLIED.value:
            try:
                effective_state: Final = OpenAIProfileState(response[1])
            except ValueError as exc:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription profile failure returned invalid data"
                ) from exc
            if effective_state is OpenAIProfileState.AVAILABLE:
                raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile failure returned invalid data")
            return OpenAIProfileFailureUpdate(
                status=OpenAIProfileFailureUpdateStatus.APPLIED,
                effective_state=effective_state,
            )
        raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile failure returned invalid data")

    async def get_profile_state(self, profile_id: str) -> OpenAIProfileStateSnapshot:
        """Read one profile state; an unknown profile starts as available."""

        cache_key: Final = self.get_profile_state_cache_key(profile_id)
        response: Final = await self._execute(_PROFILE_STATE_SCRIPT, (cache_key,), ("read",))
        return self._parse_profile_state_snapshot(response)

    async def mark_profile_available(self, profile_id: str) -> None:
        """Unconditionally mark a valid record available.

        A half-open probe should normally use ``complete_profile_half_open`` so a
        stale request cannot overwrite a newer state.
        """

        cache_key: Final = self.get_profile_state_cache_key(profile_id)
        response: Final = await self._execute(
            _PROFILE_STATE_SCRIPT,
            (cache_key,),
            ("mark_available",),
        )
        self._require_profile_state_update(response)

    async def mark_profile_cooldown(
        self,
        profile_id: str,
        reset_at: int,
        reason_code: str,
    ) -> None:
        cache_key: Final = self.get_profile_state_cache_key(profile_id)
        validated_reset_at: Final = self._validate_timestamp(reset_at, "reset_at")
        validated_reason: Final = self._validate_reason_code(reason_code)
        response: Final = await self._execute(
            _PROFILE_STATE_SCRIPT,
            (cache_key,),
            ("mark_cooldown", str(validated_reset_at), validated_reason),
        )
        self._require_profile_state_update(response)

    async def disable_profile(self, profile_id: str, reason_code: str) -> None:
        cache_key: Final = self.get_profile_state_cache_key(profile_id)
        validated_reason: Final = self._validate_reason_code(reason_code)
        response: Final = await self._execute(
            _PROFILE_STATE_SCRIPT,
            (cache_key,),
            ("disable", validated_reason),
        )
        self._require_profile_state_update(response)

    async def acquire_profile_half_open_lease(
        self,
        profile_id: str,
        lease_seconds: int = DEFAULT_HALF_OPEN_LEASE_SECONDS,
    ) -> OpenAIProfileHalfOpenLease | None:
        """Acquire the sole probe lease after cooldown or an abandoned probe."""

        validated_profile_id: Final = self._validate_profile_id(profile_id)
        cache_key: Final = self.get_profile_state_cache_key(validated_profile_id)
        validated_lease_seconds: Final = self._validate_half_open_lease_seconds(lease_seconds)
        lease_token: Final = secrets.token_hex(16)
        response: Final = await self._execute(
            _PROFILE_STATE_SCRIPT,
            (cache_key,),
            ("acquire_half_open", lease_token, str(validated_lease_seconds)),
        )
        if len(response) == 2 and response[0] == "not_acquired":
            try:
                OpenAIProfileState(response[1])
            except ValueError as exc:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription profile lease returned invalid data"
                ) from exc
            return None
        if len(response) != 2 or response[0] != "acquired":
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile lease returned invalid data")
        try:
            expires_at: Final = self._validate_timestamp(int(response[1]), "lease_expires_at")
        except (TypeError, ValueError) as exc:
            raise OpenAISubscriptionAffinityStoreError(
                "OpenAI subscription profile lease returned invalid data"
            ) from exc
        return OpenAIProfileHalfOpenLease(
            profile_id=validated_profile_id,
            token=lease_token,
            expires_at=expires_at,
        )

    async def complete_profile_half_open(
        self,
        profile_id: str,
        lease_token: str,
        target_state: OpenAIProfileState,
        *,
        reset_at: int | None = None,
        reason_code: str | None = None,
    ) -> bool:
        """Apply a probe result only while the caller still owns its lease."""

        cache_key: Final = self.get_profile_state_cache_key(profile_id)
        validated_token: Final = self._validate_half_open_lease_token(lease_token)
        if not isinstance(target_state, OpenAIProfileState) or target_state is OpenAIProfileState.HALF_OPEN:
            raise ValueError("target_state must be available, cooldown or disabled")

        validated_reset_at = ""
        validated_reason = ""
        if target_state is OpenAIProfileState.AVAILABLE:
            if reset_at is not None or reason_code is not None:
                raise ValueError("available target_state does not accept reset_at or reason_code")
        elif target_state is OpenAIProfileState.COOLDOWN:
            if reset_at is None or reason_code is None:
                raise ValueError("cooldown target_state requires reset_at and reason_code")
            validated_reset_at = str(self._validate_timestamp(reset_at, "reset_at"))
            validated_reason = self._validate_reason_code(reason_code)
        else:
            if reset_at is not None or reason_code is None:
                raise ValueError("disabled target_state requires reason_code without reset_at")
            validated_reason = self._validate_reason_code(reason_code)

        response: Final = await self._execute(
            _PROFILE_STATE_SCRIPT,
            (cache_key,),
            (
                "complete_half_open",
                validated_token,
                target_state.value,
                validated_reset_at,
                validated_reason,
            ),
        )
        if response == ("applied",):
            return True
        if response == ("stale",):
            return False
        raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile completion returned invalid data")

    async def _read_snapshot(self, user_api_key_hash: str) -> _AffinitySnapshot | None:
        cache_key: Final = self.get_cache_key(user_api_key_hash)
        response: Final = await self._execute(_AFFINITY_STORE_SCRIPT, (cache_key,), ("read",))
        if response == ("missing",):
            return None
        if len(response) != 3 or response[0] != "found":
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity read returned invalid data")

        try:
            profile_id: Final = self._validate_profile_id(response[1])
            ttl_seconds: Final = int(response[2])
        except (TypeError, ValueError) as exc:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity contains invalid data") from exc
        if ttl_seconds < 0:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity has no valid expiry")
        return _AffinitySnapshot(profile_id=profile_id, ttl_seconds=ttl_seconds)

    @classmethod
    def _parse_profile_state_snapshot(
        cls,
        response: tuple[str, ...],
    ) -> OpenAIProfileStateSnapshot:
        if response == ("missing",):
            return OpenAIProfileStateSnapshot(
                state=OpenAIProfileState.AVAILABLE,
                reset_at=None,
                lease_expires_at=None,
                reason_code=None,
                updated_at=None,
                persisted=False,
            )
        if len(response) != 6 or response[0] != "found":
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile state returned invalid data")

        try:
            state: Final = OpenAIProfileState(response[1])
            reset_at: Final = cls._parse_optional_timestamp(response[2], "reset_at")
            lease_expires_at: Final = cls._parse_optional_timestamp(
                response[3],
                "lease_expires_at",
            )
            reason_code: Final = cls._validate_reason_code(response[4]) if response[4] else None
            updated_at: Final = cls._validate_timestamp(int(response[5]), "updated_at")
        except (TypeError, ValueError) as exc:
            raise OpenAISubscriptionAffinityStoreError(
                "OpenAI subscription profile state contains invalid data"
            ) from exc

        if state is OpenAIProfileState.AVAILABLE:
            valid_shape = reset_at is None and lease_expires_at is None and reason_code is None
        elif state is OpenAIProfileState.COOLDOWN:
            valid_shape = reset_at is not None and lease_expires_at is None and reason_code is not None
        elif state is OpenAIProfileState.HALF_OPEN:
            valid_shape = reset_at is not None and lease_expires_at is not None and reason_code is not None
        else:
            valid_shape = reset_at is None and lease_expires_at is None and reason_code is not None
        if not valid_shape:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile state contains invalid data")

        return OpenAIProfileStateSnapshot(
            state=state,
            reset_at=reset_at,
            lease_expires_at=lease_expires_at,
            reason_code=reason_code,
            updated_at=updated_at,
            persisted=True,
        )

    @staticmethod
    def _require_profile_state_update(response: tuple[str, ...]) -> None:
        if response != ("updated",):
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription profile state update returned invalid data")

    async def _execute(
        self,
        source: str,
        cache_keys: Sequence[str],
        args: Sequence[str],
    ) -> tuple[str, ...]:
        redis_cache: Final = self._get_redis_cache()
        try:
            script: Final = redis_cache.async_register_script(source)
            raw_response: Final = await script(keys=cache_keys, args=args, client=None)
        except Exception as exc:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity storage is unavailable") from exc

        if not isinstance(raw_response, (list, tuple)):
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity operation returned invalid data")

        decoded: list[str] = []
        for item in raw_response:
            if isinstance(item, bytes):
                try:
                    decoded.append(item.decode("utf-8"))
                except UnicodeDecodeError as exc:
                    raise OpenAISubscriptionAffinityStoreError(
                        "OpenAI subscription affinity operation returned invalid data"
                    ) from exc
            elif isinstance(item, str):
                decoded.append(item)
            else:
                raise OpenAISubscriptionAffinityStoreError(
                    "OpenAI subscription affinity operation returned invalid data"
                )
        return tuple(decoded)

    def _get_redis_cache(self) -> _RedisScriptCache:
        redis_cache: Final = self.cache.redis_cache
        if redis_cache is None or not hasattr(redis_cache, "async_register_script"):
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity requires Redis")
        return cast(_RedisScriptCache, redis_cache)

    @staticmethod
    def _validate_user_api_key_hash(user_api_key_hash: str) -> str:
        if not isinstance(user_api_key_hash, str) or _SHA256_HEX_PATTERN.fullmatch(user_api_key_hash) is None:
            raise ValueError("user_api_key_hash must be a SHA-256 hexadecimal digest")
        return user_api_key_hash.lower()

    @staticmethod
    def _validate_profile_id(profile_id: str) -> str:
        if not isinstance(profile_id, str) or _PROFILE_ID_PATTERN.fullmatch(profile_id) is None:
            raise ValueError("profile_id has an invalid format")
        return profile_id

    @classmethod
    def _validate_timestamp(cls, timestamp: int, field_name: str) -> int:
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp <= 0
            or timestamp > cls.MAX_TIMESTAMP
        ):
            raise ValueError(f"{field_name} must be a valid positive Unix timestamp")
        return timestamp

    @classmethod
    def _parse_optional_timestamp(cls, value: str, field_name: str) -> int | None:
        if not value:
            return None
        return cls._validate_timestamp(int(value), field_name)

    @staticmethod
    def _validate_reason_code(reason_code: str) -> str:
        if not isinstance(reason_code, str) or _PROFILE_REASON_PATTERN.fullmatch(reason_code) is None:
            raise ValueError("reason_code has an invalid format")
        return reason_code

    @classmethod
    def _validate_half_open_lease_seconds(cls, lease_seconds: int) -> int:
        if (
            not isinstance(lease_seconds, int)
            or isinstance(lease_seconds, bool)
            or lease_seconds <= 0
            or lease_seconds > cls.MAX_HALF_OPEN_LEASE_SECONDS
        ):
            raise ValueError("lease_seconds must be a positive integer no greater than 3600")
        return lease_seconds

    @staticmethod
    def _validate_half_open_lease_token(lease_token: str) -> str:
        if not isinstance(lease_token, str) or _HALF_OPEN_LEASE_TOKEN_PATTERN.fullmatch(lease_token) is None:
            raise ValueError("lease_token has an invalid format")
        return lease_token

    @classmethod
    def _normalize_profile_ids(cls, profile_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(profile_ids, (str, bytes)):
            raise ValueError("available_profile_ids must be a sequence of profile identifiers")
        validated_profiles: Final = {cls._validate_profile_id(profile_id) for profile_id in profile_ids}
        if not validated_profiles:
            raise ValueError("available_profile_ids must contain at least one profile")
        return tuple(sorted(validated_profiles))

    @staticmethod
    def _validate_ttl(ttl_seconds: int) -> int:
        if not isinstance(ttl_seconds, int) or isinstance(ttl_seconds, bool) or ttl_seconds <= 0:
            raise ValueError("ttl_seconds must be a positive integer")
        return ttl_seconds
