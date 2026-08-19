"""Redis-backed affinity between a LiteLLM virtual key and an OpenAI OAuth profile.

This module only owns persistence. Selection, failover, and TTL refresh policy are
implemented by the routing layer that consumes this store.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
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

_ASSIGN_PROFILE_SCRIPT: Final = """
local current = redis.call('GET', KEYS[1])
if current ~= false then
  return {'existing', current, tostring(redis.call('TTL', KEYS[1]))}
end

local ttl = tonumber(ARGV[1])
local profile_count = tonumber(ARGV[2])
if ttl == nil or ttl <= 0 or profile_count == nil or profile_count <= 0 then
  return redis.error_reply('invalid affinity assignment arguments')
end

local sequence = redis.call('INCR', KEYS[2])
local profile_index = ((sequence - 1) % profile_count) + 1
local selected_profile = ARGV[profile_index + 2]
redis.call('SET', KEYS[1], selected_profile, 'EX', ttl)
return {'assigned', selected_profile, tostring(ttl)}
"""

_SHA256_HEX_PATTERN: Final = re.compile(r"^[0-9a-fA-F]{64}$")
_PROFILE_ID_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


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


@dataclass(frozen=True, slots=True)
class _AffinitySnapshot:
    profile_id: str
    ttl_seconds: int


class OpenAISubscriptionAffinityStore:
    """Persist one global OpenAI OAuth profile binding per virtual-key hash.

    The store deliberately bypasses ``DualCache``'s in-memory tier. Affinity must
    be shared by every proxy replica, and a Redis failure must be visible to the
    future routing layer instead of producing conflicting pod-local decisions.
    ``DualCache`` is still the owner of the Redis connection and connection pool.
    """

    CACHE_KEY_PREFIX: Final = "openai_subscription_affinity:{v1}"
    COUNTER_CACHE_KEY: Final = f"{CACHE_KEY_PREFIX}:counter"
    DEFAULT_TTL_SECONDS: Final = 24 * 60 * 60

    def __init__(self, cache: DualCache) -> None:
        self.cache = cache

    @classmethod
    def get_cache_key(cls, user_api_key_hash: str) -> str:
        normalized_hash: Final = cls._validate_user_api_key_hash(user_api_key_hash)
        return f"{cls.CACHE_KEY_PREFIX}:binding:{normalized_hash}"

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
        """Return an existing binding or atomically assign the next profile.

        Profile order supplied by callers cannot affect distribution: identifiers
        are validated, de-duplicated, and sorted before the Redis script runs.
        An existing binding is returned as-is and its TTL is never refreshed here.
        Consequently, adding a profile grows the pool only for new or expired
        bindings instead of redistributing active virtual keys.
        """

        cache_key: Final = self.get_cache_key(user_api_key_hash)
        profiles: Final = self._normalize_profile_ids(available_profile_ids)
        validated_ttl: Final = self._validate_ttl(ttl_seconds)
        response: Final = await self._execute(
            _ASSIGN_PROFILE_SCRIPT,
            (cache_key, self.COUNTER_CACHE_KEY),
            (str(validated_ttl), str(len(profiles)), *profiles),
        )
        if len(response) != 3 or response[0] not in {"existing", "assigned"}:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity assignment returned invalid data")

        try:
            profile_id: Final = self._validate_profile_id(response[1])
            remaining_ttl: Final = int(response[2])
        except (TypeError, ValueError) as exc:
            raise OpenAISubscriptionAffinityStoreError(
                "OpenAI subscription affinity assignment returned invalid data"
            ) from exc
        if remaining_ttl < 0:
            raise OpenAISubscriptionAffinityStoreError("OpenAI subscription affinity has no valid expiry")
        return profile_id

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
