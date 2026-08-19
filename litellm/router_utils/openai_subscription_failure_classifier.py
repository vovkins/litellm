"""Classify failures from one OpenAI subscription profile without side effects.

The classifier deliberately relies on exception types, HTTP status codes, and
the standard ``Retry-After`` header. Provider error text and response bodies are
not stable contracts and can contain sensitive data, so they are never parsed or
copied into the result.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from enum import Enum
from typing import Any, Final

import httpx
import openai

from litellm.exceptions import RateLimitErrorCategory, validate_rate_limit_category
from litellm.llms.chatgpt.common_utils import ChatGPTAuthError
from litellm.router_utils.openai_subscription_affinity import OpenAIProfileState


class OpenAISubscriptionFailureCategory(str, Enum):
    RATE_LIMIT = "rate_limit"
    OAUTH_ERROR = "oauth_error"
    MODEL_UNAVAILABLE = "model_unavailable"
    NETWORK_ERROR = "network_error"
    SERVER_ERROR = "server_error"
    UNKNOWN_ERROR = "unknown_error"


class OpenAISubscriptionFailureScope(str, Enum):
    PROFILE = "profile"
    MODEL = "model"
    PROVIDER = "provider"
    UNKNOWN = "unknown"


class OpenAIRecoveryTimeSource(str, Enum):
    RETRY_AFTER = "retry_after"
    DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class OpenAISubscriptionFailureClassification:
    category: OpenAISubscriptionFailureCategory
    scope: OpenAISubscriptionFailureScope
    reason_code: str
    target_profile_state: OpenAIProfileState | None
    retryable: bool
    failover_eligible: bool
    status_code: int | None
    retry_after_seconds: int | None = None
    reset_at: int | None = None
    recovery_time_source: OpenAIRecoveryTimeSource | None = None


DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS: Final = 60
MAX_RATE_LIMIT_COOLDOWN_SECONDS: Final = 7 * 24 * 60 * 60
MAX_TIMESTAMP: Final = 253402300799
_INTERNAL_RATE_LIMIT_CATEGORIES: Final = {
    RateLimitErrorCategory.LITELLM_RATE_LIMIT.value,
    RateLimitErrorCategory.LITELLM_BATCH_RATE_LIMIT.value,
}


def classify_openai_subscription_failure(
    error: BaseException,
    *,
    now: float | None = None,
) -> OpenAISubscriptionFailureClassification:
    """Return a safe, endpoint-independent classification for one failure."""

    if not isinstance(error, BaseException):
        raise TypeError("error must be an exception")
    current_time: Final = _normalize_current_time(now)
    status_code: Final = _extract_status_code(error)

    rate_limit_category: Final = validate_rate_limit_category(_safe_getattr(error, "category"))
    if rate_limit_category in _INTERNAL_RATE_LIMIT_CATEGORIES:
        return _unknown_classification(status_code)
    if isinstance(error, openai.RateLimitError) or status_code == 429:
        retry_after_seconds, source = _get_retry_after(error, current_time)
        reset_at: Final = min(current_time + retry_after_seconds, MAX_TIMESTAMP)
        return _classification(
            category=OpenAISubscriptionFailureCategory.RATE_LIMIT,
            scope=OpenAISubscriptionFailureScope.PROFILE,
            target_profile_state=OpenAIProfileState.COOLDOWN,
            retryable=True,
            failover_eligible=True,
            status_code=status_code,
            retry_after_seconds=reset_at - current_time,
            reset_at=reset_at,
            recovery_time_source=source,
        )

    if isinstance(error, ChatGPTAuthError):
        if status_code in {408, 504}:
            return _network_classification(status_code)
        if status_code is not None and 500 <= status_code <= 599:
            return _server_classification(status_code)
        return _oauth_classification(status_code)

    if isinstance(error, (openai.AuthenticationError, openai.PermissionDeniedError)) or status_code in {
        401,
        403,
    }:
        return _oauth_classification(status_code)

    if isinstance(error, openai.NotFoundError) or status_code == 404:
        return _classification(
            category=OpenAISubscriptionFailureCategory.MODEL_UNAVAILABLE,
            scope=OpenAISubscriptionFailureScope.MODEL,
            target_profile_state=None,
            retryable=False,
            failover_eligible=False,
            status_code=status_code,
        )

    if isinstance(error, (openai.APITimeoutError, httpx.TimeoutException)) or status_code in {408, 504}:
        return _network_classification(status_code)

    if isinstance(error, (openai.APIConnectionError, httpx.TransportError)):
        return _network_classification(status_code)

    if status_code is not None and 500 <= status_code <= 599:
        return _server_classification(status_code)

    return _unknown_classification(status_code)


def _oauth_classification(status_code: int | None) -> OpenAISubscriptionFailureClassification:
    return _classification(
        category=OpenAISubscriptionFailureCategory.OAUTH_ERROR,
        scope=OpenAISubscriptionFailureScope.PROFILE,
        target_profile_state=OpenAIProfileState.DISABLED,
        retryable=False,
        failover_eligible=True,
        status_code=status_code,
    )


def _network_classification(status_code: int | None) -> OpenAISubscriptionFailureClassification:
    return _classification(
        category=OpenAISubscriptionFailureCategory.NETWORK_ERROR,
        scope=OpenAISubscriptionFailureScope.PROVIDER,
        target_profile_state=None,
        retryable=True,
        failover_eligible=False,
        status_code=status_code,
    )


def _server_classification(status_code: int | None) -> OpenAISubscriptionFailureClassification:
    return _classification(
        category=OpenAISubscriptionFailureCategory.SERVER_ERROR,
        scope=OpenAISubscriptionFailureScope.PROVIDER,
        target_profile_state=None,
        retryable=True,
        failover_eligible=False,
        status_code=status_code,
    )


def _classification(
    *,
    category: OpenAISubscriptionFailureCategory,
    scope: OpenAISubscriptionFailureScope,
    target_profile_state: OpenAIProfileState | None,
    retryable: bool,
    failover_eligible: bool,
    status_code: int | None,
    retry_after_seconds: int | None = None,
    reset_at: int | None = None,
    recovery_time_source: OpenAIRecoveryTimeSource | None = None,
) -> OpenAISubscriptionFailureClassification:
    return OpenAISubscriptionFailureClassification(
        category=category,
        scope=scope,
        reason_code=category.value,
        target_profile_state=target_profile_state,
        retryable=retryable,
        failover_eligible=failover_eligible,
        status_code=status_code,
        retry_after_seconds=retry_after_seconds,
        reset_at=reset_at,
        recovery_time_source=recovery_time_source,
    )


def _unknown_classification(status_code: int | None) -> OpenAISubscriptionFailureClassification:
    return _classification(
        category=OpenAISubscriptionFailureCategory.UNKNOWN_ERROR,
        scope=OpenAISubscriptionFailureScope.UNKNOWN,
        target_profile_state=None,
        retryable=False,
        failover_eligible=False,
        status_code=status_code,
    )


def _normalize_current_time(now: float | None) -> int:
    value: Any = time.time() if now is None else now
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(value)
        or value <= 0
        or value >= MAX_TIMESTAMP
    ):
        raise ValueError("now must be a valid positive Unix timestamp")
    return int(value)


def _extract_status_code(error: BaseException) -> int | None:
    for value in (
        _safe_getattr(error, "status_code"),
        _safe_getattr(_safe_getattr(error, "response"), "status_code"),
    ):
        if isinstance(value, int) and not isinstance(value, bool) and 100 <= value <= 599:
            return value
    return None


def _get_retry_after(
    error: BaseException,
    current_time: int,
) -> tuple[int, OpenAIRecoveryTimeSource]:
    for headers in _iter_header_sources(error):
        value: Final = _get_header_value(headers, "retry-after")
        delay: Final = _parse_retry_after(value, current_time)
        if delay is not None:
            return delay, OpenAIRecoveryTimeSource.RETRY_AFTER
    return DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS, OpenAIRecoveryTimeSource.DEFAULT


def _iter_header_sources(error: BaseException) -> tuple[object, ...]:
    response: Final = _safe_getattr(error, "response")
    candidates: Final = (
        _safe_getattr(error, "headers"),
        _safe_getattr(error, "litellm_response_headers"),
        _safe_getattr(response, "headers"),
    )
    return tuple(candidate for candidate in candidates if candidate is not None)


def _get_header_value(headers: object, header_name: str) -> str | None:
    if not isinstance(headers, Mapping) and not hasattr(headers, "items"):
        return None
    try:
        items = headers.items()  # type: ignore[union-attr]
        for key, value in items:
            if isinstance(key, str) and key.lower() == header_name:
                return _normalize_header_value(value)
    except Exception:
        return None
    return None


def _normalize_header_value(value: object) -> str | None:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    return None


def _parse_retry_after(value: str | None, current_time: int) -> int | None:
    if not value:
        return None
    if value.isascii() and value.isdecimal():
        delay = int(value)
    else:
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at is None or retry_at.tzinfo is None:
            return None
        try:
            timestamp = retry_at.astimezone(timezone.utc).timestamp()
        except (OverflowError, OSError, ValueError):
            return None
        if not math.isfinite(timestamp):
            return None
        delay = math.ceil(timestamp - current_time)
    return min(max(delay, 1), MAX_RATE_LIMIT_COOLDOWN_SECONDS)


def _safe_getattr(value: object, name: str) -> object | None:
    try:
        return getattr(value, name, None)
    except Exception:
        return None
