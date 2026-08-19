from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

OpenAISubscriptionLimitWindow = Literal["primary", "secondary"]

_WINDOWS: Final[tuple[OpenAISubscriptionLimitWindow, ...]] = ("primary", "secondary")
_MAX_TIMESTAMP: Final = 253402300799
_MAX_WINDOW_MINUTES: Final = 10 * 365 * 24 * 60


@dataclass(frozen=True, slots=True)
class OpenAISubscriptionLimitObservation:
    window: OpenAISubscriptionLimitWindow
    used_ratio: float | None
    reset_timestamp_seconds: int | None
    window_seconds: int | None


def extract_openai_subscription_limit_observations(
    *,
    kwargs: Mapping[str, Any] | None,
    response_obj: object,
    error: BaseException | None = None,
) -> tuple[OpenAISubscriptionLimitObservation, ...]:
    """Parse documented Codex limit headers already attached to a response."""

    headers: dict[str, object] = {}
    for source in _iter_header_sources(kwargs=kwargs, response_obj=response_obj, error=error):
        for raw_name, value in _iter_header_items(source):
            name = raw_name.strip().lower()
            if name.startswith("llm_provider-"):
                name = name.removeprefix("llm_provider-")
            if name.startswith("x-codex-") and name not in headers:
                headers[name] = value

    observations: list[OpenAISubscriptionLimitObservation] = []
    for window in _WINDOWS:
        prefix = f"x-codex-{window}-"
        used_percent = _parse_percentage(headers.get(f"{prefix}used-percent"))
        window_minutes = _parse_positive_int(
            headers.get(f"{prefix}window-minutes"),
            maximum=_MAX_WINDOW_MINUTES,
        )
        reset_at = _parse_positive_int(
            headers.get(f"{prefix}reset-at"),
            maximum=_MAX_TIMESTAMP,
        )
        if used_percent is None and window_minutes is None and reset_at is None:
            continue
        observations.append(
            OpenAISubscriptionLimitObservation(
                window=window,
                used_ratio=used_percent / 100 if used_percent is not None else None,
                reset_timestamp_seconds=reset_at,
                window_seconds=window_minutes * 60 if window_minutes is not None else None,
            )
        )
    return tuple(observations)


def _iter_header_sources(
    *,
    kwargs: Mapping[str, Any] | None,
    response_obj: object,
    error: BaseException | None,
) -> Iterator[object]:
    yield _safe_getattr(response_obj, "_response_headers")
    yield from _iter_hidden_header_sources(response_obj)

    if error is not None:
        yield _safe_getattr(error, "headers")
        yield _safe_getattr(error, "litellm_response_headers")
        yield _safe_getattr(_safe_getattr(error, "response"), "headers")

    if kwargs is not None:
        yield kwargs.get("response_headers")
        yield from _iter_hidden_header_sources(kwargs.get("standard_logging_object"))


def _iter_hidden_header_sources(value: object) -> Iterator[object]:
    hidden_params = _safe_get(value, "_hidden_params")
    if hidden_params is None:
        hidden_params = _safe_get(value, "hidden_params")
    yield _safe_get(hidden_params, "headers")
    yield _safe_get(hidden_params, "additional_headers")


def _iter_header_items(source: object) -> Iterator[tuple[str, object]]:
    if source is None:
        return
    if isinstance(source, Mapping):
        items = source.items()
    else:
        try:
            items = source.items()  # type: ignore[union-attr]
        except Exception:
            return
    try:
        for key, value in items:
            if isinstance(key, str):
                yield key, value
    except Exception:
        return


def _safe_get(value: object, key: str) -> object:
    if isinstance(value, Mapping):
        try:
            return value.get(key)
        except Exception:
            return None
    return _safe_getattr(value, key)


def _safe_getattr(value: object, name: str) -> object:
    try:
        return getattr(value, name, None)
    except Exception:
        return None


def _parse_percentage(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(parsed) or not 0 <= parsed <= 100:
        return None
    return parsed


def _parse_positive_int(value: object, *, maximum: int) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped.isascii() or not stripped.isdecimal():
            return None
        try:
            parsed = int(stripped)
        except (ValueError, OverflowError):
            return None
    else:
        return None
    return parsed if 0 < parsed <= maximum else None
