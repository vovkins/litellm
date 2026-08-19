from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import httpx

from litellm.router_utils.openai_subscription_metrics import (
    OpenAISubscriptionLimitObservation,
    extract_openai_subscription_limit_observations,
)


def test_limit_observation_runtime_has_no_network_client_or_usage_endpoint() -> None:
    runtime_root = Path(__file__).parents[3] / "litellm"
    runtime_paths = [
        runtime_root / "router_utils" / "openai_subscription_metrics.py",
        runtime_root / "router_utils" / "openai_subscription_affinity.py",
        runtime_root / "router_utils" / "openai_subscription_failure_classifier.py",
        runtime_root / "router_utils" / "pre_call_checks" / "openai_subscription_affinity_check.py",
        *sorted((runtime_root / "llms" / "chatgpt").rglob("*.py")),
    ]
    forbidden_urls = (
        "/api/codex/usage",
        "/backend-api/codex/usage",
        "/wham/usage",
        "/account/ratelimits",
        "/rate_limits/read",
    )

    for runtime_path in runtime_paths:
        source = runtime_path.read_text(encoding="utf-8")
        lowered_source = source.lower()
        assert all(forbidden_url not in lowered_source for forbidden_url in forbidden_urls)

    observation_tree = ast.parse(runtime_paths[0].read_text(encoding="utf-8"))
    imported_modules = {
        node.names[0].name.split(".", maxsplit=1)[0]
        for node in ast.walk(observation_tree)
        if isinstance(node, ast.Import)
    } | {
        node.module.split(".", maxsplit=1)[0]
        for node in ast.walk(observation_tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert imported_modules.isdisjoint({"aiohttp", "httpx", "requests"})


def test_extracts_primary_and_secondary_windows_from_raw_response_headers() -> None:
    response = SimpleNamespace(
        _response_headers=httpx.Headers(
            {
                "X-Codex-Primary-Used-Percent": "12.5",
                "X-Codex-Primary-Window-Minutes": "300",
                "X-Codex-Primary-Reset-At": "1700000300",
                "X-Codex-Secondary-Used-Percent": "80",
                "X-Codex-Secondary-Window-Minutes": "10080",
                "X-Codex-Secondary-Reset-At": "1700604800",
            }
        )
    )

    assert extract_openai_subscription_limit_observations(
        kwargs={},
        response_obj=response,
    ) == (
        OpenAISubscriptionLimitObservation(
            window="primary",
            used_ratio=0.125,
            reset_timestamp_seconds=1700000300,
            window_seconds=18000,
        ),
        OpenAISubscriptionLimitObservation(
            window="secondary",
            used_ratio=0.8,
            reset_timestamp_seconds=1700604800,
            window_seconds=604800,
        ),
    )


def test_extracts_processed_headers_from_standard_logging_payload() -> None:
    kwargs = {
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    "llm_provider-x-codex-primary-used-percent": "42",
                    "llm_provider-x-codex-primary-window-minutes": "60",
                }
            }
        }
    }

    assert extract_openai_subscription_limit_observations(
        kwargs=kwargs,
        response_obj={},
    ) == (
        OpenAISubscriptionLimitObservation(
            window="primary",
            used_ratio=0.42,
            reset_timestamp_seconds=None,
            window_seconds=3600,
        ),
    )


def test_extracts_headers_preserved_in_streaming_model_call_details() -> None:
    kwargs = {
        "response_headers": {
            "x-codex-primary-used-percent": "55",
            "x-codex-primary-reset-at": "1700000300",
        }
    }

    assert extract_openai_subscription_limit_observations(
        kwargs=kwargs,
        response_obj={},
    ) == (
        OpenAISubscriptionLimitObservation(
            window="primary",
            used_ratio=0.55,
            reset_timestamp_seconds=1700000300,
            window_seconds=None,
        ),
    )


def test_extracts_failure_headers_from_mapped_exception() -> None:
    error = RuntimeError("provider body is not inspected")
    error.response = httpx.Response(  # type: ignore[attr-defined]
        status_code=429,
        headers={
            "x-codex-secondary-used-percent": "100",
            "x-codex-secondary-reset-at": "1700604800",
        },
        request=httpx.Request("POST", "https://example.invalid"),
    )

    assert extract_openai_subscription_limit_observations(
        kwargs={},
        response_obj={},
        error=error,
    ) == (
        OpenAISubscriptionLimitObservation(
            window="secondary",
            used_ratio=1.0,
            reset_timestamp_seconds=1700604800,
            window_seconds=None,
        ),
    )


def test_raw_response_headers_take_precedence_over_processed_duplicates() -> None:
    response = SimpleNamespace(
        _response_headers={"x-codex-primary-used-percent": "10"},
        _hidden_params={
            "additional_headers": {
                "llm_provider-x-codex-primary-used-percent": "99",
            }
        },
    )

    observation = extract_openai_subscription_limit_observations(
        kwargs={},
        response_obj=response,
    )

    assert observation[0].used_ratio == 0.1


def test_invalid_fields_are_ignored_without_discarding_valid_fields() -> None:
    response = SimpleNamespace(
        _response_headers={
            "x-codex-primary-used-percent": "nan",
            "x-codex-primary-window-minutes": "0",
            "x-codex-primary-reset-at": "1700000300",
            "x-codex-secondary-used-percent": "101",
            "x-codex-secondary-window-minutes": "1.5",
            "x-codex-secondary-reset-at": "-1",
        }
    )

    assert extract_openai_subscription_limit_observations(
        kwargs={},
        response_obj=response,
    ) == (
        OpenAISubscriptionLimitObservation(
            window="primary",
            used_ratio=None,
            reset_timestamp_seconds=1700000300,
            window_seconds=None,
        ),
    )


def test_unrelated_headers_and_response_body_are_not_parsed() -> None:
    response = {
        "x-codex-primary-used-percent": "50",
        "rate_limits": {"primary": {"used_percent": 50}},
        "_hidden_params": {
            "headers": {
                "authorization": "Bearer secret",
                "x-ratelimit-remaining-tokens": "10",
            }
        },
    }

    assert (
        extract_openai_subscription_limit_observations(
            kwargs={},
            response_obj=response,
        )
        == ()
    )


def test_malformed_header_container_is_ignored() -> None:
    class BrokenHeaders:
        def items(self):
            raise RuntimeError("must not escape")

    response = SimpleNamespace(_response_headers=BrokenHeaders())

    assert (
        extract_openai_subscription_limit_observations(
            kwargs={},
            response_obj=response,
        )
        == ()
    )


def test_oversized_numeric_header_is_ignored() -> None:
    response = SimpleNamespace(
        _response_headers={"x-codex-primary-reset-at": "9" * 10000},
    )

    assert (
        extract_openai_subscription_limit_observations(
            kwargs={},
            response_obj=response,
        )
        == ()
    )
