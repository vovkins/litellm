from __future__ import annotations

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
)

PROFILE = "openai-oauth-1"


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    yield
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)


def sample_value(name: str, **labels: str) -> float | None:
    return REGISTRY.get_sample_value(name, labels=labels)


def test_metrics_are_declared_with_bounded_labels() -> None:
    metric_names = {
        "ru_llm_proxy_openai_subscription_available",
        "ru_llm_proxy_openai_subscription_limit_used_ratio",
        "ru_llm_proxy_openai_subscription_limit_remaining_ratio",
        "ru_llm_proxy_openai_subscription_limit_reset_timestamp_seconds",
        "ru_llm_proxy_openai_subscription_limit_window_seconds",
        "ru_llm_proxy_openai_subscription_last_observation_timestamp_seconds",
        "ru_llm_proxy_openai_subscription_failovers_total",
        "ru_llm_proxy_openai_subscription_auth_errors_total",
    }

    assert metric_names <= set(DEFINED_PROMETHEUS_METRICS.__args__)
    for metric_name in metric_names:
        labels = PrometheusMetricLabels.get_labels(metric_name)  # type: ignore[arg-type]
        assert labels in (["profile"], ["profile", "window"])
        assert not ({"hashed_api_key", "account_id", "model", "path"} & set(labels))


def test_custom_prometheus_labels_cannot_expand_subscription_metric_cardinality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["request_id"])
    monkeypatch.setattr(litellm, "custom_prometheus_tags", ["customer"])

    assert PrometheusMetricLabels.get_labels("ru_llm_proxy_openai_subscription_available") == ["profile"]
    assert PrometheusMetricLabels.get_labels("ru_llm_proxy_openai_subscription_limit_used_ratio") == [
        "profile",
        "window",
    ]


def test_initialization_exposes_state_and_explicitly_marks_limit_data_missing() -> None:
    logger = PrometheusLogger()

    logger.initialize_openai_subscription_profile_metrics(PROFILE, available=True)

    assert sample_value("ru_llm_proxy_openai_subscription_available", profile=PROFILE) == 1
    assert sample_value("ru_llm_proxy_openai_subscription_failovers_total", profile=PROFILE) == 0
    assert sample_value("ru_llm_proxy_openai_subscription_auth_errors_total", profile=PROFILE) == 0
    for window in ("primary", "secondary"):
        assert (
            sample_value(
                "ru_llm_proxy_openai_subscription_last_observation_timestamp_seconds",
                profile=PROFILE,
                window=window,
            )
            == 0
        )
        assert (
            sample_value(
                "ru_llm_proxy_openai_subscription_limit_used_ratio",
                profile=PROFILE,
                window=window,
            )
            is None
        )


def test_observation_sets_only_present_values_and_derived_remaining_ratio() -> None:
    logger = PrometheusLogger()
    logger.initialize_openai_subscription_profile_metrics(PROFILE, available=True)

    logger.observe_openai_subscription_limit_window(
        profile=PROFILE,
        window="primary",
        observed_at=1700000000.5,
        used_ratio=0.42,
        reset_timestamp_seconds=1700000300,
        window_seconds=None,
    )

    labels = {"profile": PROFILE, "window": "primary"}
    assert sample_value("ru_llm_proxy_openai_subscription_limit_used_ratio", **labels) == 0.42
    assert sample_value("ru_llm_proxy_openai_subscription_limit_remaining_ratio", **labels) == pytest.approx(0.58)
    assert sample_value("ru_llm_proxy_openai_subscription_limit_reset_timestamp_seconds", **labels) == 1700000300
    assert sample_value("ru_llm_proxy_openai_subscription_limit_window_seconds", **labels) is None
    assert sample_value("ru_llm_proxy_openai_subscription_last_observation_timestamp_seconds", **labels) == 1700000000.5


def test_partial_and_empty_observations_preserve_last_valid_values() -> None:
    logger = PrometheusLogger()
    logger.initialize_openai_subscription_profile_metrics(PROFILE, available=True)
    labels = {"profile": PROFILE, "window": "primary"}

    logger.observe_openai_subscription_limit_window(
        profile=PROFILE,
        window="primary",
        observed_at=1700000000,
        used_ratio=0.25,
        reset_timestamp_seconds=1700000300,
        window_seconds=18000,
    )
    logger.observe_openai_subscription_limit_window(
        profile=PROFILE,
        window="primary",
        observed_at=1700000010,
        used_ratio=0.4,
        reset_timestamp_seconds=None,
        window_seconds=None,
    )
    logger.observe_openai_subscription_limit_window(
        profile=PROFILE,
        window="primary",
        observed_at=1700000020,
        used_ratio=None,
        reset_timestamp_seconds=None,
        window_seconds=None,
    )

    assert sample_value("ru_llm_proxy_openai_subscription_limit_used_ratio", **labels) == 0.4
    assert sample_value("ru_llm_proxy_openai_subscription_limit_remaining_ratio", **labels) == pytest.approx(0.6)
    assert sample_value("ru_llm_proxy_openai_subscription_limit_reset_timestamp_seconds", **labels) == 1700000300
    assert sample_value("ru_llm_proxy_openai_subscription_limit_window_seconds", **labels) == 18000
    assert sample_value("ru_llm_proxy_openai_subscription_last_observation_timestamp_seconds", **labels) == 1700000010


def test_profile_state_and_counters_are_updated_independently() -> None:
    logger = PrometheusLogger()
    logger.initialize_openai_subscription_profile_metrics(PROFILE, available=True)

    logger.set_openai_subscription_profile_available(PROFILE, False)
    logger.increment_openai_subscription_failover(PROFILE)
    logger.increment_openai_subscription_auth_error(PROFILE)
    logger.increment_openai_subscription_auth_error(PROFILE)

    assert sample_value("ru_llm_proxy_openai_subscription_available", profile=PROFILE) == 0
    assert sample_value("ru_llm_proxy_openai_subscription_failovers_total", profile=PROFILE) == 1
    assert sample_value("ru_llm_proxy_openai_subscription_auth_errors_total", profile=PROFILE) == 2


def test_profile_gauges_use_most_recent_multiprocess_mode() -> None:
    logger = PrometheusLogger()

    assert logger.ru_llm_proxy_openai_subscription_available._multiprocess_mode == "mostrecent"
    assert logger.ru_llm_proxy_openai_subscription_limit_used_ratio._multiprocess_mode == "mostrecent"
    assert logger.ru_llm_proxy_openai_subscription_last_observation_timestamp_seconds._multiprocess_mode == "mostrecent"


@pytest.mark.parametrize(
    ("method_name", "kwargs"),
    [
        ("initialize_openai_subscription_profile_metrics", {"profile": "unsafe profile", "available": True}),
        ("set_openai_subscription_profile_available", {"profile": "UPPER", "available": True}),
        ("increment_openai_subscription_failover", {"profile": "Bearer-secret"}),
        ("increment_openai_subscription_auth_error", {"profile": "-profile"}),
        (
            "observe_openai_subscription_limit_window",
            {
                "profile": PROFILE,
                "window": "daily",
                "observed_at": 1,
                "used_ratio": 0.5,
                "reset_timestamp_seconds": None,
                "window_seconds": None,
            },
        ),
    ],
)
def test_invalid_or_sensitive_labels_are_rejected(method_name: str, kwargs: dict) -> None:
    logger = PrometheusLogger()

    with pytest.raises(ValueError):
        getattr(logger, method_name)(**kwargs)


@pytest.mark.parametrize("used_ratio", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_observation_values_are_rejected(used_ratio: float) -> None:
    logger = PrometheusLogger()

    with pytest.raises(ValueError):
        logger.observe_openai_subscription_limit_window(
            profile=PROFILE,
            window="primary",
            observed_at=1700000000,
            used_ratio=used_ratio,
            reset_timestamp_seconds=None,
            window_seconds=None,
        )
