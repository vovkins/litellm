from __future__ import annotations

from datetime import datetime, timezone
from email.utils import format_datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.exceptions import AuthenticationError, RateLimitError, RateLimitErrorCategory
from litellm.llms.chatgpt.chat.transformation import ChatGPTConfig
from litellm.llms.chatgpt.common_utils import GetAccessTokenError
from litellm.llms.chatgpt.responses.transformation import ChatGPTResponsesAPIConfig
from litellm.llms.openai.common_utils import OpenAIError
from litellm.router_utils.openai_subscription_affinity import OpenAIProfileState
from litellm.router_utils.openai_subscription_failure_classifier import (
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS,
    MAX_RATE_LIMIT_COOLDOWN_SECONDS,
    OpenAIRecoveryTimeSource,
    OpenAISubscriptionFailureCategory,
    OpenAISubscriptionFailureScope,
    classify_openai_subscription_failure,
)

NOW = 1_700_000_000


@pytest.fixture(autouse=True)
def isolate_chatgpt_auth_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHATGPT_TOKEN_DIR", str(tmp_path / "chatgpt-auth"))
    monkeypatch.delenv("CHATGPT_AUTH_FILE", raising=False)


class StatusError(Exception):
    def __init__(
        self,
        status_code: Any,
        *,
        headers: object | None = None,
        response: object | None = None,
        message: str = "provider error",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers
        self.response = response


def test_authentication_error_disables_only_the_profile_and_allows_failover() -> None:
    error = AuthenticationError(
        message="expired secret-token-value",
        llm_provider="chatgpt",
        model="gpt-5.4",
    )

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.OAUTH_ERROR
    assert result.scope is OpenAISubscriptionFailureScope.PROFILE
    assert result.target_profile_state is OpenAIProfileState.DISABLED
    assert result.retryable is False
    assert result.failover_eligible is True
    assert result.status_code == 401
    assert "secret-token-value" not in repr(result)


@pytest.mark.parametrize("status_code", [401, 403])
def test_authentication_statuses_disable_profile(status_code: int) -> None:
    result = classify_openai_subscription_failure(StatusError(status_code), now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.OAUTH_ERROR
    assert result.target_profile_state is OpenAIProfileState.DISABLED


def test_chatgpt_authentication_failure_is_profile_scoped_even_with_generic_status() -> None:
    result = classify_openai_subscription_failure(
        GetAccessTokenError(status_code=400, message="refresh failed secret-token-value"),
        now=NOW,
    )

    assert result.category is OpenAISubscriptionFailureCategory.OAUTH_ERROR
    assert result.scope is OpenAISubscriptionFailureScope.PROFILE
    assert result.target_profile_state is OpenAIProfileState.DISABLED
    assert "secret-token-value" not in repr(result)


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (429, OpenAISubscriptionFailureCategory.RATE_LIMIT),
        (408, OpenAISubscriptionFailureCategory.NETWORK_ERROR),
        (503, OpenAISubscriptionFailureCategory.SERVER_ERROR),
    ],
)
def test_transient_oauth_endpoint_failure_does_not_disable_profile(
    status_code: int,
    expected_category: OpenAISubscriptionFailureCategory,
) -> None:
    result = classify_openai_subscription_failure(
        GetAccessTokenError(status_code=status_code, message="temporary OAuth failure"),
        now=NOW,
    )

    assert result.category is expected_category
    assert result.target_profile_state is not OpenAIProfileState.DISABLED


def test_vendor_rate_limit_enters_cooldown_with_retry_after_seconds() -> None:
    error = RateLimitError(
        message="do not parse quota text",
        llm_provider="chatgpt",
        model="gpt-5.4",
        response=httpx.Response(
            429,
            headers={"Retry-After": "120"},
            request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
        ),
    )

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.RATE_LIMIT
    assert result.scope is OpenAISubscriptionFailureScope.PROFILE
    assert result.target_profile_state is OpenAIProfileState.COOLDOWN
    assert result.retryable is True
    assert result.failover_eligible is True
    assert result.retry_after_seconds == 120
    assert result.reset_at == NOW + 120
    assert result.recovery_time_source is OpenAIRecoveryTimeSource.RETRY_AFTER


def test_rate_limit_accepts_http_date_case_insensitively() -> None:
    retry_at = datetime.fromtimestamp(NOW + 90, tz=timezone.utc)
    error = StatusError(429, headers={"rEtRy-AfTeR": format_datetime(retry_at, usegmt=True)})

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.retry_after_seconds == 90
    assert result.reset_at == NOW + 90
    assert result.recovery_time_source is OpenAIRecoveryTimeSource.RETRY_AFTER


@pytest.mark.parametrize("source", ["response", "litellm_response_headers"])
def test_retry_after_survives_supported_exception_header_sources(source: str) -> None:
    if source == "response":
        error = StatusError(
            429,
            response=httpx.Response(
                429,
                headers={"Retry-After": "75"},
                request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
            ),
        )
    else:
        error = StatusError(429)
        error.litellm_response_headers = {"Retry-After": "75"}

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.retry_after_seconds == 75
    assert result.recovery_time_source is OpenAIRecoveryTimeSource.RETRY_AFTER


@pytest.mark.parametrize("header_value", [None, "", "invalid", "-1", True, ["60"]])
def test_missing_or_malformed_retry_after_uses_bounded_default(header_value: object) -> None:
    headers = {} if header_value is None else {"Retry-After": header_value}

    result = classify_openai_subscription_failure(StatusError(429, headers=headers), now=NOW)

    assert result.retry_after_seconds == DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    assert result.reset_at == NOW + DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
    assert result.recovery_time_source is OpenAIRecoveryTimeSource.DEFAULT


def test_retry_after_is_bounded_and_never_creates_immediate_probe_loop() -> None:
    zero = classify_openai_subscription_failure(StatusError(429, headers={"Retry-After": "0"}), now=NOW)
    excessive = classify_openai_subscription_failure(
        StatusError(429, headers={"Retry-After": str(MAX_RATE_LIMIT_COOLDOWN_SECONDS * 10)}),
        now=NOW,
    )

    assert zero.retry_after_seconds == 1
    assert excessive.retry_after_seconds == MAX_RATE_LIMIT_COOLDOWN_SECONDS


def test_internal_litellm_rate_limit_does_not_penalize_oauth_profile() -> None:
    error = RateLimitError(
        message="virtual key budget",
        llm_provider="litellm",
        model="gpt-5.4",
        category=RateLimitErrorCategory.LITELLM_RATE_LIMIT,
        headers={"Retry-After": "60"},
    )

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.UNKNOWN_ERROR
    assert result.target_profile_state is None
    assert result.failover_eligible is False
    assert result.retry_after_seconds is None


def test_model_not_found_does_not_disable_the_subscription() -> None:
    result = classify_openai_subscription_failure(StatusError(404), now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.MODEL_UNAVAILABLE
    assert result.scope is OpenAISubscriptionFailureScope.MODEL
    assert result.target_profile_state is None
    assert result.retryable is False
    assert result.failover_eligible is False


@pytest.mark.parametrize("status_code", [408, 504])
def test_timeout_statuses_are_provider_scoped_and_retryable(status_code: int) -> None:
    result = classify_openai_subscription_failure(StatusError(status_code), now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.NETWORK_ERROR
    assert result.scope is OpenAISubscriptionFailureScope.PROVIDER
    assert result.target_profile_state is None
    assert result.retryable is True
    assert result.failover_eligible is False


def test_transport_exception_takes_precedence_over_synthetic_status_code() -> None:
    error = httpx.ConnectError(
        "network failure secret-token-value",
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )
    error.status_code = 500

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.NETWORK_ERROR
    assert result.status_code == 500
    assert result.target_profile_state is None
    assert "secret-token-value" not in repr(result)


@pytest.mark.parametrize("status_code", [500, 502, 503, 505, 599])
def test_server_errors_are_retryable_without_changing_profile_state(status_code: int) -> None:
    result = classify_openai_subscription_failure(StatusError(status_code), now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.SERVER_ERROR
    assert result.scope is OpenAISubscriptionFailureScope.PROVIDER
    assert result.target_profile_state is None
    assert result.retryable is True
    assert result.failover_eligible is False


def test_unknown_error_and_body_text_do_not_drive_classification() -> None:
    result = classify_openai_subscription_failure(
        StatusError(400, message="quota exhausted invalid token model_not_found"),
        now=NOW,
    )

    assert result.category is OpenAISubscriptionFailureCategory.UNKNOWN_ERROR
    assert result.scope is OpenAISubscriptionFailureScope.UNKNOWN
    assert result.target_profile_state is None
    assert result.retryable is False


@pytest.mark.parametrize(
    "invalid_now",
    [0, -1, True, float("inf"), float("nan"), 253402300799, "1700000000"],
)
def test_classifier_rejects_invalid_clock(invalid_now: Any) -> None:
    with pytest.raises(ValueError, match="now"):
        classify_openai_subscription_failure(StatusError(429), now=invalid_now)


def test_classifier_rejects_non_exception_input() -> None:
    with pytest.raises(TypeError, match="exception"):
        classify_openai_subscription_failure("not-an-exception", now=NOW)  # type: ignore[arg-type]


@pytest.mark.parametrize("api", ["chat_completions", "responses"])
def test_chatgpt_api_paths_preserve_retry_after_for_same_classification(api: str) -> None:
    if api == "chat_completions":
        error = ChatGPTConfig().get_error_class(
            error_message="rate limited",
            status_code=429,
            headers={"Retry-After": "45"},
        )
    else:
        raw_response = httpx.Response(
            429,
            headers={"Content-Type": "application/json", "Retry-After": "45"},
            json={"error": {"message": "rate limited"}},
            request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
        )
        with pytest.raises(OpenAIError) as exc_info:
            ChatGPTResponsesAPIConfig().transform_response_api_response(
                model="gpt-5.4",
                raw_response=raw_response,
                logging_obj=MagicMock(),
            )
        error = exc_info.value

    result = classify_openai_subscription_failure(error, now=NOW)

    assert result.category is OpenAISubscriptionFailureCategory.RATE_LIMIT
    assert result.retry_after_seconds == 45
    assert result.recovery_time_source is OpenAIRecoveryTimeSource.RETRY_AFTER


def test_chatgpt_sse_error_preserves_retry_after_header() -> None:
    raw_response = httpx.Response(
        429,
        headers={"Content-Type": "text/event-stream", "Retry-After": "30"},
        text='data: {"type":"error","error":{"message":"rate limited"}}\n\ndata: [DONE]\n',
        request=httpx.Request("POST", "https://chatgpt.com/backend-api/codex/responses"),
    )

    with pytest.raises(OpenAIError) as exc_info:
        ChatGPTResponsesAPIConfig().transform_response_api_response(
            model="gpt-5.4",
            raw_response=raw_response,
            logging_obj=MagicMock(),
        )

    result = classify_openai_subscription_failure(exc_info.value, now=NOW)
    assert result.retry_after_seconds == 30
