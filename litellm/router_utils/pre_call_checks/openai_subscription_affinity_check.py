"""OpenAI OAuth subscription affinity lifecycle hooks."""

from __future__ import annotations

import re
import secrets
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final, cast

import httpx

from litellm._logging import verbose_router_logger
from litellm.exceptions import RateLimitError, ServiceUnavailableError
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.router_utils.openai_subscription_affinity import (
    OpenAIProfileFailureUpdateStatus,
    OpenAIProfileHalfOpenLease,
    OpenAIProfileState,
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
    mark_openai_subscription_terminal_routing_error,
)
from litellm.router_utils.openai_subscription_failure_classifier import (
    classify_openai_subscription_failure,
)
from litellm.types.llms.openai import AllMessageValues

_PROBE_HANDLE_PATTERN: Final = re.compile(r"^[0-9a-f]{32}$")
_PROBE_HANDLE_MODEL_INFO_KEY: Final = "_openai_subscription_probe_handle"


@dataclass(frozen=True, slots=True)
class _HalfOpenProbeContext:
    lease: OpenAIProfileHalfOpenLease = field(repr=False)
    user_api_key_hash: str = field(repr=False)
    local_expires_at: float = field(repr=False)


class OpenAISubscriptionAffinityCheck(CustomLogger):
    """Route OAuth models by subscription and maintain profile availability.

    Successful responses extend the current binding. Profile-scoped provider
    failures atomically update shared availability and release only the binding
    that still points to the failed profile, allowing the Router's next attempt
    to select another available subscription.
    """

    def __init__(
        self,
        store: OpenAISubscriptionAffinityStore,
        ttl_seconds: int = OpenAISubscriptionAffinityStore.DEFAULT_TTL_SECONDS,
        half_open_lease_seconds: int = OpenAISubscriptionAffinityStore.DEFAULT_HALF_OPEN_LEASE_SECONDS,
        router: Any | None = None,
    ) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds
        self.router = router
        if (
            not isinstance(half_open_lease_seconds, int)
            or isinstance(half_open_lease_seconds, bool)
            or half_open_lease_seconds <= 0
            or half_open_lease_seconds > OpenAISubscriptionAffinityStore.MAX_HALF_OPEN_LEASE_SECONDS
        ):
            raise ValueError("half_open_lease_seconds must be a positive integer no greater than 3600")
        self.half_open_lease_seconds = half_open_lease_seconds
        self._probe_contexts: dict[str, _HalfOpenProbeContext] = {}
        self._probe_contexts_lock = threading.Lock()

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        deployments: Final = cast(list[dict], healthy_deployments)
        request_context: Final = request_kwargs or {}
        try:
            profile_ids, has_unmarked_deployments = self._get_available_profile_ids(deployments)
            configured_deployments: Final = self._get_configured_deployments(
                model=model,
                request_kwargs=request_context,
            )
            configured_profile_ids, configured_has_unmarked = self._get_available_profile_ids(configured_deployments)
            if configured_profile_ids:
                if configured_has_unmarked:
                    has_unmarked_deployments = True
                if not profile_ids:
                    profile_ids = configured_profile_ids
            else:
                configured_profile_ids = profile_ids
        except ValueError:
            verbose_router_logger.error("OpenAI subscription affinity found an invalid profile marker")
            raise self._service_unavailable(model) from None
        if not profile_ids:
            return deployments
        if has_unmarked_deployments:
            verbose_router_logger.error("OpenAI subscription affinity found a mixed OAuth/non-OAuth model group")
            raise self._service_unavailable(model)

        user_api_key_hash: Final = self._get_request_user_api_key_hash(request_context)
        if user_api_key_hash is None:
            verbose_router_logger.error("OpenAI subscription affinity requires an authenticated virtual-key hash")
            raise self._service_unavailable(model)

        try:
            selection: Final = await self.store.select_available_profile(
                user_api_key_hash=user_api_key_hash,
                available_profile_ids=profile_ids,
                ttl_seconds=self.ttl_seconds,
                allow_recovery_probe=True,
                half_open_lease_seconds=self.half_open_lease_seconds,
            )
        except (OpenAISubscriptionAffinityStoreError, ValueError):
            verbose_router_logger.error("OpenAI subscription affinity could not read or assign shared routing state")
            raise self._service_unavailable(model) from None
        selected_profile: Final = selection.profile_id
        if selected_profile is None:
            available_profiles = 0
            cooldown_profiles = selection.cooldown_profiles
            half_open_profiles = selection.half_open_profiles
            disabled_profiles = selection.disabled_profiles
            retry_after_seconds = selection.retry_after_seconds
            if set(configured_profile_ids) != set(profile_ids):
                try:
                    availability: Final = await self.store.inspect_profile_availability(configured_profile_ids)
                except (OpenAISubscriptionAffinityStoreError, ValueError):
                    verbose_router_logger.error(
                        "OpenAI subscription affinity could not inspect shared pool availability"
                    )
                    raise self._service_unavailable(model) from None
                available_profiles = availability.available_profiles
                cooldown_profiles = availability.cooldown_profiles
                half_open_profiles = availability.half_open_profiles
                disabled_profiles = availability.disabled_profiles
                retry_after_seconds = availability.retry_after_seconds
            verbose_router_logger.error(
                "OpenAI subscription affinity found no profile eligible for normal traffic "
                "(available=%s, cooldown=%s, half_open=%s, disabled=%s)",
                available_profiles,
                cooldown_profiles,
                half_open_profiles,
                disabled_profiles,
            )
            if (
                available_profiles == 0
                and cooldown_profiles + half_open_profiles > 0
                and retry_after_seconds is not None
            ):
                raise self._rate_limited(
                    model=model,
                    retry_after_seconds=retry_after_seconds,
                )
            raise self._service_unavailable(model)

        selected_deployments: Final = [
            deployment for deployment in deployments if self._get_deployment_profile_id(deployment) == selected_profile
        ]
        if not selected_deployments:
            # The store only returns a profile from this candidate set. Keep a
            # defensive fail-closed guard in case that contract is ever broken.
            verbose_router_logger.error("OpenAI subscription affinity selected a profile without a healthy deployment")
            raise self._service_unavailable(model)

        lease: Final = selection.half_open_lease
        if lease is None:
            return [self._copy_deployment_with_probe_handle(deployment, None) for deployment in selected_deployments]
        probe_handle: Final = self._remember_probe_context(
            user_api_key_hash=user_api_key_hash,
            lease=lease,
        )
        return [
            self._copy_deployment_with_probe_handle(deployment, probe_handle) for deployment in selected_deployments
        ]

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        user_api_key_hash: Final = self._get_user_api_key_hash(kwargs)
        profile_id: Final = self._get_selected_profile_id(kwargs)
        if user_api_key_hash is None or profile_id is None:
            return

        probe_handle: Final = self._get_selected_probe_handle(kwargs)
        if probe_handle is not None:
            context: Final = self._take_probe_context(probe_handle)
            if context is None:
                verbose_router_logger.error("OpenAI subscription recovery success has no local lease context")
                return
            if context.user_api_key_hash != user_api_key_hash or context.lease.profile_id != profile_id:
                verbose_router_logger.error(
                    "OpenAI subscription recovery success does not match trusted routing context"
                )
                return
            try:
                completion: Final = await self.store.complete_profile_probe_for_binding(
                    user_api_key_hash=user_api_key_hash,
                    profile_id=profile_id,
                    lease_token=context.lease.token,
                    target_state=OpenAIProfileState.AVAILABLE,
                    ttl_seconds=self.ttl_seconds,
                )
            except (OpenAISubscriptionAffinityStoreError, ValueError):
                verbose_router_logger.error(
                    "OpenAI subscription affinity could not complete a successful recovery probe"
                )
                return
            if not completion.applied:
                verbose_router_logger.debug("OpenAI subscription affinity ignored a stale successful recovery probe")
            return

        try:
            await self.store.refresh_profile_if_current(
                user_api_key_hash=user_api_key_hash,
                profile_id=profile_id,
                ttl_seconds=self.ttl_seconds,
            )
        except (OpenAISubscriptionAffinityStoreError, ValueError):
            # The provider response has already succeeded; affinity maintenance
            # must not turn that response into a client-visible failure.
            verbose_router_logger.error("OpenAI subscription affinity TTL refresh failed after a successful request")

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        if not isinstance(kwargs, Mapping):
            return
        error: Final = kwargs.get("exception")
        if not isinstance(error, BaseException):
            return

        classification: Final = classify_openai_subscription_failure(error)
        target_state: Final = classification.target_profile_state

        user_api_key_hash: Final = self._get_user_api_key_hash(kwargs)
        profile_id: Final = self._get_selected_profile_id(kwargs)
        if user_api_key_hash is None or profile_id is None:
            verbose_router_logger.error(
                "OpenAI subscription affinity could not attribute a profile-scoped failure to trusted routing metadata"
            )
            return

        probe_handle: Final = self._get_selected_probe_handle(kwargs)
        if probe_handle is not None:
            context: Final = self._take_probe_context(probe_handle)
            if context is None:
                verbose_router_logger.error("OpenAI subscription recovery failure has no local lease context")
                return
            if context.user_api_key_hash != user_api_key_hash or context.lease.profile_id != profile_id:
                verbose_router_logger.error(
                    "OpenAI subscription recovery failure does not match trusted routing context"
                )
                return
            if not classification.failover_eligible or target_state is None:
                verbose_router_logger.debug(
                    "OpenAI subscription recovery probe remains half-open until its lease expires"
                )
                return
            try:
                completion: Final = await self.store.complete_profile_probe_for_binding(
                    user_api_key_hash=user_api_key_hash,
                    profile_id=profile_id,
                    lease_token=context.lease.token,
                    target_state=target_state,
                    reason_code=classification.reason_code,
                    reset_at=classification.reset_at,
                    ttl_seconds=self.ttl_seconds,
                )
            except (OpenAISubscriptionAffinityStoreError, ValueError):
                verbose_router_logger.error("OpenAI subscription affinity could not complete a failed recovery probe")
                return
            if not completion.applied:
                verbose_router_logger.debug("OpenAI subscription affinity ignored a stale failed recovery probe")
            return

        if not classification.failover_eligible or target_state is None:
            return

        try:
            update: Final = await self.store.fail_profile_if_current_binding(
                user_api_key_hash=user_api_key_hash,
                profile_id=profile_id,
                target_state=target_state,
                reason_code=classification.reason_code,
                reset_at=classification.reset_at,
            )
        except (OpenAISubscriptionAffinityStoreError, ValueError):
            verbose_router_logger.error("OpenAI subscription affinity could not persist a profile-scoped failure")
            return

        if update.status is OpenAIProfileFailureUpdateStatus.STALE:
            verbose_router_logger.debug("OpenAI subscription affinity ignored a stale profile-scoped failure")

    def _remember_probe_context(
        self,
        *,
        user_api_key_hash: str,
        lease: OpenAIProfileHalfOpenLease,
    ) -> str:
        now: Final = time.monotonic()
        local_expires_at: Final = now + self.half_open_lease_seconds + 1
        with self._probe_contexts_lock:
            self._prune_probe_contexts(now)
            probe_handle = secrets.token_hex(16)
            while probe_handle in self._probe_contexts:
                probe_handle = secrets.token_hex(16)
            self._probe_contexts[probe_handle] = _HalfOpenProbeContext(
                lease=lease,
                user_api_key_hash=user_api_key_hash,
                local_expires_at=local_expires_at,
            )
        return probe_handle

    def _take_probe_context(self, probe_handle: str) -> _HalfOpenProbeContext | None:
        now: Final = time.monotonic()
        with self._probe_contexts_lock:
            self._prune_probe_contexts(now)
            return self._probe_contexts.pop(probe_handle, None)

    def _prune_probe_contexts(self, now: float) -> None:
        expired_handles: Final = [
            handle for handle, context in self._probe_contexts.items() if context.local_expires_at <= now
        ]
        for handle in expired_handles:
            self._probe_contexts.pop(handle, None)

    @staticmethod
    def _copy_deployment_with_probe_handle(
        deployment: Mapping[str, Any],
        probe_handle: str | None,
    ) -> dict:
        copied_deployment: Final = dict(deployment)
        model_info: Final = dict(deployment.get("model_info") or {})
        model_info.pop(_PROBE_HANDLE_MODEL_INFO_KEY, None)
        if probe_handle is not None:
            model_info[_PROBE_HANDLE_MODEL_INFO_KEY] = probe_handle
        copied_deployment["model_info"] = model_info
        return copied_deployment

    @staticmethod
    def _get_user_api_key_hash(kwargs: Mapping[str, Any]) -> str | None:
        standard_logging_object: Final = kwargs.get("standard_logging_object")
        if not isinstance(standard_logging_object, Mapping):
            return None
        metadata: Final = standard_logging_object.get("metadata")
        if not isinstance(metadata, Mapping):
            return None
        user_api_key_hash: Final = metadata.get("user_api_key_hash")
        return user_api_key_hash if isinstance(user_api_key_hash, str) else None

    @staticmethod
    def _get_request_user_api_key_hash(request_kwargs: Mapping[str, Any]) -> str | None:
        for metadata_key in ("metadata", "litellm_metadata"):
            metadata: Final = request_kwargs.get(metadata_key)
            if not isinstance(metadata, Mapping):
                continue
            user_api_key_hash: Final = metadata.get("user_api_key_hash")
            if isinstance(user_api_key_hash, str):
                return user_api_key_hash
        return None

    @classmethod
    def _get_available_profile_ids(cls, deployments: list[dict]) -> tuple[list[str], bool]:
        profile_ids: list[str] = []
        has_unmarked_deployments = False
        for deployment in deployments:
            model_info: Final = deployment.get("model_info")
            if not isinstance(model_info, Mapping) or "openai_oauth_profile" not in model_info:
                has_unmarked_deployments = True
                continue
            profile_id: Final = model_info.get("openai_oauth_profile")
            if not isinstance(profile_id, str):
                raise ValueError("invalid OpenAI OAuth profile marker")
            profile_ids.append(profile_id)
        return profile_ids, has_unmarked_deployments

    def _get_configured_deployments(
        self,
        *,
        model: str,
        request_kwargs: Mapping[str, Any],
    ) -> list[dict]:
        if self.router is None:
            return []
        team_id: str | None = None
        for metadata_key in ("metadata", "litellm_metadata"):
            metadata: Final = request_kwargs.get(metadata_key)
            if isinstance(metadata, Mapping):
                candidate_team_id: Final = metadata.get("user_api_key_team_id")
                if isinstance(candidate_team_id, str):
                    team_id = candidate_team_id
                    break
        try:
            configured: Final = self.router.get_model_list(model_name=model, team_id=team_id)
        except Exception as exc:
            raise ValueError("could not inspect configured OpenAI subscription deployments") from exc
        if configured is None:
            return []
        if not isinstance(configured, list) or any(not isinstance(deployment, dict) for deployment in configured):
            raise ValueError("configured OpenAI subscription deployments have an invalid shape")
        return cast(list[dict], configured)

    @staticmethod
    def _get_deployment_profile_id(deployment: Mapping[str, Any]) -> str | None:
        model_info: Final = deployment.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        profile_id: Final = model_info.get("openai_oauth_profile")
        return profile_id if isinstance(profile_id, str) else None

    @staticmethod
    def _service_unavailable(model: str) -> ServiceUnavailableError:
        error: Final = ServiceUnavailableError(
            message="The requested model is temporarily unavailable. Retry later.",
            llm_provider="",
            model=model,
            num_retries=0,
        )
        mark_openai_subscription_terminal_routing_error(error)
        return error

    @staticmethod
    def _rate_limited(model: str, retry_after_seconds: int) -> RateLimitError:
        retry_after: Final = max(
            1,
            min(
                retry_after_seconds,
                OpenAISubscriptionAffinityStore.MAX_RECOVERY_RETRY_AFTER_SECONDS,
            ),
        )
        headers: Final = {"retry-after": str(retry_after)}
        error: Final = RateLimitError(
            message="The requested model is temporarily rate-limited. Retry later.",
            llm_provider="",
            model=model,
            num_retries=0,
            headers=headers,
            response=httpx.Response(
                status_code=429,
                headers=headers,
                request=httpx.Request("POST", "https://litellm.invalid/"),
            ),
        )
        mark_openai_subscription_terminal_routing_error(error)
        return error

    @staticmethod
    def _get_selected_profile_id(kwargs: Mapping[str, Any]) -> str | None:
        litellm_params: Final = kwargs.get("litellm_params")
        if not isinstance(litellm_params, Mapping):
            return None
        model_info: Final = litellm_params.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        profile_id: Final = model_info.get("openai_oauth_profile")
        return profile_id if isinstance(profile_id, str) else None

    @staticmethod
    def _get_selected_probe_handle(kwargs: Mapping[str, Any]) -> str | None:
        litellm_params: Final = kwargs.get("litellm_params")
        if not isinstance(litellm_params, Mapping):
            return None
        model_info: Final = litellm_params.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        probe_handle: Final = model_info.get(_PROBE_HANDLE_MODEL_INFO_KEY)
        if not isinstance(probe_handle, str) or _PROBE_HANDLE_PATTERN.fullmatch(probe_handle) is None:
            return None
        return probe_handle
