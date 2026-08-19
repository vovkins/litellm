"""OpenAI OAuth subscription affinity lifecycle hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, cast

from litellm._logging import verbose_router_logger
from litellm.exceptions import ServiceUnavailableError
from litellm.integrations.custom_logger import CustomLogger, Span
from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
)
from litellm.types.llms.openai import AllMessageValues


class OpenAISubscriptionAffinityCheck(CustomLogger):
    """Route OAuth models by subscription and keep successful bindings alive.

    Profile health and failover are implemented separately. This callback uses
    only currently healthy deployments and intentionally extends affinity only
    from final success events.
    """

    def __init__(
        self,
        store: OpenAISubscriptionAffinityStore,
        ttl_seconds: int = OpenAISubscriptionAffinityStore.DEFAULT_TTL_SECONDS,
    ) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list,
        messages: list[AllMessageValues] | None,
        request_kwargs: dict | None = None,
        parent_otel_span: Span | None = None,
    ) -> list[dict]:
        deployments: Final = cast(list[dict], healthy_deployments)
        try:
            profile_ids, has_unmarked_deployments = self._get_available_profile_ids(deployments)
        except ValueError:
            verbose_router_logger.error("OpenAI subscription affinity found an invalid profile marker")
            raise self._service_unavailable(model) from None
        if not profile_ids:
            return deployments
        if has_unmarked_deployments:
            verbose_router_logger.error("OpenAI subscription affinity found a mixed OAuth/non-OAuth model group")
            raise self._service_unavailable(model)

        user_api_key_hash: Final = self._get_request_user_api_key_hash(request_kwargs or {})
        if user_api_key_hash is None:
            verbose_router_logger.error("OpenAI subscription affinity requires an authenticated virtual-key hash")
            raise self._service_unavailable(model)

        try:
            selected_profile: Final = await self.store.get_or_assign_profile(
                user_api_key_hash=user_api_key_hash,
                available_profile_ids=profile_ids,
                ttl_seconds=self.ttl_seconds,
            )
        except (OpenAISubscriptionAffinityStoreError, ValueError):
            verbose_router_logger.error("OpenAI subscription affinity could not read or assign shared routing state")
            raise self._service_unavailable(model) from None

        selected_deployments: Final = [
            deployment for deployment in deployments if self._get_deployment_profile_id(deployment) == selected_profile
        ]
        if not selected_deployments:
            # A binding can point at a profile that is no longer in the current
            # healthy set. State-aware failover will handle this in a later step;
            # until then, never bypass affinity with a random subscription.
            verbose_router_logger.error("OpenAI subscription affinity selected a profile without a healthy deployment")
            raise self._service_unavailable(model)
        return selected_deployments

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        user_api_key_hash: Final = self._get_user_api_key_hash(kwargs)
        profile_id: Final = self._get_selected_profile_id(kwargs)
        if user_api_key_hash is None or profile_id is None:
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

    @staticmethod
    def _get_deployment_profile_id(deployment: Mapping[str, Any]) -> str | None:
        model_info: Final = deployment.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        profile_id: Final = model_info.get("openai_oauth_profile")
        return profile_id if isinstance(profile_id, str) else None

    @staticmethod
    def _service_unavailable(model: str) -> ServiceUnavailableError:
        return ServiceUnavailableError(
            message="The requested model is temporarily unavailable. Retry later.",
            llm_provider="",
            model=model,
        )

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
