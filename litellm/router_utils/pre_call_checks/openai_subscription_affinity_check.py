"""OpenAI OAuth subscription affinity lifecycle hooks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from litellm._logging import verbose_router_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.router_utils.openai_subscription_affinity import (
    OpenAISubscriptionAffinityStore,
    OpenAISubscriptionAffinityStoreError,
)


class OpenAISubscriptionAffinityCheck(CustomLogger):
    """Keep a successful virtual-key/profile binding alive.

    Routing and failover will be added separately. This callback intentionally
    handles only final success events: provider, OAuth, rate-limit, network, and
    guardrail failures must not extend affinity.
    """

    def __init__(
        self,
        store: OpenAISubscriptionAffinityStore,
        ttl_seconds: int = OpenAISubscriptionAffinityStore.DEFAULT_TTL_SECONDS,
    ) -> None:
        self.store = store
        self.ttl_seconds = ttl_seconds

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
    def _get_selected_profile_id(kwargs: Mapping[str, Any]) -> str | None:
        litellm_params: Final = kwargs.get("litellm_params")
        if not isinstance(litellm_params, Mapping):
            return None
        model_info: Final = litellm_params.get("model_info")
        if not isinstance(model_info, Mapping):
            return None
        profile_id: Final = model_info.get("openai_oauth_profile")
        return profile_id if isinstance(profile_id, str) else None
