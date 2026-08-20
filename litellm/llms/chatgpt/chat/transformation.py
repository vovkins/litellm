from collections.abc import Mapping
from typing import Any, Final

from litellm.exceptions import AuthenticationError
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues

from ..authenticator import (
    Authenticator,
    ChatGPTAuthFileParams,
    get_cached_authenticator,
    get_chatgpt_auth_file,
)
from ..common_utils import (
    GetAccessTokenError,
    ensure_chatgpt_session_id,
    get_chatgpt_default_headers,
)
from .streaming_utils import ChatGPTToolCallNormalizer


class ChatGPTConfig(OpenAIConfig):
    def __init__(
        self,
        api_key: str | None = None,
        api_base: str | None = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        # Provider calls run inside a server process. Missing credentials must
        # fail immediately instead of starting a blocking device-code flow.
        self.authenticator = Authenticator(allow_interactive_login=False)

    def _resolve_authenticator(
        self, litellm_params: Mapping[str, object] | ChatGPTAuthFileParams | None
    ) -> Authenticator:
        auth_file: Final = get_chatgpt_auth_file(litellm_params)
        if auth_file:
            return get_cached_authenticator(auth_file)
        return self.authenticator

    @staticmethod
    def _get_access_token_or_raise(authenticator: Authenticator, model: str, llm_provider: str) -> str:
        try:
            return authenticator.get_access_token()
        except GetAccessTokenError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=llm_provider,
                message=str(e),
            )

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: str | None,
        api_key: str | None,
        custom_llm_provider: str,
        litellm_params: Mapping[str, object] | ChatGPTAuthFileParams | None = None,
    ) -> tuple[str | None, str | None, str]:
        # Provider discovery is also used by Router metadata and capability
        # checks. Keep it credential-free; the selected deployment resolves its
        # token in validate_environment immediately before the upstream call.
        dynamic_api_base: Final = api_base or self.authenticator.get_api_base()
        return dynamic_api_base, api_key, custom_llm_provider

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        auth_file: Final = get_chatgpt_auth_file(litellm_params)
        authenticator: Final = get_cached_authenticator(auth_file) if auth_file else self.authenticator
        resolved_api_key: Final = (
            self._get_access_token_or_raise(authenticator, model, "chatgpt")
            if auth_file or api_key is None
            else api_key
        )

        validated_headers: Final = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, resolved_api_key, api_base
        )

        account_id: Final = authenticator.get_account_id()
        session_id: Final = ensure_chatgpt_session_id(litellm_params)
        default_headers: Final = get_chatgpt_default_headers(resolved_api_key or "", account_id, session_id)
        return {**default_headers, **validated_headers}

    def post_stream_processing(self, stream: Any) -> Any:
        return ChatGPTToolCallNormalizer(stream)

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        optional_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)
        optional_params.setdefault("stream", False)
        return optional_params
