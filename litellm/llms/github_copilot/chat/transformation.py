from typing import Any, Optional, Tuple, cast, List, Dict

import httpx
import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.openai.openai import OpenAIConfig
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderSpecificModelInfo

from ..authenticator import Authenticator
from ..common_utils import GetAPIKeyError


class GithubCopilotConfig(OpenAIConfig, BaseLLMModelInfo):
    GITHUB_COPILOT_API_BASE = "https://api.githubcopilot.com/"
    
    # Class-level storage for model capabilities fetched from GitHub Copilot API
    _model_capabilities: Dict[str, Dict[str, Any]] = {}

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        custom_llm_provider: str = "openai",
    ) -> None:
        super().__init__()
        self.authenticator = Authenticator()
        # Preload model capabilities from GitHub Copilot API
        self._ensure_model_capabilities_loaded()

    def _get_github_copilot_headers(self, extra_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Get standard GitHub Copilot IDE headers that match VS Code implementation.
        
        Args:
            extra_headers: Optional additional headers to merge
            
        Returns:
            Dictionary of headers required for GitHub Copilot API requests
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            # Required GitHub Copilot IDE headers (VS Code values)
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "Editor-Version": "vscode/1.99.3", 
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
        }
        
        if extra_headers:
            headers.update(extra_headers)
            
        return headers

    def _ensure_model_capabilities_loaded(self) -> None:
        """
        Ensure model capabilities are loaded from GitHub Copilot API.
        This is called during initialization to populate the capabilities cache.
        """
        # Only load if we don't have any cached capabilities
        if not GithubCopilotConfig._model_capabilities:
            try:
                # Silently fetch models to populate capabilities cache
                self.get_models()
            except Exception:
                # If we can't fetch models during init, that's okay
                # They'll be fetched when needed
                pass

    def _get_openai_compatible_provider_info(
        self,
        model: str,
        api_base: Optional[str],
        api_key: Optional[str],
        custom_llm_provider: str,
    ) -> Tuple[Optional[str], Optional[str], str]:
        dynamic_api_base = (
            self.authenticator.get_api_base() or self.GITHUB_COPILOT_API_BASE
        )
        try:
            dynamic_api_key = self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise AuthenticationError(
                model=model,
                llm_provider=custom_llm_provider,
                message=str(e),
            )
        return dynamic_api_base, dynamic_api_key, custom_llm_provider

    def _transform_messages(
        self,
        messages,
        model: str,
    ):
        import litellm

        disable_copilot_system_to_assistant = (
            litellm.disable_copilot_system_to_assistant
        )
        if not disable_copilot_system_to_assistant:
            for message in messages:
                if "role" in message and message["role"] == "system":
                    cast(Any, message)["role"] = "assistant"
        return messages

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        # Get API key from authenticator if not provided
        if api_key is None:
            try:
                api_key = self.authenticator.get_api_key()
            except GetAPIKeyError as e:
                raise AuthenticationError(
                    model=model,
                    llm_provider="github_copilot",
                    message=str(e),
                )
        
        # Start with the base OpenAI headers
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        
        # Add required GitHub Copilot IDE headers (same as models endpoint)
        headers.update({
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "Editor-Version": "vscode/1.99.3", 
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
        })

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        headers["X-Initiator"] = initiator
        
        # Add Copilot-Vision-Request header if request contains images and model supports vision
        if self._has_image_content(messages) and self._model_supports_vision(model):
            headers["Copilot-Vision-Request"] = "true"

        return headers

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request for GitHub Copilot API calls.
        This ensures validate_environment is called to set proper headers.
        """
        # Call validate_environment to ensure proper headers are set
        # This is crucial for GitHub Copilot authentication
        validated_headers = self.validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )
        
        # Call parent transform_request to get the base request
        transformed_messages = self._transform_messages(messages=messages, model=model)
        
        # Include extra_headers in the request data so they get passed to OpenAI client
        request_data = {
            "model": model,
            "messages": transformed_messages,
            **optional_params,
        }
        
        # Add extra_headers with our GitHub Copilot headers
        # OpenAI Python client supports extra_headers parameter
        request_data["extra_headers"] = validated_headers
        
        return request_data

    def _determine_initiator(self, messages: List[AllMessageValues]) -> str:
        """
        Determine if request is user or agent initiated based on message roles.
        Returns 'agent' if any message has role 'tool' or 'assistant', otherwise 'user'.
        """
        for message in messages:
            role = message.get("role")
            if role in ["tool", "assistant"]:
                return "agent"
        return "user"
    
    def _has_image_content(self, messages: List[AllMessageValues]) -> bool:
        """
        Check if any message contains image content (image_url).
        
        Args:
            messages: List of chat messages to check
            
        Returns:
            True if any message contains image content, False otherwise
        """
        for message in messages:
            content = message.get("content")
            if content is None:
                continue
                
            # Check if content is a list (multimodal content)
            if isinstance(content, list):
                for content_item in content:
                    if isinstance(content_item, dict) and "image_url" in content_item:
                        return True
            
            # Check if content is a string that might contain base64 images (less common but possible)
            elif isinstance(content, str) and "data:image/" in content:
                return True
                
        return False
    
    def _model_supports_vision(self, model: str) -> bool:
        """
        Check if the specified model supports vision capabilities.
        
        Args:
            model: The model name to check (with or without github_copilot/ prefix)
            
        Returns:
            True if the model supports vision, False otherwise
        """
        # Ensure model has the github_copilot/ prefix for lookup
        if not model.startswith("github_copilot/"):
            model = f"github_copilot/{model}"
        
        # Check cached model capabilities first
        model_info = GithubCopilotConfig._model_capabilities.get(model)
        if model_info:
            capabilities = model_info.get("capabilities", {})
            # Check for explicit vision support in capabilities
            supports = capabilities.get("supports", {})
            if "vision" in supports:
                return bool(supports["vision"])
                
            # Check for vision in direct capabilities field
            if "vision" in capabilities:
                return bool(capabilities["vision"])
        
        # Fallback: Check model name patterns for known vision-capable models
        base_model = model.replace("github_copilot/", "").lower()
        
        # GPT-4 models generally support vision
        if "gpt-4" in base_model and "vision" not in base_model:
            # Most GPT-4 variants support vision except explicit non-vision versions
            return True
        
        # GPT-4o models support vision
        if "gpt-4o" in base_model:
            return True
            
        # Claude-3 models support vision
        if "claude-3" in base_model:
            return True
        
        # Default to False for unknown models to be safe
        return False

    def get_provider_info(self, model: str) -> Optional[ProviderSpecificModelInfo]:
        """
        Get provider-specific model information including capabilities from GitHub Copilot API.
        
        This method uses real capabilities data from the GitHub Copilot API response,
        with minimal fallback inference for missing data.
        """
        # Ensure model has the github_copilot/ prefix for lookup
        lookup_model = model
        if not lookup_model.startswith("github_copilot/"):
            lookup_model = f"github_copilot/{model}"
        
        # Get capabilities from GitHub Copilot API (cached)
        model_info = GithubCopilotConfig._model_capabilities.get(lookup_model, {})
        capabilities = model_info.get("capabilities", {})
        
        # Create provider info using real API data
        provider_info = ProviderSpecificModelInfo()
        
        # Function calling support - GitHub Copilot uses "tool_calls"
        provider_info["supports_function_calling"] = self._get_capability_bool(
            capabilities, ["tool_calls", "function_calling", "tools"], default=True
        )
        
        # System messages - GitHub Copilot generally supports this
        provider_info["supports_system_messages"] = self._get_capability_bool(
            capabilities, ["system_messages"], default=True
        )
        
        # Tool choice support - inferred from tool_calls support
        provider_info["supports_tool_choice"] = self._get_capability_bool(
            capabilities, ["parallel_tool_calls", "tool_choice"], default=True
        )
        
        # Vision support - GitHub Copilot uses "vision" field
        provider_info["supports_vision"] = self._get_capability_bool(
            capabilities, ["vision", "multimodal"], default=False
        )
        
        # Response schema support - GitHub Copilot uses "structured_outputs"
        provider_info["supports_response_schema"] = self._get_capability_bool(
            capabilities, ["structured_outputs", "response_format"], default=True
        )
        
        # Assistant prefill support - typically false for GitHub Copilot
        provider_info["supports_assistant_prefill"] = self._get_capability_bool(
            capabilities, ["assistant_prefill"], default=False
        )
        
        # Prompt caching - check if API exposes this
        provider_info["supports_prompt_caching"] = self._get_capability_bool(
            capabilities, ["prompt_caching", "cache"], default=False
        )
        
        # Audio capabilities
        provider_info["supports_audio_input"] = self._get_capability_bool(
            capabilities, ["audio_input"], default=False
        )
        provider_info["supports_audio_output"] = self._get_capability_bool(
            capabilities, ["audio_output"], default=False
        )
        
        # PDF input support
        provider_info["supports_pdf_input"] = self._get_capability_bool(
            capabilities, ["pdf", "document_input"], default=False
        )
        
        # Native streaming - GitHub Copilot uses "streaming" field
        provider_info["supports_native_streaming"] = self._get_capability_bool(
            capabilities, ["streaming"], default=True
        )
        
        # Web search capabilities
        provider_info["supports_web_search"] = self._get_capability_bool(
            capabilities, ["web_search", "search"], default=False
        )
        
        # Reasoning capabilities (O-series models)
        # GitHub Copilot API doesn't expose reasoning explicitly, use family/model inference
        provider_info["supports_reasoning"] = self._detect_reasoning_support(
            model=lookup_model, capabilities=capabilities
        )
        
        # Computer use - not supported by GitHub Copilot
        provider_info["supports_computer_use"] = self._get_capability_bool(
            capabilities, ["computer_use"], default=False
        )
        
        return provider_info

    def _get_capability_bool(self, capabilities: dict, capability_keys: List[str], default: bool = False) -> bool:
        """
        Get a boolean capability from the GitHub Copilot API response.
        
        Args:
            capabilities: The capabilities dict from GitHub Copilot API
            capability_keys: List of possible keys to check for this capability
            default: Default value if no capability data is found
            
        Returns:
            Boolean indicating if the capability is supported
        """
        # Check direct capabilities dict
        for key in capability_keys:
            if key in capabilities:
                value = capabilities[key]
                if isinstance(value, bool):
                    return value
                elif isinstance(value, str):
                    return value.lower() in ("true", "yes", "1", "enabled")
                elif value is not None:
                    return bool(value)
        
        # Check nested 'supports' dict
        supports = capabilities.get("supports", {})
        if supports:
            for key in capability_keys:
                if key in supports:
                    value = supports[key]
                    if isinstance(value, bool):
                        return value
                    elif isinstance(value, str):
                        return value.lower() in ("true", "yes", "1", "enabled")
                    elif value is not None:
                        return bool(value)
        
        # No capability data found, use default
        return default

    def _detect_reasoning_support(self, model: str, capabilities: dict) -> bool:
        """
        Detect reasoning support using multiple strategies.
        
        Args:
            model: The model name to check
            capabilities: The capabilities dict from GitHub Copilot API
            
        Returns:
            Boolean indicating if the model supports reasoning
        """
        # Strategy 1: Check family field from capabilities
        family = capabilities.get("family", "").lower()
        
        # Known reasoning model families
        reasoning_families = [
            "o1", "o1-mini", "o1-preview", 
            "o2", "o2-mini", 
            "o3", "o3-mini", "o3-pro",
            "o4", "o4-mini",
            # Future O-series models
        ]
        
        if family in reasoning_families:
            return True
            
        # Strategy 2: Check if family starts with known reasoning prefixes
        reasoning_prefixes = ["o1", "o2", "o3", "o4", "o5"]
        for prefix in reasoning_prefixes:
            if family.startswith(prefix):
                return True
        
        # Strategy 3: Model name pattern matching (fallback)
        model_lower = model.lower()
        reasoning_patterns = [
            "o1", "o1-", "o1_", 
            "o2", "o2-", "o2_",
            "o3", "o3-", "o3_",
            "o4", "o4-", "o4_",
            "reasoning", "think", "deep", "research"
        ]
        
        for pattern in reasoning_patterns:
            if pattern in model_lower:
                return True
                
        return False


    def get_models(
        self, api_key: Optional[str] = None, api_base: Optional[str] = None, 
        extra_headers: Optional[Dict[str, str]] = None
    ) -> List[str]:
        """
        Calls GitHub Copilot's `/models` endpoint and returns the list of available models.
        
        GitHub Copilot uses a `/models` endpoint (not `/v1/models`) that returns
        an array of model objects directly, each containing id, name, and capabilities.
        
        Args:
            api_key: Optional API key override
            api_base: Optional API base URL override
            extra_headers: Optional additional headers (can override default editor headers)
        """
        # Get the API base and key using GitHub Copilot's authentication system
        try:
            dynamic_api_base = (
                api_base or 
                self.authenticator.get_api_base() or 
                self.GITHUB_COPILOT_API_BASE
            )
            dynamic_api_key = api_key or self.authenticator.get_api_key()
        except GetAPIKeyError as e:
            raise ValueError(
                f"GitHub Copilot API key not available: {str(e)}. "
                "Please ensure GitHub Copilot authentication is properly configured."
            )
        
        # Ensure the API base ends with a slash for proper URL construction
        if not dynamic_api_base.endswith("/"):
            dynamic_api_base += "/"
        
        # Make request to the models endpoint (GitHub Copilot uses /models, not /v1/models)
        try:
            # Get standard GitHub Copilot headers
            headers = self._get_github_copilot_headers(extra_headers)
            headers["Authorization"] = f"Bearer {dynamic_api_key}"
            
            response = litellm.module_level_client.get(
                url=f"{dynamic_api_base.rstrip('/')}/models",
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"Failed to fetch models from GitHub Copilot. Status code: {e.response.status_code}, "
                f"Response: {e.response.text}"
            )
        except Exception as e:
            raise Exception(f"Error connecting to GitHub Copilot models endpoint: {str(e)}")
        
        # Parse the response - GitHub Copilot returns an array of models directly
        try:
            models_data = response.json()
            
            # GitHub Copilot returns models as an array directly, not wrapped in a "data" key
            if isinstance(models_data, list):
                models = models_data
            else:
                # Fallback: check if it's wrapped in "data" key (in case format changes)
                models = models_data.get("data", models_data)
                if not isinstance(models, list):
                    models = []
            
            # Extract model IDs and capabilities, prefix them with github_copilot/ for LiteLLM naming convention
            litellm_model_names = []
            for model in models:
                if isinstance(model, dict):
                    # GitHub Copilot models have "id" field for the model identifier
                    model_id = model.get("id") or model.get("name")
                    if model_id:
                        # Prefix with github_copilot/ to follow LiteLLM's provider naming pattern
                        litellm_model_name = f"github_copilot/{model_id}"
                        litellm_model_names.append(litellm_model_name)
                        
                        # Store comprehensive model info for later use in get_provider_info
                        capabilities = model.get("capabilities", {})
                        model_info = {
                            "id": model_id,
                            "name": model.get("name", model_id),
                            "capabilities": capabilities,
                            "preview": model.get("preview", False),
                            "is_fallback": model.get("is_fallback", False),
                            "description": model.get("description", ""),
                            "version": model.get("version", ""),
                            "family": model.get("family", ""),
                            "limits": model.get("limits", {}),
                            "tags": model.get("tags", []),
                            "publisher": model.get("publisher", ""),
                            "created": model.get("created", ""),
                        }
                        
                        # Store in class-level cache
                        GithubCopilotConfig._model_capabilities[litellm_model_name] = model_info
            
            return litellm_model_names
        except (KeyError, ValueError, TypeError) as e:
            raise Exception(f"Failed to parse models response from GitHub Copilot: {str(e)}")
        
    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get GitHub Copilot API key. This is implemented to maintain compatibility
        with BaseLLMModelInfo interface, but GitHub Copilot uses its own authentication system.
        """
        if api_key:
            return api_key
        
        # For GitHub Copilot, we rely on the authenticator system rather than simple env vars
        try:
            authenticator = Authenticator()
            return authenticator.get_api_key()
        except GetAPIKeyError:
            return None
    
    @staticmethod
    def get_api_base(api_base: Optional[str] = None) -> Optional[str]:
        """
        Get GitHub Copilot API base URL.
        """
        if api_base:
            return api_base
        
        try:
            authenticator = Authenticator()
            return authenticator.get_api_base() or "https://api.githubcopilot.com/"
        except Exception:
            return "https://api.githubcopilot.com/"
    
    @staticmethod
    def get_base_model(model: Optional[str] = None) -> Optional[str]:
        """
        Get the base model name by removing the github_copilot/ prefix.
        """
        if model is not None:
            return model.replace("github_copilot/", "")
        return None