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
        # Get base headers from parent
        validated_headers = super().validate_environment(
            headers, model, messages, optional_params, litellm_params, api_key, api_base
        )

        # Add X-Initiator header based on message roles
        initiator = self._determine_initiator(messages)
        validated_headers["X-Initiator"] = initiator
        
        # Add Copilot-Vision-Request header if request contains images and model supports vision
        if self._has_image_content(messages) and self._model_supports_vision(model):
            validated_headers["Copilot-Vision-Request"] = "true"

        return validated_headers

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
        Get provider-specific model information including capabilities from GitHub Copilot.
        
        This method maps GitHub Copilot's capability information to LiteLLM's
        ProviderSpecificModelInfo format.
        """
        # Ensure model has the github_copilot/ prefix
        if not model.startswith("github_copilot/"):
            model = f"github_copilot/{model}"
        
        # Get capabilities from stored model info
        model_info = GithubCopilotConfig._model_capabilities.get(model)
        if not model_info:
            # If no cached info, return default capabilities for GitHub Copilot models
            return ProviderSpecificModelInfo(
                supports_function_calling=True,
                supports_system_messages=True,
                supports_tool_choice=True,
                supports_vision=True,  # Most GitHub Copilot models support vision
                supports_response_schema=True,
            )
        
        capabilities = model_info.get("capabilities", {})
        
        # Map GitHub Copilot capabilities to LiteLLM ProviderSpecificModelInfo
        provider_info = ProviderSpecificModelInfo()
        
        # Parse capability type (chat, embedding, etc.)
        capability_type = capabilities.get("type", "").lower()
        
        # GitHub Copilot chat models typically support these features
        if capability_type == "chat":
            provider_info["supports_function_calling"] = capabilities.get("function_calling", True)
            provider_info["supports_system_messages"] = True
            provider_info["supports_tool_choice"] = True
            provider_info["supports_vision"] = capabilities.get("vision", True)
            provider_info["supports_response_schema"] = capabilities.get("response_schema", True)
            
        # Parse additional capabilities if available
        if "limits" in capabilities:
            # Could add context window information if GitHub Copilot provides it in future
            pass
        
        # Parse family information for additional capabilities
        family = capabilities.get("family", "").lower()
        if "gpt-4" in family or "gpt-4" in model.lower():
            provider_info["supports_vision"] = True
            provider_info["supports_function_calling"] = True
            
        return provider_info

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
            # GitHub Copilot requires specific editor headers for IDE authentication
            headers = {
                "Authorization": f"Bearer {dynamic_api_key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                # Required GitHub Copilot IDE headers (default to VS Code values)
                "User-Agent": "GitHubCopilotChat/0.26.7",
                "Editor-Version": "vscode/1.99.3", 
                "Editor-Plugin-Version": "copilot-chat/0.26.7",
                "Copilot-Integration-Id": "vscode-chat",
            }
            
            # Allow overriding headers (useful for different editors or testing)
            if extra_headers:
                headers.update(extra_headers)
            
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
                        
                        # Store model capabilities for later use in get_provider_info
                        capabilities = model.get("capabilities", {})
                        model_info = {
                            "id": model_id,
                            "name": model.get("name", model_id),
                            "capabilities": capabilities,
                            "preview": model.get("preview", False),
                            "is_fallback": model.get("is_fallback", False),
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