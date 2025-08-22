import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

sys.path.insert(0, os.path.abspath("../.."))

import httpx
import pytest
from respx import MockRouter

import litellm

# Import at the top to make the patch work correctly
import litellm.llms.github_copilot.chat.transformation
from litellm import Choices, Message, ModelResponse, Usage, acompletion, completion
from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.authenticator import Authenticator
from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig
from litellm.llms.github_copilot.common_utils import (
    APIKeyExpiredError,
    GetAccessTokenError,
    GetAPIKeyError,
    GetDeviceCodeError,
    RefreshAPIKeyError,
)


def test_github_copilot_config_get_openai_compatible_provider_info():
    """Test the GitHub Copilot configuration provider info retrieval."""

    config = GithubCopilotConfig()

    # Mock the authenticator to avoid actual API calls
    mock_api_key = "gh.test-key-123456789"
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = mock_api_key
    # Test with dynamic endpoint
    config.authenticator.get_api_base.return_value = "https://api.enterprise.githubcopilot.com"

    # Test with default values
    model = "github_copilot/gpt-4"
    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model=model,
        api_base=None,
        api_key=None,
        custom_llm_provider="github_copilot",
    )

    assert api_base == "https://api.enterprise.githubcopilot.com"
    assert dynamic_api_key == mock_api_key
    assert custom_llm_provider == "github_copilot"

    # Test fallback to default if no dynamic endpoint
    config.authenticator.get_api_base.return_value = None
    (
        api_base,
        dynamic_api_key,
        custom_llm_provider,
    ) = config._get_openai_compatible_provider_info(
        model=model,
        api_base=None,
        api_key=None,
        custom_llm_provider="github_copilot",
    )
    assert api_base == "https://api.githubcopilot.com/"

    # Test with authentication failure
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        message="Failed to get API key",
        status_code=401,
    )

    with pytest.raises(AuthenticationError) as excinfo:
        config._get_openai_compatible_provider_info(
            model=model,
            api_base=None,
            api_key=None,
            custom_llm_provider="github_copilot",
        )

    assert "Failed to get API key" in str(excinfo.value)


@patch("litellm.llms.github_copilot.authenticator.Authenticator.get_api_key")
@patch("litellm.llms.openai.openai.OpenAIChatCompletion.completion")
def test_completion_github_copilot_mock_response(mock_completion, mock_get_api_key):
    """Test the completion function with GitHub Copilot provider."""

    # Mock the API key return value
    mock_api_key = "gh.test-key-123456789"
    mock_get_api_key.return_value = mock_api_key

    # Mock completion response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello, I'm GitHub Copilot!"
    mock_completion.return_value = mock_response

    # Test non-streaming completion
    messages = [
        {"role": "system", "content": "You're GitHub Copilot, an AI assistant."},
        {"role": "user", "content": "Hello, who are you?"},
    ]

    # Create a properly formatted headers dictionary
    headers = {
        "editor-version": "Neovim/0.9.0",
        "Copilot-Integration-Id": "vscode-chat",
    }

    response = completion(
        model="github_copilot/gpt-4",
        messages=messages,
        extra_headers=headers,
    )

    assert response is not None

    # Verify the get_api_key call was made (can be called multiple times)
    assert mock_get_api_key.call_count >= 1

    # Verify the completion call was made with the expected params
    mock_completion.assert_called_once()
    args, kwargs = mock_completion.call_args

    # Check that the proper authorization header is set
    assert "headers" in kwargs
    # Check that the model name is correctly formatted
    assert (
        kwargs.get("model") == "gpt-4"
    )  # Model name should be without provider prefix
    assert kwargs.get("messages") == messages


def test_transform_messages_disable_copilot_system_to_assistant(monkeypatch):
    """Test that system messages are converted to assistant unless disable_copilot_system_to_assistant is True."""
    import litellm
    from litellm.llms.github_copilot.chat.transformation import GithubCopilotConfig

    # Save original value
    original_flag = litellm.disable_copilot_system_to_assistant
    try:
        # Case 1: Flag is False (default, conversion happens)
        litellm.disable_copilot_system_to_assistant = False
        config = GithubCopilotConfig()
        messages = [
            {"role": "system", "content": "System message."},
            {"role": "user", "content": "User message."},
        ]
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"

        # Case 2: Flag is True (conversion does not happen)
        litellm.disable_copilot_system_to_assistant = True
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "system"
        assert out[1]["role"] == "user"

        # Case 3: Flag is False again (conversion happens)
        litellm.disable_copilot_system_to_assistant = False
        out = config._transform_messages([m.copy() for m in messages], model="github_copilot/gpt-4")
        assert out[0]["role"] == "assistant"
        assert out[1]["role"] == "user"
    finally:
        # Restore original value
        litellm.disable_copilot_system_to_assistant = original_flag


def test_x_initiator_header_user_request():
    """Test that user-only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello!"},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_agent_request_with_assistant():
    """Test that messages with assistant role result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "assistant", "content": "I can help you."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4", 
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_agent_request_with_tool():
    """Test that messages with tool role result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "tool", "content": "Tool response.", "tool_call_id": "123"},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4", 
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_mixed_messages_with_agent_roles():
    """Test that mixed messages with agent roles (assistant/tool) result in X-Initiator: agent header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator  
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Previous response."},
        {"role": "user", "content": "Follow up question."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages, 
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "agent"


def test_x_initiator_header_user_only_messages():
    """Test that user + system only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator  
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "user", "content": "Follow up question."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages, 
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_empty_messages():
    """Test that empty messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = []
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_x_initiator_header_system_only_messages():
    """Test that system-only messages result in X-Initiator: user header"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    messages = [
        {"role": "system", "content": "You are an assistant."},
    ]
    
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["X-Initiator"] == "user"


def test_github_copilot_get_models_success():
    """Test successful retrieval of models from GitHub Copilot API"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123456789"
    config.authenticator.get_api_base.return_value = "https://api.githubcopilot.com/"
    
    # Mock the HTTP response - GitHub Copilot returns array directly with capabilities
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "gpt-4", 
            "name": "GPT-4", 
            "capabilities": {
                "type": "chat",
                "family": "gpt-4",
                "function_calling": True,
                "vision": True,
                "response_schema": True
            },
            "preview": False
        },
        {
            "id": "gpt-3.5-turbo", 
            "name": "GPT-3.5-Turbo", 
            "capabilities": {
                "type": "chat",
                "family": "gpt-3.5",
                "function_calling": True,
                "vision": False,
                "response_schema": True
            },
            "preview": False
        },
        {
            "id": "claude-3-sonnet", 
            "name": "Claude-3-Sonnet", 
            "capabilities": {
                "type": "chat",
                "family": "claude-3",
                "function_calling": True,
                "vision": True,
                "response_schema": True
            },
            "preview": True
        },
    ]
    mock_response.raise_for_status.return_value = None
    
    with patch.object(litellm.module_level_client, "get", return_value=mock_response) as mock_get:
        models = config.get_models()
        
        # Verify the API call was made correctly
        expected_headers = {
            "Authorization": "Bearer gh.test-key-123456789",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "GitHubCopilotChat/0.26.7",
            "Editor-Version": "vscode/1.99.3",
            "Editor-Plugin-Version": "copilot-chat/0.26.7",
            "Copilot-Integration-Id": "vscode-chat",
        }
        mock_get.assert_called_once_with(
            url="https://api.githubcopilot.com/models",
            headers=expected_headers,
        )
        
        # Verify the models are correctly formatted with github_copilot/ prefix
        expected_models = [
            "github_copilot/gpt-4",
            "github_copilot/gpt-3.5-turbo", 
            "github_copilot/claude-3-sonnet",
        ]
        assert models == expected_models
        
        # Verify that model capabilities were stored
        assert "github_copilot/gpt-4" in GithubCopilotConfig._model_capabilities
        assert "github_copilot/gpt-3.5-turbo" in GithubCopilotConfig._model_capabilities
        assert "github_copilot/claude-3-sonnet" in GithubCopilotConfig._model_capabilities
        
        # Verify capability details for GPT-4
        gpt4_info = GithubCopilotConfig._model_capabilities["github_copilot/gpt-4"]
        assert gpt4_info["capabilities"]["type"] == "chat"
        assert gpt4_info["capabilities"]["vision"] is True
        assert gpt4_info["preview"] is False


def test_github_copilot_get_models_authentication_error():
    """Test get_models with authentication error"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator to raise GetAPIKeyError
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = GetAPIKeyError(
        status_code=401, message="Authentication failed"
    )
    config.authenticator.get_api_base.return_value = "https://api.githubcopilot.com/"
    
    with pytest.raises(ValueError) as excinfo:
        config.get_models()
    
    assert "GitHub Copilot API key not available" in str(excinfo.value)
    assert "Authentication failed" in str(excinfo.value)


def test_github_copilot_get_provider_info():
    """Test get_provider_info method with cached model capabilities"""
    config = GithubCopilotConfig()
    
    # Set up mock model capabilities in the class cache
    GithubCopilotConfig._model_capabilities = {
        "github_copilot/gpt-4": {
            "id": "gpt-4",
            "name": "GPT-4",
            "capabilities": {
                "type": "chat",
                "family": "gpt-4",
                "function_calling": True,
                "vision": True,
                "response_schema": True
            },
            "preview": False
        },
        "github_copilot/gpt-3.5-turbo": {
            "id": "gpt-3.5-turbo", 
            "name": "GPT-3.5-Turbo",
            "capabilities": {
                "type": "chat",
                "family": "gpt-3.5",
                "function_calling": True,
                "vision": False,
                "response_schema": True
            },
            "preview": False
        }
    }
    
    # Test GPT-4 capabilities
    provider_info = config.get_provider_info("github_copilot/gpt-4")
    assert provider_info is not None
    assert provider_info["supports_function_calling"] is True
    assert provider_info["supports_vision"] is True
    assert provider_info["supports_system_messages"] is True
    assert provider_info["supports_tool_choice"] is True
    assert provider_info["supports_response_schema"] is True
    
    # Test GPT-3.5-Turbo capabilities (no vision)
    provider_info = config.get_provider_info("github_copilot/gpt-3.5-turbo")
    assert provider_info is not None
    assert provider_info["supports_function_calling"] is True
    assert provider_info["supports_vision"] is False  # Different from GPT-4
    assert provider_info["supports_system_messages"] is True
    assert provider_info["supports_tool_choice"] is True
    assert provider_info["supports_response_schema"] is True


def test_github_copilot_get_base_model():
    """Test static get_base_model method"""
    # Test with github_copilot/ prefix
    assert GithubCopilotConfig.get_base_model("github_copilot/gpt-4") == "gpt-4"
    
    # Test with no prefix
    assert GithubCopilotConfig.get_base_model("gpt-4") == "gpt-4"
    
    # Test with None
    assert GithubCopilotConfig.get_base_model(None) is None
    
    # Test with empty string
    assert GithubCopilotConfig.get_base_model("") == ""


def test_copilot_vision_request_header_with_image_content():
    """Test that Copilot-Vision-Request header is set when messages contain images"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    # Messages with image content (OpenAI multimodal format)
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
            ]
        },
    ]
    
    # Test with a vision-capable model (GPT-4)
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    assert headers["Copilot-Vision-Request"] == "true"
    assert headers["X-Initiator"] == "user"


def test_copilot_vision_request_header_with_non_vision_model():
    """Test that Copilot-Vision-Request header is NOT set for non-vision models"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    # Messages with image content
    messages = [
        {
            "role": "user", 
            "content": [
                {"type": "text", "text": "What's in this image?"},
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQ..."}}
            ]
        },
    ]
    
    # Test with a non-vision model (GPT-3.5-Turbo)
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-3.5-turbo",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    # Should not have vision header for non-vision model
    assert "Copilot-Vision-Request" not in headers
    assert headers["X-Initiator"] == "user"


def test_copilot_vision_request_header_without_images():
    """Test that Copilot-Vision-Request header is NOT set when no images are present"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    # Messages without image content
    messages = [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Hello, how are you?"},
    ]
    
    # Test with a vision-capable model (GPT-4)
    headers = config.validate_environment(
        headers={},
        model="github_copilot/gpt-4",
        messages=messages,
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )
    
    # Should not have vision header without images
    assert "Copilot-Vision-Request" not in headers
    assert headers["X-Initiator"] == "user"


def test_has_image_content_various_formats():
    """Test _has_image_content method with various image content formats"""
    config = GithubCopilotConfig()
    
    # Test with image_url in multimodal content
    messages_with_image_url = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Look at this"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}}
            ]
        }
    ]
    assert config._has_image_content(messages_with_image_url) is True
    
    # Test with base64 image in string content
    messages_with_base64 = [
        {
            "role": "user",
            "content": "Here's an image: data:image/png;base64,iVBORw0KGgoAAAANS..."
        }
    ]
    assert config._has_image_content(messages_with_base64) is True
    
    # Test with no images
    messages_no_images = [
        {"role": "user", "content": "Just text here"},
        {"role": "user", "content": [{"type": "text", "text": "More text"}]}
    ]
    assert config._has_image_content(messages_no_images) is False
    
    # Test with None content
    messages_with_none = [
        {"role": "user", "content": None}
    ]
    assert config._has_image_content(messages_with_none) is False


def test_model_supports_vision_various_models():
    """Test _model_supports_vision method with various model names"""
    config = GithubCopilotConfig()
    
    # Clear any cached capabilities for clean test
    GithubCopilotConfig._model_capabilities = {}
    
    # Test GPT-4 models (should support vision)
    assert config._model_supports_vision("github_copilot/gpt-4") is True
    assert config._model_supports_vision("gpt-4") is True  # Without prefix
    assert config._model_supports_vision("github_copilot/gpt-4-turbo") is True
    
    # Test GPT-4o models (should support vision)
    assert config._model_supports_vision("github_copilot/gpt-4o") is True
    assert config._model_supports_vision("github_copilot/gpt-4o-mini") is True
    
    # Test Claude-3 models (should support vision)
    assert config._model_supports_vision("github_copilot/claude-3-sonnet") is True
    assert config._model_supports_vision("github_copilot/claude-3-haiku") is True
    
    # Test GPT-3.5 models (should not support vision by default)
    assert config._model_supports_vision("github_copilot/gpt-3.5-turbo") is False
    
    # Test unknown model (should not support vision by default)
    assert config._model_supports_vision("github_copilot/unknown-model") is False


def test_model_supports_vision_with_cached_capabilities():
    """Test _model_supports_vision method with cached model capabilities"""
    config = GithubCopilotConfig()
    
    # Set up cached model capabilities
    GithubCopilotConfig._model_capabilities = {
        "github_copilot/vision-model": {
            "capabilities": {
                "supports": {"vision": True}
            }
        },
        "github_copilot/no-vision-model": {
            "capabilities": {
                "supports": {"vision": False}
            }
        },
        "github_copilot/legacy-vision-model": {
            "capabilities": {
                "vision": True  # Direct field instead of supports object
            }
        }
    }
    
    # Test with cached vision support
    assert config._model_supports_vision("github_copilot/vision-model") is True
    assert config._model_supports_vision("github_copilot/no-vision-model") is False
    assert config._model_supports_vision("github_copilot/legacy-vision-model") is True


def test_github_copilot_capabilities_integration():
    """Test integration between get_models and get_provider_info"""
    config = GithubCopilotConfig()
    
    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123456789"
    config.authenticator.get_api_base.return_value = "https://api.githubcopilot.com/"
    
    # Mock response with model capabilities
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {
            "id": "test-model",
            "name": "Test Model",
            "capabilities": {
                "type": "chat",
                "function_calling": False,  # Explicitly disabled
                "vision": True,
                "response_schema": False
            },
            "preview": True
        }
    ]
    mock_response.raise_for_status.return_value = None
    
    with patch.object(litellm.module_level_client, "get", return_value=mock_response):
        # First call get_models to populate capabilities cache
        models = config.get_models()
        assert models == ["github_copilot/test-model"]
        
        # Then test that get_provider_info uses the cached capabilities
        provider_info = config.get_provider_info("github_copilot/test-model")
        assert provider_info is not None
        assert provider_info["supports_function_calling"] is False  # From API response
        assert provider_info["supports_vision"] is True  # From API response
        assert provider_info["supports_response_schema"] is False  # From API response
        assert provider_info["supports_system_messages"] is True  # Always true for chat models
        assert provider_info["supports_tool_choice"] is True  # Always true for chat models
