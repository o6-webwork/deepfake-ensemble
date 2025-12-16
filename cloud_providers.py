"""
Cloud VLM Provider Adapters for Gemini, OpenAI, and Anthropic Claude.

This module provides unified interfaces to cloud vision-language models,
normalizing different API formats to work with the OSINT detection pipeline.
"""

import base64
from typing import Dict, List, Any, Optional
from openai import OpenAI


class CloudProviderAdapter:
    """Base adapter for cloud VLM providers."""

    def __init__(self, model_name: str, api_key: str):
        self.model_name = model_name
        self.api_key = api_key

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create a chat completion. Must be implemented by subclasses."""
        raise NotImplementedError


class OpenAIAdapter(CloudProviderAdapter):
    """Adapter for OpenAI GPT-4V models (uses native OpenAI SDK)."""

    def __init__(self, model_name: str, api_key: str, base_url: str = "https://api.openai.com/v1/"):
        super().__init__(model_name, api_key)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=180.0,
            max_retries=0
        )

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using OpenAI API."""
        kwargs = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if logprobs:
            kwargs["logprobs"] = True
            if top_logprobs:
                kwargs["top_logprobs"] = top_logprobs

        return self.client.chat.completions.create(**kwargs)


class AnthropicAdapter(CloudProviderAdapter):
    """Adapter for Anthropic Claude models."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            import anthropic
            self.client = anthropic.Anthropic(api_key=api_key, timeout=180.0)
        except ImportError:
            raise ImportError(
                "Anthropic SDK not installed. Run: pip install anthropic"
            )

    def _convert_messages(self, messages: List[Dict]) -> tuple:
        """
        Convert OpenAI-style messages to Anthropic format.

        Returns:
            (system_prompt, anthropic_messages)
        """
        system_prompt = ""
        anthropic_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                # Convert content format
                if isinstance(msg["content"], str):
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
                elif isinstance(msg["content"], list):
                    # Convert multi-part content
                    converted_content = []
                    for part in msg["content"]:
                        if part["type"] == "text":
                            converted_content.append({
                                "type": "text",
                                "text": part["text"]
                            })
                        elif part["type"] == "image_url":
                            # Extract base64 from data URI
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                # Format: data:image/png;base64,<base64_data>
                                media_type = image_url.split(";")[0].split(":")[1]
                                base64_data = image_url.split(",")[1]
                                converted_content.append({
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": base64_data
                                    }
                                })

                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": converted_content
                    })

        return system_prompt, anthropic_messages

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using Anthropic API."""
        system_prompt, anthropic_messages = self._convert_messages(messages)

        # Note: Claude doesn't support logprobs natively
        # We'll return a mock response object with the text
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt,
            messages=anthropic_messages
        )

        # Convert to OpenAI-like format
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})
                self.logprobs = None

        class MockResponse:
            def __init__(self, response):
                self.choices = [MockChoice(response.content[0].text)]

        return MockResponse(response)


class GeminiAdapter(CloudProviderAdapter):
    """Adapter for Google Gemini models."""

    def __init__(self, model_name: str, api_key: str):
        super().__init__(model_name, api_key)
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(model_name)
        except ImportError:
            raise ImportError(
                "Google Generative AI SDK not installed. Run: pip install google-generativeai"
            )

    def _convert_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Convert OpenAI-style messages to Gemini format.

        Gemini doesn't have a system role, so we prepend system content to first user message.
        """
        import google.generativeai as genai

        system_content = ""
        gemini_messages = []

        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"] + "\n\n"
            elif msg["role"] == "user":
                parts = []

                # Add system prompt to first user message
                if system_content and len(gemini_messages) == 0:
                    parts.append(system_content)

                # Process content
                if isinstance(msg["content"], str):
                    parts.append(msg["content"])
                elif isinstance(msg["content"], list):
                    for part in msg["content"]:
                        if part["type"] == "text":
                            parts.append(part["text"])
                        elif part["type"] == "image_url":
                            # Extract base64 from data URI
                            image_url = part["image_url"]["url"]
                            if image_url.startswith("data:"):
                                base64_data = image_url.split(",")[1]
                                # Decode base64 to bytes
                                import base64 as b64
                                image_bytes = b64.b64decode(base64_data)
                                # Create PIL Image
                                from PIL import Image
                                import io
                                pil_image = Image.open(io.BytesIO(image_bytes))
                                parts.append(pil_image)

                gemini_messages.append({
                    "role": "user",
                    "parts": parts
                })
            elif msg["role"] == "assistant":
                if isinstance(msg["content"], str):
                    gemini_messages.append({
                        "role": "model",
                        "parts": [msg["content"]]
                    })

        return gemini_messages

    def create_completion(
        self,
        messages: List[Dict],
        temperature: float = 0.0,
        max_tokens: int = 500,
        logprobs: bool = False,
        top_logprobs: Optional[int] = None
    ) -> Any:
        """Create chat completion using Gemini API."""
        gemini_messages = self._convert_messages(messages)

        # Extract parts from last user message
        last_message = gemini_messages[-1]
        parts = last_message["parts"]

        # Build chat history (all messages except last)
        history = []
        for msg in gemini_messages[:-1]:
            history.append({
                "role": msg["role"],
                "parts": msg["parts"]
            })

        # Generate response
        generation_config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }

        if history:
            chat = self.client.start_chat(history=history)
            response = chat.send_message(parts, generation_config=generation_config)
        else:
            response = self.client.generate_content(parts, generation_config=generation_config)

        # Convert to OpenAI-like format
        class MockChoice:
            def __init__(self, content):
                self.message = type('obj', (object,), {'content': content})
                self.logprobs = None

        class MockResponse:
            def __init__(self, response):
                self.choices = [MockChoice(response.text)]

        return MockResponse(response)


def get_cloud_adapter(provider: str, model_name: str, api_key: str, base_url: Optional[str] = None):
    """
    Factory function to get the appropriate cloud provider adapter.

    Args:
        provider: "openai", "anthropic", or "gemini"
        model_name: Model identifier
        api_key: API key for the provider
        base_url: Base URL (only for OpenAI-compatible)

    Returns:
        CloudProviderAdapter instance
    """
    if provider == "openai":
        return OpenAIAdapter(model_name, api_key, base_url or "https://api.openai.com/v1/")
    elif provider == "anthropic":
        return AnthropicAdapter(model_name, api_key)
    elif provider == "gemini":
        return GeminiAdapter(model_name, api_key)
    elif provider == "vllm":
        # For vLLM, return OpenAI adapter (OpenAI-compatible)
        return OpenAIAdapter(model_name, api_key or "dummy", base_url)
    else:
        raise ValueError(f"Unknown provider: {provider}")
