"""
�� Unified LLM Provider - Support for Ollama, OpenAI, and Anthropic (Claude)
Handles model management, chat operations for multiple LLM providers
"""

import os
import json
import requests
from typing import List, Dict, Optional, Generator
import logging
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def get_models(self) -> List[str]:
        """Get list of available models."""
        pass

    @abstractmethod
    def chat(self, model: str, messages: List[Dict], **kwargs) -> Generator[str, None, None]:
        """Send chat message and stream response."""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM Provider - Local/Remote"""

    AVAILABLE_MODELS = {
        "tinyllama": {"name": "TinyLlama", "size": "1.1B", "memory": "~2GB", "speed": "⭐⭐⭐⭐⭐", "quality": "⭐⭐", "best_for": "Testing, simple tasks"},
        "orca-mini": {"name": "Orca Mini", "size": "3B", "memory": "~3GB", "speed": "⭐⭐⭐⭐", "quality": "⭐⭐⭐", "best_for": "Quick responses"},
        "gemma2:2b": {"name": "Google Gemma 2", "size": "2B", "memory": "~2.5GB", "speed": "⭐⭐⭐⭐⭐", "quality": "⭐⭐⭐", "best_for": "Lightweight"},
        "openchat": {"name": "OpenChat", "size": "3.5B", "memory": "~4GB", "speed": "⭐⭐⭐⭐", "quality": "⭐⭐⭐", "best_for": "Chat, balanced"},
        "neural-chat": {"name": "Neural Chat", "size": "7B", "memory": "~5GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "Chat"},
        "mistral": {"name": "Mistral", "size": "7B", "memory": "~5GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "General purpose, coding"},
        "llama2": {"name": "Llama 2", "size": "7B", "memory": "~5GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐", "best_for": "General purpose"},
        "zephyr": {"name": "Zephyr", "size": "7B", "memory": "~5GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "High-quality responses"},
        "starling-lm": {"name": "Starling LM", "size": "7B", "memory": "~5GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "High-quality chat"},
        "solar": {"name": "Solar", "size": "10.7B", "memory": "~8GB", "speed": "⭐⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "Complex reasoning"},
        "dolphin-mixtral": {"name": "Dolphin Mixtral", "size": "47B", "memory": "~32GB", "speed": "⭐⭐", "quality": "⭐⭐⭐⭐⭐", "best_for": "Enterprise"},
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')
        self.timeout = 30
        self.session = requests.Session()
        self.provider_name = "Ollama"

    def get_models(self) -> List[str]:
        """Get installed models from Ollama server."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            if response.status_code == 200:
                data = response.json()
                models = [m["name"].split(":")[0] for m in data.get("models", [])]
                return sorted(list(set(models)))
            return []
        except Exception as e:
            logger.error(f"Failed to get Ollama models: {e}")
            return []

    def is_running(self) -> bool:
        """Check if Ollama server is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=self.timeout)
            return response.status_code == 200
        except:
            return False

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 40,
        num_predict: int = 512,
        num_ctx: int = 2048,
        **kwargs
    ) -> Generator[str, None, None]:
        """Chat with Ollama model."""
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": True,
                "options": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                    "num_predict": num_predict,
                    "num_ctx": num_ctx
                }
            }

            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=None,
                stream=True
            )

            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line)
                        chunk = data.get("message", {}).get("content", "")
                        if chunk:
                            yield chunk
        except Exception as e:
            logger.error(f"Ollama chat error: {e}")
            yield f"Error: {e}"

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information."""
        return self.AVAILABLE_MODELS.get(model_name)


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider - GPT-4, GPT-3.5-turbo"""

    AVAILABLE_MODELS = {
        "gpt-4-turbo": {"name": "GPT-4 Turbo", "provider": "OpenAI", "quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐⭐", "best_for": "Complex reasoning, coding"},
        "gpt-4": {"name": "GPT-4", "provider": "OpenAI", "quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐", "best_for": "Advanced reasoning"},
        "gpt-3.5-turbo": {"name": "GPT-3.5 Turbo", "provider": "OpenAI", "quality": "⭐⭐⭐⭐", "speed": "⭐⭐⭐⭐⭐", "best_for": "Fast, cost-effective"},
    }

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.provider_name = "OpenAI"
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    def get_models(self) -> List[str]:
        """Get list of available OpenAI models."""
        return list(self.AVAILABLE_MODELS.keys())

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Chat with OpenAI model."""
        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                stream=True
            )

            for chunk in response:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"OpenAI chat error: {e}")
            yield f"Error: {e}"

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information."""
        return self.AVAILABLE_MODELS.get(model_name)


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM Provider - Sonnet, Haiku, Opus"""

    AVAILABLE_MODELS = {
        "claude-3-sonnet-20240229": {"name": "Claude 3 Sonnet", "provider": "Anthropic", "quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐⭐⭐", "best_for": "Balanced quality and speed"},
        "claude-3-haiku-20240307": {"name": "Claude 3 Haiku", "provider": "Anthropic", "quality": "⭐⭐⭐⭐", "speed": "⭐⭐⭐⭐⭐", "best_for": "Fast, lightweight"},
        "claude-3-opus-20240229": {"name": "Claude 3 Opus", "provider": "Anthropic", "quality": "⭐⭐⭐⭐⭐", "speed": "⭐⭐⭐", "best_for": "Complex reasoning"},
    }

    def __init__(self, api_key: str):
        try:
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            self.provider_name = "Anthropic"
        except ImportError:
            raise ImportError("anthropic package not installed. Run: pip install anthropic")

    def get_models(self) -> List[str]:
        """Get list of available Claude models."""
        return list(self.AVAILABLE_MODELS.keys())

    def chat(
        self,
        model: str,
        messages: List[Dict],
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """Chat with Claude model."""
        try:
            with self.client.messages.stream(
                model=model,
                max_tokens=1024,
                temperature=temperature,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.error(f"Anthropic chat error: {e}")
            yield f"Error: {e}"

    def get_model_info(self, model_name: str) -> Optional[Dict]:
        """Get model information."""
        return self.AVAILABLE_MODELS.get(model_name)


class UnifiedLLMManager:
    """Unified manager for multiple LLM providers."""

    def __init__(self):
        self.providers = {}
        self._init_providers()

    def _init_providers(self):
        """Initialize all available providers."""
        # Ollama
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.providers["ollama"] = OllamaProvider(ollama_url)

        # OpenAI
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key:
            try:
                self.providers["openai"] = OpenAIProvider(openai_key)
            except ImportError:
                logger.warning("OpenAI provider not available")

        # Anthropic
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_key:
            try:
                self.providers["anthropic"] = AnthropicProvider(anthropic_key)
            except ImportError:
                logger.warning("Anthropic provider not available")

    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return list(self.providers.keys())

    def get_provider(self, provider_name: str) -> Optional[LLMProvider]:
        """Get a specific provider."""
        return self.providers.get(provider_name)

    def get_models_for_provider(self, provider_name: str) -> List[str]:
        """Get available models for a provider."""
        provider = self.get_provider(provider_name)
        if provider:
            return provider.get_models()
        return []

    def get_all_models(self) -> Dict[str, List[Dict]]:
        """Get all models organized by provider."""
        all_models = {}
        for provider_name, provider in self.providers.items():
            models = provider.get_models()
            all_models[provider_name] = [
                {
                    "name": model,
                    "info": provider.get_model_info(model) or {"name": model}
                }
                for model in models
            ]
        return all_models

    def chat(
        self,
        provider: str,
        model: str,
        messages: List[Dict],
        **kwargs
    ) -> Generator[str, None, None]:
        """Send chat message using specified provider."""
        prov = self.get_provider(provider)
        if not prov:
            yield f"Provider '{provider}' not available"
            return

        yield from prov.chat(model, messages, **kwargs)


# Factory function
def get_llm_manager() -> UnifiedLLMManager:
    """Get unified LLM manager instance."""
    return UnifiedLLMManager()
