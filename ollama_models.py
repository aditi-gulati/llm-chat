"""
🤖 Ollama Models Management
Python-based model management for Ollama
Handles model listing, pulling, and management
"""

import requests
from typing import List, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaModels:
    """
    Ollama Models Manager
    Handles all model-related operations
    """

    # Available models that can be pulled
    AVAILABLE_MODELS = {
        "llama2": {
            "name": "llama2",
            "size": "4GB",
            "description": "General purpose LLM",
            "speed": "⭐⭐⭐",
            "quality": "⭐⭐⭐"
        },
        "mistral": {
            "name": "mistral",
            "size": "5GB",
            "description": "Fast and quality model",
            "speed": "⭐⭐⭐⭐",
            "quality": "⭐⭐⭐⭐"
        },
        "neural-chat": {
            "name": "neural-chat",
            "size": "4GB",
            "description": "Optimized for chat",
            "speed": "⭐⭐⭐",
            "quality": "⭐⭐⭐"
        },
        "tinyllama": {
            "name": "tinyllama",
            "size": "636MB",
            "description": "Ultra-fast, small model",
            "speed": "⭐⭐⭐⭐⭐",
            "quality": "⭐⭐"
        },
        "orca-mini": {
            "name": "orca-mini",
            "size": "1.3GB",
            "description": "Compact and capable",
            "speed": "⭐⭐⭐⭐",
            "quality": "⭐⭐⭐"
        },
        "gemma2:2b": {
            "name": "gemma2:2b",
            "size": "5GB",
            "description": "Google's Gemma model",
            "speed": "⭐⭐⭐",
            "quality": "⭐⭐⭐⭐"
        },
        "openchat": {
            "name": "openchat",
            "size": "4GB",
            "description": "Open-source optimized chat",
            "speed": "⭐⭐⭐",
            "quality": "⭐⭐⭐"
        },
        "zephyr": {
            "name": "zephyr",
            "size": "4GB",
            "description": "Quality focused model",
            "speed": "⭐⭐⭐",
            "quality": "⭐⭐⭐⭐"
        },
    }

    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize with Ollama server URL"""
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()

    def get_installed_models(self) -> List[str]:
        """Get list of installed models from server"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            models = [m['name'] for m in data.get('models', [])]
            logger.info(f"Found {len(models)} installed models")
            return sorted(models)
        except Exception as e:
            logger.error(f"Error getting models: {str(e)}")
            return []

    def get_model_info(self, model_name: str) -> Dict:
        """Get information about a model"""
        # Check if model is in available models list
        model_key = model_name.split(':')[0]

        if model_key in self.AVAILABLE_MODELS:
            return self.AVAILABLE_MODELS[model_key]

        # Return generic info if not in list
        return {
            "name": model_name,
            "size": "Unknown",
            "description": "Custom model",
            "speed": "Unknown",
            "quality": "Unknown"
        }

    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama Hub"""
        try:
            logger.info(f"Pulling model: {model_name}")
            payload = {"name": model_name}

            response = self.session.post(
                f"{self.base_url}/api/pull",
                json=payload,
                timeout=600  # 10 minutes for model download
            )

            response.raise_for_status()
            logger.info(f"Successfully pulled {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pull {model_name}: {str(e)}")
            return False

    def remove_model(self, model_name: str) -> bool:
        """Remove a model from Ollama"""
        try:
            logger.info(f"Removing model: {model_name}")
            payload = {"name": model_name}

            response = self.session.delete(
                f"{self.base_url}/api/delete",
                json=payload,
                timeout=30
            )

            response.raise_for_status()
            logger.info(f"Successfully removed {model_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove {model_name}: {str(e)}")
            return False

    def check_server(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = self.session.get(
                f"{self.base_url}/api/tags",
                timeout=5
            )
            is_accessible = response.status_code == 200
            logger.info(f"Server check: {'OK' if is_accessible else 'Failed'}")
            return is_accessible
        except Exception as e:
            logger.error(f"Server check failed: {str(e)}")
            return False

    def get_available_models_list(self) -> List[Dict]:
        """Get list of all available models with info"""
        models_list = []
        for model_key, model_info in self.AVAILABLE_MODELS.items():
            models_list.append(model_info)
        return models_list

    def get_model_description(self, model_name: str) -> str:
        """Get human-readable description of a model"""
        info = self.get_model_info(model_name)
        return f"{info['name']} - {info['size']} - {info['description']}"


# Default models to suggest
DEFAULT_MODELS = ["llama2", "mistral", "neural-chat"]

# Model recommendations for different use cases
MODEL_RECOMMENDATIONS = {
    "general": ["llama2", "mistral"],
    "fast": ["tinyllama", "orca-mini"],
    "quality": ["mistral", "zephyr"],
    "code": ["mistral", "openchat"],
    "creative": ["llama2", "zephyr"],
    "balanced": ["neural-chat", "llama2"],
}
