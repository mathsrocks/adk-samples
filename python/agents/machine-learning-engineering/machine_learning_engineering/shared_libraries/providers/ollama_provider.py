"""Ollama provider implementation for local LLM inference."""

import os
import requests
from typing import Optional, Dict, Any

from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType


class OllamaLLMProvider(LLMProvider):
    """Ollama provider for local LLM inference."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.initialize_client()
    
    def initialize_client(self) -> None:
        """Initialize the Ollama client."""
        # Ollama uses HTTP API, no special client needed
        self._base_url = self.config.base_url or "http://localhost:11434"
        self._client = requests.Session()
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using Ollama API."""
        try:
            # Prepare the prompt with system message if provided
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "model": self.config.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": temperature or self.config.temperature or 0.7,
                    "num_predict": max_tokens or self.config.max_tokens or 2048,
                    **kwargs
                }
            }
            
            response = self._client.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=300  # 5 minute timeout for local generation
            )
            response.raise_for_status()
            
            result = response.json()
            content = result.get("response", "")
            
            # Ollama doesn't provide token usage, estimate it
            estimated_input_tokens = len(full_prompt.split())
            estimated_output_tokens = len(content.split())
            
            usage = {
                "input_tokens": estimated_input_tokens,
                "output_tokens": estimated_output_tokens,
                "total_tokens": estimated_input_tokens + estimated_output_tokens
            }
            
            return GenerateResponse(
                text=content,
                usage=usage,
                model=self.config.model_name,
                provider=ProviderType.OLLAMA,
                raw_response=result,
                cost_estimate=0.0  # Local inference is free
            )
            
        except Exception as e:
            raise RuntimeError(f"Ollama generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Ollama is available."""
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                # Check if the specific model is available
                models = response.json().get("models", [])
                model_names = [model.get("name", "") for model in models]
                return any(self.config.model_name in name for name in model_names)
            return False
        except Exception:
            return False
    
    def list_available_models(self) -> list:
        """List available models in Ollama."""
        try:
            response = self._client.get(f"{self._base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model.get("name", "") for model in models]
            return []
        except Exception:
            return []