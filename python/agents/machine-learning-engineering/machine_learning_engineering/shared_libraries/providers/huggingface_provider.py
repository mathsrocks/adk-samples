"""Hugging Face provider implementation."""

import os
import requests
from typing import Optional

from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType


class HuggingFaceLLMProvider(LLMProvider):
    """Hugging Face provider implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.initialize_client()
    
    def initialize_client(self) -> None:
        """Initialize the Hugging Face client."""
        self._base_url = self.config.base_url or "https://api-inference.huggingface.co"
        self._session = requests.Session()
        
        api_key = None
        if self.config.api_key_env_var:
            api_key = os.getenv(self.config.api_key_env_var)
        
        if api_key:
            self._session.headers.update({"Authorization": f"Bearer {api_key}"})
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using Hugging Face API."""
        try:
            # Prepare the input
            full_prompt = prompt
            if system_prompt:
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "temperature": temperature or self.config.temperature or 0.7,
                    "max_new_tokens": max_tokens or self.config.max_tokens or 512,
                    "return_full_text": False,
                    **kwargs
                }
            }
            
            response = self._session.post(
                f"{self._base_url}/models/{self.config.model_name}",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            else:
                content = str(result)
            
            # HF doesn't provide token usage, estimate it
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
                provider=ProviderType.HUGGINGFACE,
                raw_response=result,
                cost_estimate=0.0  # Free tier available
            )
            
        except Exception as e:
            raise RuntimeError(f"Hugging Face generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Hugging Face is available."""
        try:
            # Try to ping the API
            response = self._session.get(f"{self._base_url}/models/{self.config.model_name}", timeout=10)
            return response.status_code == 200
        except Exception:
            return False