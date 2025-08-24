"""OpenAI provider implementation."""

import os
from typing import Optional

from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType


class OpenAILLMProvider(LLMProvider):
    """OpenAI provider implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.initialize_client()
    
    def initialize_client(self) -> None:
        """Initialize the OpenAI client."""
        try:
            from openai import OpenAI
            
            api_key = None
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
            
            self._client = OpenAI(
                api_key=api_key,
                base_url=self.config.base_url
            )
            
        except ImportError as e:
            raise RuntimeError(f"OpenAI package not available. Install with: pip install openai. Error: {e}")
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using OpenAI API."""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            messages.append({"role": "user", "content": prompt})
            
            response = self._client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature or 0.7,
                max_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            
            content = response.choices[0].message.content
            usage = {
                "input_tokens": response.usage.prompt_tokens if response.usage else 0,
                "output_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }
            
            # Calculate cost estimate
            cost_estimate = None
            if self.config.cost_per_token and response.usage:
                cost_estimate = self.estimate_cost(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens
                )
            
            return GenerateResponse(
                text=content,
                usage=usage,
                model=response.model,
                provider=ProviderType.OPENAI,
                raw_response=response,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            raise RuntimeError(f"OpenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenAI is available."""
        try:
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
                if not api_key:
                    return False
            
            # Try to make a minimal request to test availability
            if self._client:
                return True
            return False
        except Exception:
            return False