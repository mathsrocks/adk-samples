"""Anthropic Claude provider implementation."""

import os
from typing import Optional

from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType


class AnthropicLLMProvider(LLMProvider):
    """Anthropic Claude provider implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.initialize_client()
    
    def initialize_client(self) -> None:
        """Initialize the Anthropic client."""
        try:
            from anthropic import Anthropic
            
            api_key = None
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
            
            self._client = Anthropic(
                api_key=api_key,
                base_url=self.config.base_url
            )
            
        except ImportError as e:
            raise RuntimeError(f"Anthropic package not available. Install with: pip install anthropic. Error: {e}")
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using Anthropic Claude API."""
        try:
            # Prepare messages
            messages = [{"role": "user", "content": prompt}]
            
            # Prepare request parameters
            request_params = {
                "model": self.config.model_name,
                "messages": messages,
                "temperature": temperature or self.config.temperature or 0.7,
                "max_tokens": max_tokens or self.config.max_tokens or 4096,
                **kwargs
            }
            
            # Add system prompt if provided
            if system_prompt:
                request_params["system"] = system_prompt
            
            response = self._client.messages.create(**request_params)
            
            content = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens if response.usage else 0,
                "output_tokens": response.usage.output_tokens if response.usage else 0,
                "total_tokens": (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0
            }
            
            # Calculate cost estimate
            cost_estimate = None
            if self.config.cost_per_token and response.usage:
                cost_estimate = self.estimate_cost(
                    response.usage.input_tokens,
                    response.usage.output_tokens
                )
            
            return GenerateResponse(
                text=content,
                usage=usage,
                model=response.model,
                provider=ProviderType.ANTHROPIC,
                raw_response=response,
                cost_estimate=cost_estimate
            )
            
        except Exception as e:
            raise RuntimeError(f"Anthropic generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Anthropic is available."""
        try:
            if self.config.api_key_env_var:
                api_key = os.getenv(self.config.api_key_env_var)
                if not api_key:
                    return False
            
            if self._client:
                return True
            return False
        except Exception:
            return False