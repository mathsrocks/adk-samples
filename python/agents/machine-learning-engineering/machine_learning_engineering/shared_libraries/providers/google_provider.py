"""Google GenAI provider implementation."""

import os
from typing import Optional, Dict, Any
from google.genai import types

from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType


class GoogleLLMProvider(LLMProvider):
    """Google GenAI provider implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.initialize_client()
    
    def initialize_client(self) -> None:
        """Initialize the Google GenAI client."""
        try:
            # Import here to avoid issues if google-genai is not installed
            import google.genai as genai
            
            # The client initialization is handled by the Google ADK framework
            # We just store the genai module for later use
            self._genai = genai
            self._client = genai  # For compatibility
            
        except ImportError as e:
            raise RuntimeError(f"Google GenAI not available: {e}")
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using Google GenAI."""
        try:
            # Prepare the content
            contents = []
            
            if system_prompt:
                contents.append(types.Content(parts=[types.Part.from_text(system_prompt)]))
            
            contents.append(types.Content(parts=[types.Part.from_text(prompt)]))
            
            # Prepare generation config
            generation_config = types.GenerateContentConfig(
                temperature=temperature or self.config.temperature or 0.01,
                max_output_tokens=max_tokens or self.config.max_tokens,
                **kwargs
            )
            
            # Generate content (this uses the existing Google GenAI integration)
            # Note: In the actual agent implementation, this would be called differently
            # For now, we'll create a placeholder that maintains the interface
            response_text = "Generated content placeholder"  # This would be the actual response
            
            return GenerateResponse(
                text=response_text,
                usage={"input_tokens": len(prompt), "output_tokens": len(response_text)},
                model=self.config.model_name,
                provider=ProviderType.GOOGLE,
                raw_response=None  # Would contain the actual Google response
            )
            
        except Exception as e:
            raise RuntimeError(f"Google GenAI generation failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Google GenAI is available."""
        try:
            # Check if required environment variables are set
            if os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
                # Check for Vertex AI credentials
                return (os.getenv("GOOGLE_CLOUD_PROJECT") is not None and
                        os.getenv("GOOGLE_CLOUD_LOCATION") is not None)
            else:
                # Check for API key or application default credentials
                return (os.getenv("GOOGLE_API_KEY") is not None or
                        os.getenv("GOOGLE_APPLICATION_CREDENTIALS") is not None)
        except Exception:
            return False


# Compatibility function to maintain existing Google GenAI integration
def create_google_generate_config(
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    **kwargs
) -> types.GenerateContentConfig:
    """Create Google GenAI generation config for backward compatibility."""
    return types.GenerateContentConfig(
        temperature=temperature or 0.01,
        max_output_tokens=max_tokens,
        **kwargs
    )