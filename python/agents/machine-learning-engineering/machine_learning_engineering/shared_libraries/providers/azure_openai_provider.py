"""Azure_openai provider implementation."""

from typing import Optional
from ..llm_providers import LLMProvider, ModelConfig, GenerateResponse, ProviderType

class Azure_openaiLLMProvider(LLMProvider):
    """Azure_openai provider implementation."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        
    def initialize_client(self) -> None:
        """Initialize the Azure_openai client."""
        # TODO: Implement Azure_openai client initialization
        pass
    
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using Azure_openai API."""
        # TODO: Implement Azure_openai content generation
        raise NotImplementedError("Azure_openai provider not yet implemented")
    
    def is_available(self) -> bool:
        """Check if Azure_openai is available."""
        # TODO: Implement availability check
        return False
