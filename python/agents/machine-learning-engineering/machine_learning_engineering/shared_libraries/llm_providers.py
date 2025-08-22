"""LLM Provider abstraction layer for multi-provider support."""

import abc
import os
from typing import Any, Dict, List, Optional, Union
from enum import Enum
import dataclasses


class ProviderType(Enum):
    """Supported LLM providers."""
    GOOGLE = "google"
    OPENAI = "openai" 
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"
    OLLAMA = "ollama"
    TOGETHER = "together"
    GROQ = "groq"
    MISTRAL = "mistral"
    HUGGINGFACE = "huggingface"
    REPLICATE = "replicate"
    COHERE = "cohere"
    AZURE_OPENAI = "azure_openai"
    AWS_BEDROCK = "aws_bedrock"
    PERPLEXITY = "perplexity"
    FIREWORKS = "fireworks"


@dataclasses.dataclass
class ModelConfig:
    """Configuration for a specific model."""
    model_name: str
    provider: ProviderType
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context_length: Optional[int] = None
    cost_per_token: Optional[Dict[str, float]] = None  # {"input": 0.001, "output": 0.002}
    additional_params: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class GenerateResponse:
    """Standardized response from LLM providers."""
    text: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    provider: Optional[ProviderType] = None
    raw_response: Optional[Any] = None
    cost_estimate: Optional[float] = None


class LLMProvider(abc.ABC):
    """Abstract base class for LLM providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self._client = None
    
    @abc.abstractmethod
    def initialize_client(self) -> None:
        """Initialize the provider-specific client."""
        pass
    
    @abc.abstractmethod
    def generate_content(
        self, 
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content using the LLM provider."""
        pass
    
    @abc.abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is properly configured and available."""
        pass
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.config.model_name
    
    def get_provider_type(self) -> ProviderType:
        """Get the provider type."""
        return self.config.provider
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> Optional[float]:
        """Estimate cost based on token usage."""
        if not self.config.cost_per_token:
            return None
        
        input_cost = input_tokens * self.config.cost_per_token.get("input", 0)
        output_cost = output_tokens * self.config.cost_per_token.get("output", 0)
        return input_cost + output_cost


class LLMProviderFactory:
    """Factory for creating LLM provider instances."""
    
    # Default model configurations with cost and performance data
    DEFAULT_MODELS = {
        # Closed-source providers
        ProviderType.GOOGLE: ModelConfig(
            model_name=os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001"),
            provider=ProviderType.GOOGLE,
            api_key_env_var="GOOGLE_APPLICATION_CREDENTIALS",
            context_length=1048576,  # 1M tokens
            cost_per_token={"input": 0.00015, "output": 0.0006}  # Gemini 1.5 Flash pricing
        ),
        ProviderType.OPENAI: ModelConfig(
            model_name="gpt-4o",
            provider=ProviderType.OPENAI,
            api_key_env_var="OPENAI_API_KEY",
            context_length=128000,
            cost_per_token={"input": 0.0025, "output": 0.01}  # GPT-4o pricing
        ),
        ProviderType.ANTHROPIC: ModelConfig(
            model_name="claude-3-5-sonnet-20241022",
            provider=ProviderType.ANTHROPIC,
            api_key_env_var="ANTHROPIC_API_KEY",
            context_length=200000,
            cost_per_token={"input": 0.003, "output": 0.015}  # Claude 3.5 Sonnet pricing
        ),
        ProviderType.DEEPSEEK: ModelConfig(
            model_name="deepseek-chat",
            provider=ProviderType.DEEPSEEK,
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            context_length=64000,
            cost_per_token={"input": 0.00014, "output": 0.00028}  # Very cost-effective
        ),
        ProviderType.GROQ: ModelConfig(
            model_name="llama-3.3-70b-versatile",
            provider=ProviderType.GROQ,
            api_key_env_var="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
            context_length=131072,
            cost_per_token={"input": 0.00059, "output": 0.00079}  # Fast inference
        ),
        ProviderType.TOGETHER: ModelConfig(
            model_name="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            provider=ProviderType.TOGETHER,
            api_key_env_var="TOGETHER_API_KEY",
            base_url="https://api.together.xyz/v1",
            context_length=131072,
            cost_per_token={"input": 0.00088, "output": 0.00088}
        ),
        ProviderType.MISTRAL: ModelConfig(
            model_name="mistral-large-latest",
            provider=ProviderType.MISTRAL,
            api_key_env_var="MISTRAL_API_KEY",
            base_url="https://api.mistral.ai/v1",
            context_length=128000,
            cost_per_token={"input": 0.002, "output": 0.006}
        ),
        ProviderType.PERPLEXITY: ModelConfig(
            model_name="llama-3.1-sonar-large-128k-online",
            provider=ProviderType.PERPLEXITY,
            api_key_env_var="PERPLEXITY_API_KEY",
            base_url="https://api.perplexity.ai",
            context_length=127072,
            cost_per_token={"input": 0.001, "output": 0.001}  # With web search
        ),
        ProviderType.FIREWORKS: ModelConfig(
            model_name="accounts/fireworks/models/llama-v3p1-70b-instruct",
            provider=ProviderType.FIREWORKS,
            api_key_env_var="FIREWORKS_API_KEY",
            base_url="https://api.fireworks.ai/inference/v1",
            context_length=131072,
            cost_per_token={"input": 0.0009, "output": 0.0009}  # Fast and cost-effective
        ),
        
        # Open-source / Local providers
        ProviderType.OLLAMA: ModelConfig(
            model_name="llama3.2:latest",
            provider=ProviderType.OLLAMA,
            base_url="http://localhost:11434",
            context_length=128000,
            cost_per_token={"input": 0.0, "output": 0.0}  # Free local inference
        ),
        ProviderType.HUGGINGFACE: ModelConfig(
            model_name="microsoft/Phi-3.5-mini-instruct",
            provider=ProviderType.HUGGINGFACE,
            api_key_env_var="HUGGINGFACE_API_KEY",
            base_url="https://api-inference.huggingface.co",
            context_length=128000,
            cost_per_token={"input": 0.0, "output": 0.0}  # Free tier available
        ),
        ProviderType.REPLICATE: ModelConfig(
            model_name="meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            provider=ProviderType.REPLICATE,
            api_key_env_var="REPLICATE_API_TOKEN",
            cost_per_token={"input": 0.00065, "output": 0.00275}
        ),
        
        # Enterprise providers
        ProviderType.AZURE_OPENAI: ModelConfig(
            model_name="gpt-4o",
            provider=ProviderType.AZURE_OPENAI,
            api_key_env_var="AZURE_OPENAI_API_KEY",
            additional_params={"azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT")}
        ),
        ProviderType.AWS_BEDROCK: ModelConfig(
            model_name="anthropic.claude-3-sonnet-20240229-v1:0",
            provider=ProviderType.AWS_BEDROCK,
            additional_params={
                "aws_access_key_id": os.environ.get("AWS_ACCESS_KEY_ID"),
                "aws_secret_access_key": os.environ.get("AWS_SECRET_ACCESS_KEY"),
                "aws_region": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
            }
        ),
        ProviderType.COHERE: ModelConfig(
            model_name="command-r-plus",
            provider=ProviderType.COHERE,
            api_key_env_var="COHERE_API_KEY",
            context_length=128000,
            cost_per_token={"input": 0.003, "output": 0.015}
        )
    }
    
    # Performance tiers for intelligent routing
    PERFORMANCE_TIERS = {
        "coding": [ProviderType.ANTHROPIC, ProviderType.DEEPSEEK, ProviderType.OPENAI, ProviderType.GOOGLE],
        "reasoning": [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE, ProviderType.DEEPSEEK],
        "cost_effective": [ProviderType.DEEPSEEK, ProviderType.OLLAMA, ProviderType.GROQ, ProviderType.TOGETHER],
        "fast_inference": [ProviderType.GROQ, ProviderType.FIREWORKS, ProviderType.TOGETHER, ProviderType.OLLAMA],
        "long_context": [ProviderType.GOOGLE, ProviderType.ANTHROPIC, ProviderType.GROQ, ProviderType.TOGETHER],
        "web_search": [ProviderType.PERPLEXITY, ProviderType.GOOGLE],
        "open_source": [ProviderType.OLLAMA, ProviderType.HUGGINGFACE, ProviderType.TOGETHER, ProviderType.GROQ]
    }
    
    @classmethod
    def create_provider(
        cls, 
        provider_type: Union[ProviderType, str],
        model_config: Optional[ModelConfig] = None
    ) -> LLMProvider:
        """Create an LLM provider instance."""
        if isinstance(provider_type, str):
            provider_type = ProviderType(provider_type.lower())
        
        if model_config is None:
            model_config = cls.DEFAULT_MODELS[provider_type]
        
        provider_mapping = {
            ProviderType.GOOGLE: ("google_provider", "GoogleLLMProvider"),
            ProviderType.OPENAI: ("openai_provider", "OpenAILLMProvider"),
            ProviderType.ANTHROPIC: ("anthropic_provider", "AnthropicLLMProvider"),
            ProviderType.DEEPSEEK: ("deepseek_provider", "DeepSeekLLMProvider"),
            ProviderType.OLLAMA: ("ollama_provider", "OllamaLLMProvider"),
            ProviderType.TOGETHER: ("together_provider", "TogetherLLMProvider"),
            ProviderType.GROQ: ("groq_provider", "GroqLLMProvider"),
            ProviderType.MISTRAL: ("mistral_provider", "MistralLLMProvider"),
            ProviderType.HUGGINGFACE: ("huggingface_provider", "HuggingFaceLLMProvider"),
            ProviderType.REPLICATE: ("replicate_provider", "ReplicateLLMProvider"),
            ProviderType.COHERE: ("cohere_provider", "CohereLLMProvider"),
            ProviderType.AZURE_OPENAI: ("azure_openai_provider", "AzureOpenAILLMProvider"),
            ProviderType.AWS_BEDROCK: ("aws_bedrock_provider", "AWSBedrockLLMProvider"),
            ProviderType.PERPLEXITY: ("perplexity_provider", "PerplexityLLMProvider"),
            ProviderType.FIREWORKS: ("fireworks_provider", "FireworksLLMProvider")
        }
        
        if provider_type not in provider_mapping:
            raise ValueError(f"Unsupported provider type: {provider_type}")
        
        module_name, class_name = provider_mapping[provider_type]
        module = __import__(f"machine_learning_engineering.shared_libraries.providers.{module_name}", 
                          fromlist=[class_name])
        provider_class = getattr(module, class_name)
        return provider_class(model_config)
    
    @classmethod
    def get_available_providers(cls) -> List[ProviderType]:
        """Get list of available providers based on environment configuration."""
        available = []
        for provider_type in ProviderType:
            try:
                provider = cls.create_provider(provider_type)
                if provider.is_available():
                    available.append(provider_type)
            except Exception:
                continue
        return available
    
    @classmethod
    def get_providers_by_tier(cls, tier: str) -> List[ProviderType]:
        """Get providers optimized for specific use cases."""
        return cls.PERFORMANCE_TIERS.get(tier, [])
    
    @classmethod
    def create_default_provider(cls) -> LLMProvider:
        """Create default provider based on environment configuration."""
        # Try providers in order of preference
        provider_order = [
            ProviderType.GOOGLE,      # Current default
            ProviderType.DEEPSEEK,    # Best cost/performance
            ProviderType.ANTHROPIC,   # Best for coding
            ProviderType.GROQ,        # Fast inference
            ProviderType.OLLAMA,      # Local fallback
            ProviderType.OPENAI       # Widely supported
        ]
        
        for provider_type in provider_order:
            try:
                provider = cls.create_provider(provider_type)
                if provider.is_available():
                    return provider
            except Exception:
                continue
        
        raise RuntimeError("No LLM provider is available. Please configure at least one provider.")
    
    @classmethod
    def create_cost_optimized_provider(cls) -> LLMProvider:
        """Create most cost-effective available provider."""
        cost_order = cls.get_providers_by_tier("cost_effective")
        
        for provider_type in cost_order:
            try:
                provider = cls.create_provider(provider_type)
                if provider.is_available():
                    return provider
            except Exception:
                continue
        
        return cls.create_default_provider()


class MultiProviderLLM:
    """Multi-provider LLM client with intelligent routing and fallback support."""
    
    def __init__(
        self, 
        primary_provider: Optional[Union[LLMProvider, ProviderType]] = None,
        fallback_providers: Optional[List[Union[LLMProvider, ProviderType]]] = None,
        routing_strategy: str = "default"  # "default", "cost_optimized", "performance", "coding"
    ):
        self.providers = []
        self.routing_strategy = routing_strategy
        
        if primary_provider is None:
            if routing_strategy == "cost_optimized":
                self.providers.append(LLMProviderFactory.create_cost_optimized_provider())
            elif routing_strategy in LLMProviderFactory.PERFORMANCE_TIERS:
                tier_providers = LLMProviderFactory.get_providers_by_tier(routing_strategy)
                for provider_type in tier_providers:
                    try:
                        provider = LLMProviderFactory.create_provider(provider_type)
                        if provider.is_available():
                            self.providers.append(provider)
                            break
                    except Exception:
                        continue
                if not self.providers:
                    self.providers.append(LLMProviderFactory.create_default_provider())
            else:
                self.providers.append(LLMProviderFactory.create_default_provider())
        elif isinstance(primary_provider, ProviderType):
            self.providers.append(LLMProviderFactory.create_provider(primary_provider))
        else:
            self.providers.append(primary_provider)
        
        # Add fallback providers
        if fallback_providers:
            for provider in fallback_providers:
                if isinstance(provider, ProviderType):
                    self.providers.append(LLMProviderFactory.create_provider(provider))
                else:
                    self.providers.append(provider)
        else:
            # Add automatic fallbacks based on strategy
            self._add_automatic_fallbacks()
    
    def _add_automatic_fallbacks(self):
        """Add automatic fallback providers based on routing strategy."""
        if self.routing_strategy == "cost_optimized":
            fallback_order = [ProviderType.DEEPSEEK, ProviderType.GROQ, ProviderType.OLLAMA]
        elif self.routing_strategy == "coding":
            fallback_order = [ProviderType.ANTHROPIC, ProviderType.DEEPSEEK, ProviderType.OPENAI]
        elif self.routing_strategy == "fast_inference":
            fallback_order = [ProviderType.GROQ, ProviderType.FIREWORKS, ProviderType.TOGETHER]
        else:
            fallback_order = [ProviderType.DEEPSEEK, ProviderType.ANTHROPIC, ProviderType.GROQ]
        
        current_providers = {p.get_provider_type() for p in self.providers}
        
        for provider_type in fallback_order:
            if provider_type not in current_providers:
                try:
                    provider = LLMProviderFactory.create_provider(provider_type)
                    if provider.is_available():
                        self.providers.append(provider)
                        current_providers.add(provider_type)
                        if len(self.providers) >= 3:  # Limit fallback chain
                            break
                except Exception:
                    continue
    
    def generate_content(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> GenerateResponse:
        """Generate content with intelligent routing and fallback support."""
        last_error = None
        
        for provider in self.providers:
            try:
                if not provider.is_available():
                    continue
                
                response = provider.generate_content(
                    prompt=prompt,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
                response.provider = provider.get_provider_type()
                return response
                
            except Exception as e:
                last_error = e
                continue
        
        raise RuntimeError(f"All providers failed. Last error: {last_error}")
    
    def get_primary_provider(self) -> LLMProvider:
        """Get the primary provider."""
        return self.providers[0] if self.providers else None
    
    def get_available_providers(self) -> List[ProviderType]:
        """Get list of available providers in the chain."""
        return [p.get_provider_type() for p in self.providers if p.is_available()]