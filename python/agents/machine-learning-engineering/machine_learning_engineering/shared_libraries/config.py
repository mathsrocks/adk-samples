"""Configuration for Machine Learning Engineering Agent."""

import dataclasses
import os
from typing import Dict, List, Optional, Union, Any
from enum import Enum


class ProviderStrategy(Enum):
    """Strategies for provider selection and routing."""
    DEFAULT = "default"
    COST_OPTIMIZED = "cost_optimized"
    PERFORMANCE = "performance"
    CODING = "coding"
    REASONING = "reasoning"
    FAST_INFERENCE = "fast_inference"
    LONG_CONTEXT = "long_context"
    OPEN_SOURCE = "open_source"


@dataclasses.dataclass
class ProviderConfig:
    """Configuration for a specific LLM provider."""
    provider_type: str
    model_name: str
    api_key_env_var: Optional[str] = None
    base_url: Optional[str] = None
    enabled: bool = True
    priority: int = 0  # Higher number = higher priority
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    additional_params: Optional[Dict[str, Any]] = None


@dataclasses.dataclass
class DefaultConfig:
    """Default configuration."""
    data_dir: str = "./machine_learning_engineering/tasks/"  # the directory path where the machine learning tasks and their data are stored.
    task_name: str = "california-housing-prices"  # The name of the specific task to be loaded and processed.
    task_type: str = "Tabular Regression"  # The type of machine learning problem.
    lower: bool = True  # True if a lower value of the metric is better.
    workspace_dir: str = "./machine_learning_engineering/workspace/"  # Directory used for saving intermediate outputs, results, logs.
    agent_model: str = os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001")  # Name the LLM model to be used by the agent.
    task_description: str = ""  # The detailed description of the task.
    task_summary: str = ""  # The concise summary of the task.
    start_time: float = 0.0  # Timestamp indicating the start time of the task. Typically represented in seconds since the epoch.
    seed: int = 42  # The random seed value used to ensure reproducibility of experiments.
    exec_timeout: int = 600  # The maximum time in seconds allowed to complete the task.
    num_solutions: int = 2  # The number of different solutions to generate or attempt for the given task.
    num_model_candidates: int = 2  # The number of different model architectures or hyperparameter sets to consider as candidates.
    max_retry: int = 10  # The maximum number of times to retry a failed operation.
    max_debug_round: int = 5  # The maximum number of iterations or rounds allowed for the debugging step.
    max_rollback_round: int = 2  # The maximum number of times the system can rollback to a previous state, in case of errors or poor performance.
    inner_loop_round: int = 1  # The number of iterations or rounds to be executed within an inner loop of the system.
    outer_loop_round: int = 1  # The number of iterations or rounds to be executed within the outer loop, which might encompass multiple inner loops.
    ensemble_loop_round: int = 1  # The number of rounds or iterations dedicated to ensembling, combining multiple models or solutions.
    num_top_plans: int = 2  # The number of highest-scoring plans or strategies to select or retain.
    use_data_leakage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for data leakage in the machine learning pipeline.
    use_data_usage_checker: bool = False  # Enable (`True`) or disable (`False`) a check for how data is being used, potentially for compliance or best practices.
    
    # =============================================================================
    # Multi-Provider LLM Configuration
    # =============================================================================
    
    # Primary provider configuration (maintains backward compatibility)
    provider_type: Optional[str] = None  # Primary LLM provider type (e.g., "google", "openai", "anthropic")
    provider_strategy: ProviderStrategy = ProviderStrategy.DEFAULT  # Strategy for provider selection and routing
    
    # API key mappings for different providers
    api_key_mapping: Dict[str, str] = dataclasses.field(default_factory=lambda: {
        "google": "GOOGLE_APPLICATION_CREDENTIALS",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY", 
        "deepseek": "DEEPSEEK_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "cohere": "COHERE_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "replicate": "REPLICATE_API_TOKEN",
        "perplexity": "PERPLEXITY_API_KEY",
        "fireworks": "FIREWORKS_API_KEY",
        "azure_openai": "AZURE_OPENAI_API_KEY",
        "aws_bedrock": "AWS_ACCESS_KEY_ID"
    })
    
    # Model mappings for different providers and use cases
    model_mapping: Dict[str, Dict[str, str]] = dataclasses.field(default_factory=lambda: {
        "google": {
            "default": "gemini-2.0-flash-001",
            "coding": "gemini-2.0-flash-001", 
            "reasoning": "gemini-2.0-flash-001",
            "long_context": "gemini-1.5-pro-002"
        },
        "openai": {
            "default": "gpt-4o",
            "coding": "gpt-4o",
            "reasoning": "o1-preview",
            "long_context": "gpt-4o-128k"
        },
        "anthropic": {
            "default": "claude-3-5-sonnet-20241022",
            "coding": "claude-3-5-sonnet-20241022",
            "reasoning": "claude-3-5-sonnet-20241022",
            "long_context": "claude-3-5-sonnet-20241022"
        },
        "deepseek": {
            "default": "deepseek-chat",
            "coding": "deepseek-coder",
            "reasoning": "deepseek-reasoner"
        },
        "groq": {
            "default": "llama-3.3-70b-versatile",
            "coding": "llama-3.3-70b-versatile",
            "fast_inference": "llama-3.3-70b-versatile"
        },
        "ollama": {
            "default": "llama3.2:latest",
            "coding": "codellama:latest",
            "open_source": "llama3.2:latest"
        }
    })
    
    # Provider-specific configurations
    provider_configs: Dict[str, ProviderConfig] = dataclasses.field(default_factory=lambda: {
        "google": ProviderConfig(
            provider_type="google",
            model_name=os.environ.get("ROOT_AGENT_MODEL", "gemini-2.0-flash-001"),
            api_key_env_var="GOOGLE_APPLICATION_CREDENTIALS",
            enabled=True,
            priority=100,  # Highest priority for backward compatibility
            temperature=0.01
        ),
        "openai": ProviderConfig(
            provider_type="openai", 
            model_name="gpt-4o",
            api_key_env_var="OPENAI_API_KEY",
            enabled=True,
            priority=80
        ),
        "anthropic": ProviderConfig(
            provider_type="anthropic",
            model_name="claude-3-5-sonnet-20241022", 
            api_key_env_var="ANTHROPIC_API_KEY",
            enabled=True,
            priority=85
        ),
        "deepseek": ProviderConfig(
            provider_type="deepseek",
            model_name="deepseek-chat",
            api_key_env_var="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            enabled=True,
            priority=90  # High priority for cost effectiveness
        ),
        "groq": ProviderConfig(
            provider_type="groq",
            model_name="llama-3.3-70b-versatile",
            api_key_env_var="GROQ_API_KEY",
            base_url="https://api.groq.com/openai/v1",
            enabled=True,
            priority=70
        ),
        "ollama": ProviderConfig(
            provider_type="ollama",
            model_name="llama3.2:latest", 
            base_url="http://localhost:11434",
            enabled=True,
            priority=60  # Lower priority as it's local
        )
    })
    
    # Fallback provider chain (ordered by preference)
    fallback_providers: List[str] = dataclasses.field(default_factory=lambda: [
        "google",    # Current default
        "deepseek",  # Cost effective
        "anthropic", # Good for coding
        "groq",      # Fast inference
        "openai",    # Widely supported
        "ollama"     # Local fallback
    ])
    
    # Provider routing settings
    enable_provider_fallback: bool = True  # Enable automatic fallback to other providers
    max_fallback_attempts: int = 3  # Maximum number of providers to try
    provider_timeout: int = 30  # Timeout in seconds for provider requests
    enable_cost_tracking: bool = True  # Track and log costs for different providers
    
    def __post_init__(self):
        """Post-initialization to handle backward compatibility and validation."""
        # Maintain backward compatibility with ROOT_AGENT_MODEL
        if self.provider_type is None:
            # Try to infer provider from ROOT_AGENT_MODEL or default to Google
            root_model = os.environ.get("ROOT_AGENT_MODEL", self.agent_model)
            
            # Update agent_model if ROOT_AGENT_MODEL is set
            if "ROOT_AGENT_MODEL" in os.environ:
                self.agent_model = root_model
            
            # Infer provider type from model name
            self.provider_type = self._infer_provider_from_model(root_model)
        
        # Update the Google provider config with the current agent_model
        if "google" in self.provider_configs:
            self.provider_configs["google"].model_name = self.agent_model
    
    def _infer_provider_from_model(self, model_name: str) -> str:
        """Infer provider type from model name for backward compatibility."""
        model_lower = model_name.lower()
        
        if any(x in model_lower for x in ["gemini", "palm", "gecko", "bison"]):
            return "google"
        elif any(x in model_lower for x in ["gpt", "o1-", "chatgpt"]):
            return "openai"
        elif any(x in model_lower for x in ["claude"]):
            return "anthropic"
        elif any(x in model_lower for x in ["deepseek"]):
            return "deepseek"
        elif any(x in model_lower for x in ["llama", "mixtral", "qwen"]):
            return "groq"  # Default for open-source models
        else:
            return "google"  # Default fallback
    
    def get_primary_provider_config(self) -> ProviderConfig:
        """Get the configuration for the primary provider."""
        if self.provider_type and self.provider_type in self.provider_configs:
            return self.provider_configs[self.provider_type]
        
        # Fallback to the first enabled provider with highest priority
        enabled_providers = [
            (name, config) for name, config in self.provider_configs.items()
            if config.enabled
        ]
        
        if enabled_providers:
            # Sort by priority (higher first)
            enabled_providers.sort(key=lambda x: x[1].priority, reverse=True)
            return enabled_providers[0][1]
        
        raise RuntimeError("No enabled provider configurations found")
    
    def get_model_for_task(self, task_type: str = "default") -> str:
        """Get the appropriate model for a specific task type."""
        provider = self.provider_type or "google"
        
        if provider in self.model_mapping:
            task_models = self.model_mapping[provider]
            if task_type in task_models:
                return task_models[task_type]
            elif "default" in task_models:
                return task_models["default"]
        
        # Fallback to agent_model for backward compatibility
        return self.agent_model
    
    def get_enabled_providers(self) -> List[str]:
        """Get list of enabled provider names sorted by priority."""
        enabled = [
            (name, config.priority) for name, config in self.provider_configs.items()
            if config.enabled
        ]
        enabled.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in enabled]
    
    def get_api_key_for_provider(self, provider: str) -> Optional[str]:
        """Get the API key environment variable name for a provider."""
        return self.api_key_mapping.get(provider)
    
    def is_provider_configured(self, provider: str) -> bool:
        """Check if a provider is properly configured."""
        if provider not in self.provider_configs:
            return False
        
        config = self.provider_configs[provider]
        if not config.enabled:
            return False
        
        # Check if API key is available (if required)
        if config.api_key_env_var:
            return os.getenv(config.api_key_env_var) is not None
        
        return True
    
    def get_fallback_chain(self) -> List[str]:
        """Get the ordered fallback provider chain."""
        if self.enable_provider_fallback:
            # Filter to only enabled and configured providers
            return [
                provider for provider in self.fallback_providers
                if provider in self.provider_configs 
                and self.provider_configs[provider].enabled
                and self.is_provider_configured(provider)
            ][:self.max_fallback_attempts]
        else:
            # Return only the primary provider if fallback is disabled
            primary = self.provider_type or "google"
            return [primary] if self.is_provider_configured(primary) else []
    
    def update_provider_strategy(self, strategy: Union[str, ProviderStrategy]):
        """Update the provider strategy and reorder fallback chain accordingly."""
        if isinstance(strategy, str):
            strategy = ProviderStrategy(strategy)
        
        self.provider_strategy = strategy
        
        # Reorder fallback providers based on strategy
        strategy_priorities = {
            ProviderStrategy.COST_OPTIMIZED: ["deepseek", "groq", "ollama", "google", "anthropic", "openai"],
            ProviderStrategy.CODING: ["anthropic", "deepseek", "openai", "google", "groq", "ollama"],
            ProviderStrategy.REASONING: ["openai", "anthropic", "google", "deepseek", "groq", "ollama"],
            ProviderStrategy.FAST_INFERENCE: ["groq", "deepseek", "google", "openai", "anthropic", "ollama"],
            ProviderStrategy.OPEN_SOURCE: ["ollama", "groq", "deepseek", "anthropic", "openai", "google"],
            ProviderStrategy.LONG_CONTEXT: ["google", "anthropic", "openai", "groq", "deepseek", "ollama"]
        }
        
        if strategy in strategy_priorities:
            self.fallback_providers = strategy_priorities[strategy]
    
    def to_provider_abstraction_config(self) -> Dict[str, Any]:
        """Convert to configuration format for the provider abstraction layer."""
        return {
            "primary_provider": self.provider_type,
            "fallback_providers": self.get_fallback_chain(),
            "routing_strategy": self.provider_strategy.value,
            "provider_configs": {
                name: {
                    "model_name": config.model_name,
                    "api_key_env_var": config.api_key_env_var,
                    "base_url": config.base_url,
                    "temperature": config.temperature,
                    "max_tokens": config.max_tokens,
                    "additional_params": config.additional_params or {}
                }
                for name, config in self.provider_configs.items()
                if config.enabled
            },
            "timeout": self.provider_timeout,
            "enable_cost_tracking": self.enable_cost_tracking
        }


CONFIG = DefaultConfig()
