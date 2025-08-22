"""Bridge between enhanced configuration and provider abstraction layer."""

import os
from typing import Optional, Union, List, Dict, Any

from .config import CONFIG, ProviderConfig, ProviderStrategy
from .llm_providers import (
    LLMProviderFactory, 
    MultiProviderLLM, 
    ProviderType, 
    ModelConfig,
    LLMProvider
)
from .environment_manager import get_environment_manager, detect_available_providers


class ConfiguredProviderFactory:
    """Factory that creates providers based on the enhanced configuration."""
    
    @staticmethod
    def create_provider_from_config(provider_name: str) -> Optional[LLMProvider]:
        """Create a provider instance from configuration."""
        if provider_name not in CONFIG.provider_configs:
            return None
        
        provider_config = CONFIG.provider_configs[provider_name]
        
        if not provider_config.enabled:
            return None
        
        # Convert ProviderConfig to ModelConfig
        model_config = ModelConfig(
            model_name=provider_config.model_name,
            provider=ProviderType(provider_name),
            api_key_env_var=provider_config.api_key_env_var,
            base_url=provider_config.base_url,
            max_tokens=provider_config.max_tokens,
            temperature=provider_config.temperature,
            additional_params=provider_config.additional_params
        )
        
        try:
            return LLMProviderFactory.create_provider(ProviderType(provider_name), model_config)
        except Exception:
            return None
    
    @staticmethod
    def create_primary_provider() -> LLMProvider:
        """Create the primary provider based on configuration."""
        primary_provider_name = CONFIG.provider_type or "google"
        
        provider = ConfiguredProviderFactory.create_provider_from_config(primary_provider_name)
        if provider:
            return provider
        
        # Fallback to first available configured provider
        for provider_name in CONFIG.get_enabled_providers():
            provider = ConfiguredProviderFactory.create_provider_from_config(provider_name)
            if provider and provider.is_available():
                return provider
        
        # Ultimate fallback to default factory
        return LLMProviderFactory.create_default_provider()
    
    @staticmethod
    def create_multi_provider_llm() -> MultiProviderLLM:
        """Create a MultiProviderLLM based on configuration."""
        # Create primary provider
        primary_provider = ConfiguredProviderFactory.create_primary_provider()
        
        # Create fallback providers
        fallback_providers = []
        for provider_name in CONFIG.get_fallback_chain()[1:]:  # Skip first (primary)
            provider = ConfiguredProviderFactory.create_provider_from_config(provider_name)
            if provider:
                fallback_providers.append(provider)
        
        return MultiProviderLLM(
            primary_provider=primary_provider,
            fallback_providers=fallback_providers,
            routing_strategy=CONFIG.provider_strategy.value
        )


class BackwardCompatibilityAdapter:
    """Adapter to maintain backward compatibility with existing agent code."""
    
    @staticmethod
    def get_agent_model() -> str:
        """Get the agent model maintaining backward compatibility."""
        return CONFIG.agent_model
    
    @staticmethod
    def get_agent_model_for_task(task_type: str = "coding") -> str:
        """Get the appropriate model for a specific task type."""
        return CONFIG.get_model_for_task(task_type)
    
    @staticmethod
    def create_compatible_provider() -> LLMProvider:
        """Create a provider that maintains backward compatibility."""
        return ConfiguredProviderFactory.create_primary_provider()
    
    @staticmethod
    def get_generate_content_config() -> Dict[str, Any]:
        """Get generation config for backward compatibility."""
        primary_config = CONFIG.get_primary_provider_config()
        
        return {
            "temperature": primary_config.temperature or 0.01,
            "max_output_tokens": primary_config.max_tokens,
        }


class ConfigurationManager:
    """Manager for configuration updates and provider management."""
    
    @staticmethod
    def update_provider_strategy(strategy: Union[str, ProviderStrategy]):
        """Update the provider strategy."""
        CONFIG.update_provider_strategy(strategy)
    
    @staticmethod
    def enable_provider(provider_name: str, enabled: bool = True):
        """Enable or disable a provider."""
        if provider_name in CONFIG.provider_configs:
            CONFIG.provider_configs[provider_name].enabled = enabled
    
    @staticmethod
    def set_provider_model(provider_name: str, model_name: str):
        """Set the model for a specific provider."""
        if provider_name in CONFIG.provider_configs:
            CONFIG.provider_configs[provider_name].model_name = model_name
    
    @staticmethod
    def add_custom_provider(
        provider_name: str,
        model_name: str,
        api_key_env_var: Optional[str] = None,
        base_url: Optional[str] = None,
        priority: int = 50,
        **kwargs
    ):
        """Add a custom provider configuration."""
        CONFIG.provider_configs[provider_name] = ProviderConfig(
            provider_type=provider_name,
            model_name=model_name,
            api_key_env_var=api_key_env_var,
            base_url=base_url,
            priority=priority,
            additional_params=kwargs
        )
    
    @staticmethod
    def get_configuration_summary() -> Dict[str, Any]:
        """Get a summary of the current configuration."""
        return {
            "primary_provider": CONFIG.provider_type,
            "agent_model": CONFIG.agent_model,
            "provider_strategy": CONFIG.provider_strategy.value,
            "enabled_providers": CONFIG.get_enabled_providers(),
            "fallback_chain": CONFIG.get_fallback_chain(),
            "cost_tracking": CONFIG.enable_cost_tracking,
            "provider_fallback": CONFIG.enable_provider_fallback
        }
    
    @staticmethod
    def validate_configuration() -> List[str]:
        """Validate the current configuration and return any issues."""
        issues = []
        
        # Check if primary provider is configured
        if CONFIG.provider_type:
            if not CONFIG.is_provider_configured(CONFIG.provider_type):
                issues.append(f"Primary provider '{CONFIG.provider_type}' is not properly configured")
        
        # Check fallback chain
        fallback_chain = CONFIG.get_fallback_chain()
        if not fallback_chain:
            issues.append("No providers are available in the fallback chain")
        
        # Check for at least one configured provider
        enabled_providers = CONFIG.get_enabled_providers()
        if not enabled_providers:
            issues.append("No providers are enabled")
        
        configured_providers = [p for p in enabled_providers if CONFIG.is_provider_configured(p)]
        if not configured_providers:
            issues.append("No enabled providers are properly configured")
        
        return issues
    
    @staticmethod
    def auto_configure_from_environment():
        """Auto-configure providers based on available environment variables using EnvironmentManager."""
        env_manager = get_environment_manager()
        
        # Detect available providers
        available_providers = env_manager.detect_available_providers()
        configured_count = len(available_providers)
        
        # Enable available providers in configuration
        for provider_type in available_providers:
            provider_name = provider_type.value
            if provider_name in CONFIG.provider_configs:
                CONFIG.provider_configs[provider_name].enabled = True
        
        # Disable providers that are not available
        for provider_name, config in CONFIG.provider_configs.items():
            try:
                provider_type = ProviderType(provider_name)
                if provider_type not in available_providers:
                    config.enabled = False
            except ValueError:
                # Unknown provider type, disable it
                config.enabled = False
        
        # Auto-configure specific providers
        for provider_type in available_providers:
            env_manager.auto_configure_provider(provider_type)
        
        # Set primary provider if not already set
        if not CONFIG.provider_type and configured_count > 0:
            # Use the first available provider (they're sorted by priority)
            CONFIG.provider_type = available_providers[0].value
        
        # Update fallback chain with available providers
        fallback_chain = [p.value for p in env_manager.get_fallback_chain()]
        if fallback_chain:
            CONFIG.fallback_providers = fallback_chain
        
        return configured_count


# Convenience functions for backward compatibility
def get_configured_llm() -> MultiProviderLLM:
    """Get a configured MultiProviderLLM instance."""
    return ConfiguredProviderFactory.create_multi_provider_llm()


def get_agent_model() -> str:
    """Get the agent model (backward compatible)."""
    return BackwardCompatibilityAdapter.get_agent_model()


def create_agent_provider() -> LLMProvider:
    """Create an agent provider (backward compatible)."""
    return BackwardCompatibilityAdapter.create_compatible_provider()


# Environment-aware provider functions
def get_available_providers_from_env() -> List[str]:
    """Get list of providers available in the environment."""
    return [p.value for p in detect_available_providers()]


def get_environment_diagnostics() -> Dict[str, Any]:
    """Get comprehensive environment diagnostics."""
    env_manager = get_environment_manager()
    return env_manager.export_environment_config()


def create_environment_aware_llm() -> MultiProviderLLM:
    """Create a MultiProviderLLM that uses environment-detected providers."""
    env_manager = get_environment_manager()
    available_providers = env_manager.detect_available_providers()
    
    if not available_providers:
        raise RuntimeError("No LLM providers available in environment")
    
    # Create with detected providers
    primary_provider = available_providers[0]
    fallback_providers = available_providers[1:3]  # Limit fallbacks
    
    return MultiProviderLLM(
        primary_provider=primary_provider,
        fallback_providers=fallback_providers,
        routing_strategy="default"
    )


def get_best_provider_for_task(task_type: str, cost_priority: bool = False) -> Optional[str]:
    """Get the best available provider for a specific task."""
    env_manager = get_environment_manager()
    best_provider = env_manager.get_best_provider_for_task(task_type, cost_priority)
    return best_provider.value if best_provider else None


def validate_environment_setup() -> Dict[str, Any]:
    """Validate the current environment setup and return status."""
    env_manager = get_environment_manager()
    available = env_manager.detect_available_providers()
    
    validation = {
        "status": "healthy" if available else "no_providers",
        "available_providers": [p.value for p in available],
        "total_providers_configured": len(available),
        "issues": [],
        "recommendations": []
    }
    
    if not available:
        validation["issues"].append("No LLM providers are configured in the environment")
        validation["recommendations"].extend([
            "Set OPENAI_API_KEY for OpenAI GPT-4 access",
            "Set ANTHROPIC_API_KEY for Claude access",
            "Set DEEPSEEK_API_KEY for cost-effective inference",
            "Install Ollama for local inference",
            "Set GOOGLE_CLOUD_PROJECT and credentials for Google Gemini"
        ])
    
    # Check for common issues
    for provider_type in [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.ANTHROPIC]:
        diagnostic = env_manager.get_provider_diagnostics(provider_type)
        if diagnostic.issues:
            validation["issues"].extend([f"{provider_type.value}: {issue}" for issue in diagnostic.issues])
        if diagnostic.recommendations:
            validation["recommendations"].extend([f"{provider_type.value}: {rec}" for rec in diagnostic.recommendations])
    
    return validation


# Auto-configure on import if no providers are configured
if not any(CONFIG.is_provider_configured(p) for p in CONFIG.provider_configs.keys()):
    configured_count = ConfigurationManager.auto_configure_from_environment()
    if configured_count > 0:
        # Log successful auto-configuration
        available = get_available_providers_from_env()
        print(f"✅ Auto-configured {configured_count} LLM providers: {available}")
    else:
        # Log configuration guidance
        print("⚠️  No LLM providers detected. Set environment variables:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY, or install Ollama")