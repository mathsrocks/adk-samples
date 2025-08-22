"""Environment Variable Strategy for Multi-Provider LLM Support."""

import os
import json
import logging
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from enum import Enum
from dataclasses import dataclass, field

from .llm_providers import ProviderType


class EnvironmentStatus(Enum):
    """Status of environment variable configuration."""
    CONFIGURED = "configured"
    MISSING = "missing"
    INVALID = "invalid"
    PARTIALLY_CONFIGURED = "partially_configured"


@dataclass
class ProviderEnvironmentSpec:
    """Specification for a provider's environment variables."""
    provider_type: ProviderType
    primary_api_key: str  # Primary API key env var
    secondary_keys: List[str] = field(default_factory=list)  # Alternative API key env vars
    config_vars: List[str] = field(default_factory=list)  # Additional config env vars
    auth_methods: List[str] = field(default_factory=list)  # Supported auth methods
    required_vars: Set[str] = field(default_factory=set)  # Variables that must be set
    optional_vars: Set[str] = field(default_factory=set)  # Variables that are optional
    validation_func: Optional[callable] = None  # Custom validation function


@dataclass
class EnvironmentDiagnostic:
    """Diagnostic information about environment configuration."""
    provider: str
    status: EnvironmentStatus
    available_vars: Dict[str, str]  # var_name -> value (masked for secrets)
    missing_vars: List[str]
    issues: List[str]
    recommendations: List[str]
    auth_method: Optional[str] = None


class EnvironmentManager:
    """Manager for provider-specific environment variables with auto-detection and fallback."""
    
    # Comprehensive provider environment specifications
    PROVIDER_SPECS = {
        ProviderType.GOOGLE: ProviderEnvironmentSpec(
            provider_type=ProviderType.GOOGLE,
            primary_api_key="GOOGLE_APPLICATION_CREDENTIALS",
            secondary_keys=["GOOGLE_API_KEY"],
            config_vars=[
                "GOOGLE_GENAI_USE_VERTEXAI",
                "GOOGLE_CLOUD_PROJECT", 
                "GOOGLE_CLOUD_LOCATION",
                "GOOGLE_CLOUD_STORAGE_BUCKET"
            ],
            auth_methods=["service_account", "api_key", "adc"],
            required_vars={"GOOGLE_CLOUD_PROJECT"} if os.getenv("GOOGLE_GENAI_USE_VERTEXAI") else set(),
            optional_vars={"GOOGLE_CLOUD_LOCATION", "GOOGLE_CLOUD_STORAGE_BUCKET"}
        ),
        
        ProviderType.OPENAI: ProviderEnvironmentSpec(
            provider_type=ProviderType.OPENAI,
            primary_api_key="OPENAI_API_KEY",
            secondary_keys=["OPENAI_TOKEN"],
            config_vars=[
                "OPENAI_BASE_URL",
                "OPENAI_ORGANIZATION",
                "OPENAI_PROJECT"
            ],
            auth_methods=["api_key"],
            required_vars={"OPENAI_API_KEY"},
            optional_vars={"OPENAI_BASE_URL", "OPENAI_ORGANIZATION", "OPENAI_PROJECT"}
        ),
        
        ProviderType.ANTHROPIC: ProviderEnvironmentSpec(
            provider_type=ProviderType.ANTHROPIC,
            primary_api_key="ANTHROPIC_API_KEY",
            secondary_keys=["CLAUDE_API_KEY"],
            config_vars=[
                "ANTHROPIC_BASE_URL",
                "ANTHROPIC_MAX_RETRIES",
                "ANTHROPIC_TIMEOUT"
            ],
            auth_methods=["api_key"],
            required_vars={"ANTHROPIC_API_KEY"},
            optional_vars={"ANTHROPIC_BASE_URL", "ANTHROPIC_MAX_RETRIES", "ANTHROPIC_TIMEOUT"}
        ),
        
        ProviderType.DEEPSEEK: ProviderEnvironmentSpec(
            provider_type=ProviderType.DEEPSEEK,
            primary_api_key="DEEPSEEK_API_KEY",
            secondary_keys=["DEEPSEEK_TOKEN"],
            config_vars=[
                "DEEPSEEK_BASE_URL",
                "DEEPSEEK_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"DEEPSEEK_API_KEY"},
            optional_vars={"DEEPSEEK_BASE_URL", "DEEPSEEK_MAX_RETRIES"}
        ),
        
        ProviderType.GROQ: ProviderEnvironmentSpec(
            provider_type=ProviderType.GROQ,
            primary_api_key="GROQ_API_KEY",
            secondary_keys=["GROQ_TOKEN"],
            config_vars=[
                "GROQ_BASE_URL",
                "GROQ_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"GROQ_API_KEY"},
            optional_vars={"GROQ_BASE_URL", "GROQ_MAX_RETRIES"}
        ),
        
        ProviderType.TOGETHER: ProviderEnvironmentSpec(
            provider_type=ProviderType.TOGETHER,
            primary_api_key="TOGETHER_API_KEY",
            secondary_keys=["TOGETHER_TOKEN"],
            config_vars=[
                "TOGETHER_BASE_URL",
                "TOGETHER_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"TOGETHER_API_KEY"},
            optional_vars={"TOGETHER_BASE_URL", "TOGETHER_MAX_RETRIES"}
        ),
        
        ProviderType.MISTRAL: ProviderEnvironmentSpec(
            provider_type=ProviderType.MISTRAL,
            primary_api_key="MISTRAL_API_KEY",
            secondary_keys=["MISTRAL_TOKEN"],
            config_vars=[
                "MISTRAL_BASE_URL",
                "MISTRAL_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"MISTRAL_API_KEY"},
            optional_vars={"MISTRAL_BASE_URL", "MISTRAL_MAX_RETRIES"}
        ),
        
        ProviderType.COHERE: ProviderEnvironmentSpec(
            provider_type=ProviderType.COHERE,
            primary_api_key="COHERE_API_KEY",
            secondary_keys=["COHERE_TOKEN"],
            config_vars=[
                "COHERE_BASE_URL",
                "COHERE_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"COHERE_API_KEY"},
            optional_vars={"COHERE_BASE_URL", "COHERE_MAX_RETRIES"}
        ),
        
        ProviderType.HUGGINGFACE: ProviderEnvironmentSpec(
            provider_type=ProviderType.HUGGINGFACE,
            primary_api_key="HUGGINGFACE_API_KEY",
            secondary_keys=["HF_TOKEN", "HUGGINGFACE_TOKEN"],
            config_vars=[
                "HUGGINGFACE_HUB_CACHE",
                "HUGGINGFACE_BASE_URL"
            ],
            auth_methods=["api_key", "none"],
            required_vars=set(),  # HuggingFace can work without API key (rate limited)
            optional_vars={"HUGGINGFACE_API_KEY", "HF_TOKEN", "HUGGINGFACE_HUB_CACHE"}
        ),
        
        ProviderType.OLLAMA: ProviderEnvironmentSpec(
            provider_type=ProviderType.OLLAMA,
            primary_api_key="",  # No API key needed
            secondary_keys=[],
            config_vars=[
                "OLLAMA_HOST",
                "OLLAMA_PORT", 
                "OLLAMA_KEEP_ALIVE",
                "OLLAMA_NUM_PARALLEL",
                "OLLAMA_MAX_LOADED_MODELS"
            ],
            auth_methods=["none"],
            required_vars=set(),
            optional_vars={"OLLAMA_HOST", "OLLAMA_PORT", "OLLAMA_KEEP_ALIVE"}
        ),
        
        ProviderType.AZURE_OPENAI: ProviderEnvironmentSpec(
            provider_type=ProviderType.AZURE_OPENAI,
            primary_api_key="AZURE_OPENAI_API_KEY",
            secondary_keys=["AZURE_OPENAI_KEY"],
            config_vars=[
                "AZURE_OPENAI_ENDPOINT",
                "AZURE_OPENAI_DEPLOYMENT_NAME",
                "AZURE_OPENAI_API_VERSION"
            ],
            auth_methods=["api_key", "azure_ad"],
            required_vars={"AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"},
            optional_vars={"AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION"}
        ),
        
        ProviderType.AWS_BEDROCK: ProviderEnvironmentSpec(
            provider_type=ProviderType.AWS_BEDROCK,
            primary_api_key="AWS_ACCESS_KEY_ID",
            secondary_keys=[],
            config_vars=[
                "AWS_SECRET_ACCESS_KEY",
                "AWS_DEFAULT_REGION",
                "AWS_PROFILE",
                "AWS_SESSION_TOKEN"
            ],
            auth_methods=["access_keys", "iam_role", "profile"],
            required_vars={"AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"},
            optional_vars={"AWS_DEFAULT_REGION", "AWS_PROFILE", "AWS_SESSION_TOKEN"}
        ),
        
        ProviderType.PERPLEXITY: ProviderEnvironmentSpec(
            provider_type=ProviderType.PERPLEXITY,
            primary_api_key="PERPLEXITY_API_KEY",
            secondary_keys=["PERPLEXITY_TOKEN"],
            config_vars=[
                "PERPLEXITY_BASE_URL",
                "PERPLEXITY_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"PERPLEXITY_API_KEY"},
            optional_vars={"PERPLEXITY_BASE_URL", "PERPLEXITY_MAX_RETRIES"}
        ),
        
        ProviderType.FIREWORKS: ProviderEnvironmentSpec(
            provider_type=ProviderType.FIREWORKS,
            primary_api_key="FIREWORKS_API_KEY",
            secondary_keys=["FIREWORKS_TOKEN"],
            config_vars=[
                "FIREWORKS_BASE_URL",
                "FIREWORKS_MAX_RETRIES"
            ],
            auth_methods=["api_key"],
            required_vars={"FIREWORKS_API_KEY"},
            optional_vars={"FIREWORKS_BASE_URL", "FIREWORKS_MAX_RETRIES"}
        ),
        
        ProviderType.REPLICATE: ProviderEnvironmentSpec(
            provider_type=ProviderType.REPLICATE,
            primary_api_key="REPLICATE_API_TOKEN",
            secondary_keys=["REPLICATE_API_KEY"],
            config_vars=[
                "REPLICATE_BASE_URL",
                "REPLICATE_MAX_RETRIES"
            ],
            auth_methods=["api_token"],
            required_vars={"REPLICATE_API_TOKEN"},
            optional_vars={"REPLICATE_BASE_URL", "REPLICATE_MAX_RETRIES"}
        )
    }
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize the environment manager."""
        self.logger = logger or logging.getLogger(__name__)
        self._env_cache = {}  # Cache for environment variables
        self._diagnostics_cache = {}  # Cache for diagnostics
        
    def detect_available_providers(self, force_refresh: bool = False) -> List[ProviderType]:
        """Auto-detect available providers based on environment variables."""
        if not force_refresh and hasattr(self, '_available_providers'):
            return self._available_providers
        
        available_providers = []
        
        for provider_type, spec in self.PROVIDER_SPECS.items():
            if self._is_provider_configured(provider_type, spec):
                available_providers.append(provider_type)
        
        # Sort by preference (Google first for backward compatibility, then others)
        provider_priority = {
            ProviderType.GOOGLE: 0,
            ProviderType.DEEPSEEK: 1,  # Cost effective
            ProviderType.ANTHROPIC: 2,  # Good for coding
            ProviderType.OPENAI: 3,  # Widely supported
            ProviderType.GROQ: 4,  # Fast inference
            ProviderType.OLLAMA: 5,  # Local
        }
        
        available_providers.sort(key=lambda p: provider_priority.get(p, 999))
        self._available_providers = available_providers
        
        self.logger.info(f"Detected {len(available_providers)} available providers: {[p.value for p in available_providers]}")
        return available_providers
    
    def _is_provider_configured(self, provider_type: ProviderType, spec: ProviderEnvironmentSpec) -> bool:
        """Check if a provider is properly configured."""
        try:
            # Check primary API key
            if spec.primary_api_key:
                primary_key = os.getenv(spec.primary_api_key)
                if primary_key:
                    return self._validate_provider_config(provider_type, spec)
                
                # Check secondary keys
                for secondary_key in spec.secondary_keys:
                    if os.getenv(secondary_key):
                        return self._validate_provider_config(provider_type, spec)
                
                # If API key is required but not found
                if spec.required_vars and spec.primary_api_key in spec.required_vars:
                    return False
            
            # For providers that don't need API keys (like Ollama)
            if not spec.primary_api_key and "none" in spec.auth_methods:
                return self._validate_provider_config(provider_type, spec)
            
            # Check if all required vars are present
            for required_var in spec.required_vars:
                if not os.getenv(required_var):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error checking provider {provider_type.value}: {e}")
            return False
    
    def _validate_provider_config(self, provider_type: ProviderType, spec: ProviderEnvironmentSpec) -> bool:
        """Validate provider-specific configuration."""
        try:
            # Google-specific validation
            if provider_type == ProviderType.GOOGLE:
                if os.getenv("GOOGLE_GENAI_USE_VERTEXAI", "").lower() in ["true", "1", "yes"]:
                    # Vertex AI mode requires project and location
                    return bool(os.getenv("GOOGLE_CLOUD_PROJECT") and os.getenv("GOOGLE_CLOUD_LOCATION"))
                else:
                    # Direct API mode requires API key or ADC
                    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_APPLICATION_CREDENTIALS"))
            
            # Ollama-specific validation (check if service is accessible)
            if provider_type == ProviderType.OLLAMA:
                return self._check_ollama_availability()
            
            # Azure OpenAI specific validation
            if provider_type == ProviderType.AZURE_OPENAI:
                return bool(os.getenv("AZURE_OPENAI_API_KEY") and os.getenv("AZURE_OPENAI_ENDPOINT"))
            
            # AWS Bedrock specific validation
            if provider_type == ProviderType.AWS_BEDROCK:
                # Check for access keys or AWS profile
                has_keys = bool(os.getenv("AWS_ACCESS_KEY_ID") and os.getenv("AWS_SECRET_ACCESS_KEY"))
                has_profile = bool(os.getenv("AWS_PROFILE"))
                return has_keys or has_profile
            
            # For other providers, basic API key check is sufficient
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating {provider_type.value} config: {e}")
            return False
    
    def _check_ollama_availability(self) -> bool:
        """Check if Ollama service is available."""
        try:
            import requests
            ollama_host = os.getenv("OLLAMA_HOST", "localhost")
            ollama_port = os.getenv("OLLAMA_PORT", "11434")
            url = f"http://{ollama_host}:{ollama_port}/api/version"
            
            response = requests.get(url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False
    
    def get_provider_diagnostics(self, provider_type: ProviderType) -> EnvironmentDiagnostic:
        """Get detailed diagnostics for a provider's environment configuration."""
        if provider_type in self._diagnostics_cache:
            return self._diagnostics_cache[provider_type]
        
        spec = self.PROVIDER_SPECS.get(provider_type)
        if not spec:
            return EnvironmentDiagnostic(
                provider=provider_type.value,
                status=EnvironmentStatus.INVALID,
                available_vars={},
                missing_vars=[],
                issues=[f"Unknown provider: {provider_type.value}"],
                recommendations=[]
            )
        
        # Collect available variables (mask sensitive values)
        available_vars = {}
        missing_vars = []
        issues = []
        recommendations = []
        
        # Check primary API key
        if spec.primary_api_key:
            value = os.getenv(spec.primary_api_key)
            if value:
                available_vars[spec.primary_api_key] = self._mask_secret(value)
            else:
                missing_vars.append(spec.primary_api_key)
        
        # Check secondary keys
        for key in spec.secondary_keys:
            value = os.getenv(key)
            if value:
                available_vars[key] = self._mask_secret(value)
        
        # Check config variables
        for var in spec.config_vars:
            value = os.getenv(var)
            if value:
                available_vars[var] = value  # Config vars are usually not secrets
            elif var in spec.required_vars:
                missing_vars.append(var)
        
        # Determine status
        if missing_vars and any(var in spec.required_vars for var in missing_vars):
            status = EnvironmentStatus.MISSING
            issues.append(f"Required variables missing: {missing_vars}")
        elif missing_vars:
            status = EnvironmentStatus.PARTIALLY_CONFIGURED
            issues.append(f"Optional variables missing: {missing_vars}")
        elif self._is_provider_configured(provider_type, spec):
            status = EnvironmentStatus.CONFIGURED
        else:
            status = EnvironmentStatus.INVALID
            issues.append("Configuration validation failed")
        
        # Generate recommendations
        recommendations.extend(self._generate_recommendations(provider_type, spec, missing_vars))
        
        # Determine auth method
        auth_method = self._determine_auth_method(provider_type, spec, available_vars)
        
        diagnostic = EnvironmentDiagnostic(
            provider=provider_type.value,
            status=status,
            available_vars=available_vars,
            missing_vars=missing_vars,
            issues=issues,
            recommendations=recommendations,
            auth_method=auth_method
        )
        
        self._diagnostics_cache[provider_type] = diagnostic
        return diagnostic
    
    def _mask_secret(self, value: str) -> str:
        """Mask sensitive values for logging."""
        if len(value) <= 8:
            return "*" * len(value)
        return value[:4] + "*" * (len(value) - 8) + value[-4:]
    
    def _determine_auth_method(self, provider_type: ProviderType, spec: ProviderEnvironmentSpec, available_vars: Dict[str, str]) -> Optional[str]:
        """Determine the authentication method being used."""
        if provider_type == ProviderType.GOOGLE:
            if "GOOGLE_APPLICATION_CREDENTIALS" in available_vars:
                return "service_account"
            elif "GOOGLE_API_KEY" in available_vars:
                return "api_key"
            elif os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
                return "adc"
        
        if provider_type == ProviderType.AWS_BEDROCK:
            if "AWS_ACCESS_KEY_ID" in available_vars:
                return "access_keys"
            elif "AWS_PROFILE" in available_vars:
                return "profile"
        
        if spec.primary_api_key and spec.primary_api_key in available_vars:
            return "api_key"
        
        if "none" in spec.auth_methods:
            return "none"
        
        return None
    
    def _generate_recommendations(self, provider_type: ProviderType, spec: ProviderEnvironmentSpec, missing_vars: List[str]) -> List[str]:
        """Generate recommendations for provider configuration."""
        recommendations = []
        
        if provider_type == ProviderType.GOOGLE:
            if "GOOGLE_CLOUD_PROJECT" in missing_vars:
                recommendations.append("Set GOOGLE_CLOUD_PROJECT to your Google Cloud project ID")
            if "GOOGLE_APPLICATION_CREDENTIALS" in missing_vars and "GOOGLE_API_KEY" in missing_vars:
                recommendations.append("Set either GOOGLE_APPLICATION_CREDENTIALS (service account) or GOOGLE_API_KEY")
        
        elif provider_type == ProviderType.OPENAI:
            if "OPENAI_API_KEY" in missing_vars:
                recommendations.append("Get API key from https://platform.openai.com/api-keys and set OPENAI_API_KEY")
        
        elif provider_type == ProviderType.ANTHROPIC:
            if "ANTHROPIC_API_KEY" in missing_vars:
                recommendations.append("Get API key from https://console.anthropic.com and set ANTHROPIC_API_KEY")
        
        elif provider_type == ProviderType.DEEPSEEK:
            if "DEEPSEEK_API_KEY" in missing_vars:
                recommendations.append("Get API key from https://platform.deepseek.com and set DEEPSEEK_API_KEY")
        
        elif provider_type == ProviderType.OLLAMA:
            recommendations.append("Install Ollama from https://ollama.ai and ensure service is running")
            if missing_vars:
                recommendations.append("Set OLLAMA_HOST and OLLAMA_PORT if not using default localhost:11434")
        
        # Generic recommendations
        for var in missing_vars:
            if var in spec.required_vars:
                recommendations.append(f"Set required environment variable: {var}")
        
        return recommendations
    
    def auto_configure_provider(self, provider_type: ProviderType) -> bool:
        """Attempt to auto-configure a provider using available environment variables."""
        spec = self.PROVIDER_SPECS.get(provider_type)
        if not spec:
            return False
        
        try:
            # For Google, detect and set Vertex AI mode if Cloud Project is available
            if provider_type == ProviderType.GOOGLE:
                if os.getenv("GOOGLE_CLOUD_PROJECT") and not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
                    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "true"
                    self.logger.info("Auto-configured Google provider to use Vertex AI")
                    return True
            
            # For Ollama, set default host if not set
            if provider_type == ProviderType.OLLAMA:
                if not os.getenv("OLLAMA_HOST"):
                    os.environ["OLLAMA_HOST"] = "localhost"
                if not os.getenv("OLLAMA_PORT"):
                    os.environ["OLLAMA_PORT"] = "11434"
                self.logger.info("Auto-configured Ollama with default settings")
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Auto-configuration failed for {provider_type.value}: {e}")
            return False
    
    def get_fallback_chain(self, exclude_providers: Optional[List[ProviderType]] = None) -> List[ProviderType]:
        """Get ordered fallback chain of available providers."""
        available = self.detect_available_providers()
        
        if exclude_providers:
            available = [p for p in available if p not in exclude_providers]
        
        return available
    
    def get_best_provider_for_task(self, task_type: str, cost_priority: bool = False) -> Optional[ProviderType]:
        """Get the best provider for a specific task considering availability."""
        available = self.detect_available_providers()
        
        if not available:
            return None
        
        # Task-specific preferences
        task_preferences = {
            "coding": [ProviderType.ANTHROPIC, ProviderType.DEEPSEEK, ProviderType.OPENAI],
            "reasoning": [ProviderType.OPENAI, ProviderType.ANTHROPIC, ProviderType.GOOGLE],
            "cost_effective": [ProviderType.DEEPSEEK, ProviderType.GROQ, ProviderType.OLLAMA],
            "fast_inference": [ProviderType.GROQ, ProviderType.DEEPSEEK, ProviderType.GOOGLE],
            "local": [ProviderType.OLLAMA],
            "enterprise": [ProviderType.AZURE_OPENAI, ProviderType.AWS_BEDROCK, ProviderType.GOOGLE]
        }
        
        if cost_priority:
            preferences = task_preferences.get("cost_effective", [])
        else:
            preferences = task_preferences.get(task_type, [])
        
        # Find first available provider from preferences
        for provider in preferences:
            if provider in available:
                return provider
        
        # Fallback to first available provider
        return available[0] if available else None
    
    def export_environment_config(self, mask_secrets: bool = True) -> Dict[str, Any]:
        """Export current environment configuration for debugging."""
        config = {
            "available_providers": [p.value for p in self.detect_available_providers()],
            "provider_diagnostics": {},
            "environment_summary": {
                "total_providers": len(self.PROVIDER_SPECS),
                "configured_providers": len(self.detect_available_providers()),
                "unconfigured_providers": []
            }
        }
        
        for provider_type in self.PROVIDER_SPECS:
            diagnostic = self.get_provider_diagnostics(provider_type)
            config["provider_diagnostics"][provider_type.value] = {
                "status": diagnostic.status.value,
                "auth_method": diagnostic.auth_method,
                "available_vars": list(diagnostic.available_vars.keys()),
                "missing_vars": diagnostic.missing_vars,
                "issues": diagnostic.issues,
                "recommendations": diagnostic.recommendations
            }
            
            if diagnostic.status != EnvironmentStatus.CONFIGURED:
                config["environment_summary"]["unconfigured_providers"].append(provider_type.value)
        
        return config
    
    def clear_cache(self):
        """Clear internal caches."""
        self._env_cache.clear()
        self._diagnostics_cache.clear()
        if hasattr(self, '_available_providers'):
            delattr(self, '_available_providers')


# Global instance
_environment_manager = None


def get_environment_manager() -> EnvironmentManager:
    """Get the global environment manager instance."""
    global _environment_manager
    if _environment_manager is None:
        _environment_manager = EnvironmentManager()
    return _environment_manager


# Convenience functions
def detect_available_providers() -> List[ProviderType]:
    """Detect available providers based on environment."""
    return get_environment_manager().detect_available_providers()


def get_provider_diagnostics(provider: Union[ProviderType, str]) -> EnvironmentDiagnostic:
    """Get diagnostics for a provider."""
    if isinstance(provider, str):
        provider = ProviderType(provider)
    return get_environment_manager().get_provider_diagnostics(provider)


def get_best_provider_for_task(task_type: str, cost_priority: bool = False) -> Optional[ProviderType]:
    """Get the best available provider for a task."""
    return get_environment_manager().get_best_provider_for_task(task_type, cost_priority)