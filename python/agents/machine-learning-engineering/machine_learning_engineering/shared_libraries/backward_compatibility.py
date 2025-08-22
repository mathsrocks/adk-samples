#!/usr/bin/env python3
"""Backward compatibility layer for maintaining Google ADK integration and Vertex AI deployment."""

import os
import sys
import json
import warnings
from typing import Dict, Any, Optional, List, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from google.adk import agents
from google.genai import types

from .config import CONFIG


class CompatibilityMode(Enum):
    """Compatibility modes for different deployment scenarios."""
    STRICT_ADK = "strict_adk"           # Strict Google ADK compatibility (original behavior)
    ENHANCED_ADK = "enhanced_adk"       # Enhanced with multi-provider but ADK-compatible
    GRADUAL_MIGRATION = "gradual_migration"  # Gradual transition to multi-provider
    FULL_MULTI_PROVIDER = "full_multi_provider"  # Full multi-provider capabilities


@dataclass
class LegacyConfiguration:
    """Legacy configuration structure for backward compatibility."""
    root_agent_model: Optional[str] = None
    google_genai_use_vertexai: Optional[str] = None
    google_cloud_project: Optional[str] = None
    google_cloud_location: Optional[str] = None
    google_cloud_storage_bucket: Optional[str] = None
    
    # Legacy temperature and configuration
    temperature: float = 0.01
    max_output_tokens: Optional[int] = None
    
    # Legacy task configuration
    task_name: Optional[str] = None
    workspace_dir: str = "machine_learning_engineering/workspace"
    
    # Migration tracking
    migrated_to_multi_provider: bool = False
    compatibility_mode: CompatibilityMode = CompatibilityMode.STRICT_ADK
    
    @classmethod
    def from_environment(cls) -> 'LegacyConfiguration':
        """Create legacy configuration from environment variables."""
        return cls(
            root_agent_model=os.getenv("ROOT_AGENT_MODEL"),
            google_genai_use_vertexai=os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
            google_cloud_project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            google_cloud_location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            google_cloud_storage_bucket=os.getenv("GOOGLE_CLOUD_STORAGE_BUCKET"),
            task_name=os.getenv("TASK_NAME", "default_task")
        )
    
    def is_valid_legacy_setup(self) -> bool:
        """Check if this is a valid legacy Google-only setup."""
        return (
            self.root_agent_model is not None and
            self.google_genai_use_vertexai == "true" and
            self.google_cloud_project is not None and
            self.google_cloud_location is not None
        )
    
    def detect_compatibility_mode(self) -> CompatibilityMode:
        """Detect the appropriate compatibility mode based on environment."""
        
        # Check for multi-provider environment variables
        multi_provider_vars = [
            "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
            "GROQ_API_KEY", "COHERE_API_KEY"
        ]
        
        has_multi_provider = any(os.getenv(var) for var in multi_provider_vars)
        has_google_setup = self.is_valid_legacy_setup()
        
        # Detect provider strategy setting
        provider_strategy = os.getenv("PROVIDER_STRATEGY", "").lower()
        
        if not has_google_setup and not has_multi_provider:
            # No proper setup detected
            return CompatibilityMode.STRICT_ADK
        elif has_google_setup and not has_multi_provider:
            # Pure Google setup
            return CompatibilityMode.STRICT_ADK
        elif has_google_setup and has_multi_provider and provider_strategy in ["gradual", "migration"]:
            # Gradual migration mode
            return CompatibilityMode.GRADUAL_MIGRATION
        elif has_google_setup and has_multi_provider:
            # Enhanced ADK with multi-provider support
            return CompatibilityMode.ENHANCED_ADK
        elif has_multi_provider:
            # Full multi-provider mode
            return CompatibilityMode.FULL_MULTI_PROVIDER
        else:
            # Default to strict ADK
            return CompatibilityMode.STRICT_ADK


@dataclass
class MigrationPlan:
    """Migration plan for upgrading from legacy to multi-provider setup."""
    current_mode: CompatibilityMode
    target_mode: CompatibilityMode
    migration_steps: List[str] = field(default_factory=list)
    required_changes: Dict[str, Any] = field(default_factory=dict)
    risk_level: str = "low"  # low, medium, high
    estimated_effort: str = "minimal"  # minimal, moderate, significant
    rollback_plan: List[str] = field(default_factory=list)


class BackwardCompatibilityManager:
    """Manager for maintaining backward compatibility with existing ADK deployments."""
    
    def __init__(self):
        """Initialize the backward compatibility manager."""
        self.legacy_config = LegacyConfiguration.from_environment()
        self.compatibility_mode = self.legacy_config.detect_compatibility_mode()
        self._original_agents_module = None
        self._compatibility_warnings_shown = set()
    
    def ensure_adk_compatibility(self) -> bool:
        """Ensure compatibility with Google ADK integration."""
        
        if self.compatibility_mode == CompatibilityMode.STRICT_ADK:
            # Pure legacy mode - no changes needed
            return True
        
        # For enhanced modes, ensure ADK patterns still work
        try:
            # Verify that agents.Agent() still works with original parameters
            self._test_legacy_agent_creation()
            
            # Verify that GenerateContentConfig works as expected
            self._test_legacy_generate_config()
            
            # Verify that sub-agent patterns still work
            self._test_legacy_sub_agent_patterns()
            
            return True
            
        except Exception as e:
            self._show_compatibility_warning(
                "adk_compatibility",
                f"ADK compatibility check failed: {e}. Falling back to strict ADK mode."
            )
            self.compatibility_mode = CompatibilityMode.STRICT_ADK
            return False
    
    def ensure_vertex_ai_compatibility(self) -> bool:
        """Ensure compatibility with Vertex AI deployment pipeline."""
        
        if not self.legacy_config.is_valid_legacy_setup():
            self._show_compatibility_warning(
                "vertex_ai_setup",
                "Vertex AI environment variables not fully configured. Multi-provider features may not work in deployment."
            )
            return False
        
        # Check Vertex AI specific requirements
        vertex_requirements = [
            ("GOOGLE_GENAI_USE_VERTEXAI", "true"),
            ("GOOGLE_CLOUD_PROJECT", "not empty"),
            ("GOOGLE_CLOUD_LOCATION", "not empty")
        ]
        
        for var, requirement in vertex_requirements:
            value = os.getenv(var)
            if requirement == "true" and value != "true":
                self._show_compatibility_warning(
                    "vertex_ai_config",
                    f"{var} should be 'true' for Vertex AI deployment, got '{value}'"
                )
                return False
            elif requirement == "not empty" and not value:
                self._show_compatibility_warning(
                    "vertex_ai_config",
                    f"{var} is required for Vertex AI deployment but not set"
                )
                return False
        
        return True
    
    def create_legacy_compatible_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        model: Optional[str] = None,
        description: Optional[str] = None,
        global_instruction: Optional[Union[str, Callable]] = None,
        sub_agents: Optional[List[agents.Agent]] = None,
        generate_content_config: Optional[types.GenerateContentConfig] = None,
        **kwargs
    ) -> agents.Agent:
        """Create an agent with full backward compatibility."""
        
        # Use legacy model if not specified
        final_model = model or self.legacy_config.root_agent_model or CONFIG.get_model_for_task()
        
        # Create legacy-compatible generate content config if not provided
        if generate_content_config is None:
            generate_content_config = types.GenerateContentConfig(
                temperature=self.legacy_config.temperature,
                max_output_tokens=self.legacy_config.max_output_tokens
            )
        
        # For strict ADK mode, use original Agent constructor directly
        if self.compatibility_mode == CompatibilityMode.STRICT_ADK:
            return agents.Agent(
                model=final_model,
                name=name,
                instruction=instruction,
                description=description or f"Legacy compatible agent: {name}",
                global_instruction=global_instruction,
                sub_agents=sub_agents or [],
                generate_content_config=generate_content_config,
                **kwargs
            )
        
        # For enhanced modes, use the agent factory but ensure ADK compatibility
        try:
            from .agent_factory import get_agent_factory
            
            factory = get_agent_factory(provider="google")  # Force Google provider for compatibility
            
            return factory.create_agent(
                name=name,
                instruction=instruction,
                model=final_model,
                description=description,
                global_instruction=global_instruction,
                sub_agents=sub_agents,
                generate_content_config=generate_content_config,
                provider="google",  # Explicitly use Google for compatibility
                **kwargs
            )
            
        except Exception as e:
            self._show_compatibility_warning(
                "agent_factory_fallback",
                f"Agent factory failed, using legacy Agent constructor: {e}"
            )
            
            # Fallback to original Agent constructor
            return agents.Agent(
                model=final_model,
                name=name,
                instruction=instruction,
                description=description or f"Legacy fallback agent: {name}",
                global_instruction=global_instruction,
                sub_agents=sub_agents or [],
                generate_content_config=generate_content_config,
                **kwargs
            )
    
    def get_migration_plan(self, target_mode: Optional[CompatibilityMode] = None) -> MigrationPlan:
        """Get a migration plan for upgrading to multi-provider support."""
        
        if target_mode is None:
            target_mode = CompatibilityMode.ENHANCED_ADK
        
        plan = MigrationPlan(
            current_mode=self.compatibility_mode,
            target_mode=target_mode
        )
        
        # Generate migration steps based on current and target modes
        if self.compatibility_mode == CompatibilityMode.STRICT_ADK:
            if target_mode == CompatibilityMode.ENHANCED_ADK:
                plan.migration_steps = [
                    "1. Install additional provider dependencies: poetry install --extras essential",
                    "2. Set environment variables for desired providers (optional)",
                    "3. Update agent.py to use get_agent_factory() (already done)",
                    "4. Test existing functionality to ensure no regressions",
                    "5. Gradually enable multi-provider features as needed"
                ]
                plan.risk_level = "low"
                plan.estimated_effort = "minimal"
                
            elif target_mode == CompatibilityMode.FULL_MULTI_PROVIDER:
                plan.migration_steps = [
                    "1. Install all provider dependencies: poetry install --extras professional",
                    "2. Set environment variables for all desired providers",
                    "3. Update configuration to specify provider preferences",
                    "4. Update agent creation to use task-specific optimization",
                    "5. Enable cost optimization features",
                    "6. Test thoroughly with fallback scenarios",
                    "7. Update deployment configurations if needed"
                ]
                plan.risk_level = "medium"
                plan.estimated_effort = "moderate"
        
        elif self.compatibility_mode == CompatibilityMode.ENHANCED_ADK:
            if target_mode == CompatibilityMode.FULL_MULTI_PROVIDER:
                plan.migration_steps = [
                    "1. Install additional provider dependencies as needed",
                    "2. Update agent creation to use cost optimization",
                    "3. Configure provider preferences and fallback chains",
                    "4. Enable performance routing features",
                    "5. Test cost optimization scenarios"
                ]
                plan.risk_level = "low"
                plan.estimated_effort = "minimal"
        
        # Add required environment variables
        plan.required_changes = {
            "environment_variables": {
                "OPENAI_API_KEY": "Optional - for OpenAI provider",
                "ANTHROPIC_API_KEY": "Optional - for Claude provider", 
                "DEEPSEEK_API_KEY": "Optional - for DeepSeek provider",
                "PROVIDER_STRATEGY": "Optional - provider selection strategy"
            },
            "dependencies": {
                "essential": "poetry install --extras essential",
                "professional": "poetry install --extras professional",
                "all": "poetry install --extras all-providers"
            },
            "configuration_changes": {
                "agent_creation": "Use get_agent_factory() methods for enhanced features",
                "cost_optimization": "Use create_budget_conscious_agent() for cost savings",
                "performance_routing": "Use create_coding_optimized_agent() for better performance"
            }
        }
        
        # Rollback plan
        plan.rollback_plan = [
            "1. Remove multi-provider environment variables",
            "2. Ensure GOOGLE_* environment variables are properly set",
            "3. Set COMPATIBILITY_MODE=strict_adk if needed",
            "4. Restart services to pick up configuration changes",
            "5. Verify original functionality works as expected"
        ]
        
        return plan
    
    def create_migration_utilities(self) -> Dict[str, Any]:
        """Create utilities to help with migration."""
        
        utilities = {
            "configuration_validator": self._create_config_validator(),
            "environment_checker": self._create_environment_checker(),
            "compatibility_tester": self._create_compatibility_tester(),
            "migration_assistant": self._create_migration_assistant()
        }
        
        return utilities
    
    def _create_config_validator(self) -> Callable:
        """Create a configuration validator utility."""
        
        def validate_configuration() -> Dict[str, Any]:
            """Validate current configuration for compatibility."""
            
            results = {
                "legacy_config_valid": self.legacy_config.is_valid_legacy_setup(),
                "compatibility_mode": self.compatibility_mode.value,
                "adk_compatible": self.ensure_adk_compatibility(),
                "vertex_ai_compatible": self.ensure_vertex_ai_compatibility(),
                "recommendations": []
            }
            
            # Add specific recommendations
            if not results["legacy_config_valid"]:
                results["recommendations"].append(
                    "Set required Google Cloud environment variables for Vertex AI deployment"
                )
            
            if self.compatibility_mode == CompatibilityMode.STRICT_ADK:
                results["recommendations"].append(
                    "Consider upgrading to ENHANCED_ADK mode for additional provider options"
                )
            
            if not results["adk_compatible"]:
                results["recommendations"].append(
                    "Check ADK compatibility issues - may need to downgrade features"
                )
            
            return results
        
        return validate_configuration
    
    def _create_environment_checker(self) -> Callable:
        """Create an environment checker utility."""
        
        def check_environment() -> Dict[str, Any]:
            """Check environment for compatibility and migration readiness."""
            
            # Check required Google environment variables
            google_vars = {
                "GOOGLE_GENAI_USE_VERTEXAI": os.getenv("GOOGLE_GENAI_USE_VERTEXAI"),
                "GOOGLE_CLOUD_PROJECT": os.getenv("GOOGLE_CLOUD_PROJECT"),
                "GOOGLE_CLOUD_LOCATION": os.getenv("GOOGLE_CLOUD_LOCATION"),
                "ROOT_AGENT_MODEL": os.getenv("ROOT_AGENT_MODEL")
            }
            
            # Check optional multi-provider variables
            provider_vars = {
                "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
                "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
                "DEEPSEEK_API_KEY": bool(os.getenv("DEEPSEEK_API_KEY")),
                "GROQ_API_KEY": bool(os.getenv("GROQ_API_KEY"))
            }
            
            # Check installed dependencies
            installed_packages = self._check_installed_packages()
            
            return {
                "google_environment": google_vars,
                "provider_environment": provider_vars,
                "installed_packages": installed_packages,
                "compatibility_mode": self.compatibility_mode.value,
                "migration_ready": len([v for v in provider_vars.values() if v]) > 0
            }
        
        return check_environment
    
    def _create_compatibility_tester(self) -> Callable:
        """Create a compatibility tester utility."""
        
        def test_compatibility() -> Dict[str, Any]:
            """Test compatibility with existing patterns."""
            
            test_results = {
                "legacy_agent_creation": False,
                "generate_content_config": False,
                "sub_agent_patterns": False,
                "agent_factory_fallback": False,
                "error_messages": []
            }
            
            try:
                # Test legacy agent creation
                self._test_legacy_agent_creation()
                test_results["legacy_agent_creation"] = True
            except Exception as e:
                test_results["error_messages"].append(f"Legacy agent creation failed: {e}")
            
            try:
                # Test generate content config
                self._test_legacy_generate_config()
                test_results["generate_content_config"] = True
            except Exception as e:
                test_results["error_messages"].append(f"Generate content config failed: {e}")
            
            try:
                # Test sub-agent patterns
                self._test_legacy_sub_agent_patterns()
                test_results["sub_agent_patterns"] = True
            except Exception as e:
                test_results["error_messages"].append(f"Sub-agent patterns failed: {e}")
            
            try:
                # Test agent factory fallback
                from .agent_factory import get_agent_factory
                factory = get_agent_factory()
                test_agent = factory.create_agent(
                    name="test_agent",
                    instruction="test instruction",
                    provider="google"
                )
                test_results["agent_factory_fallback"] = True
            except Exception as e:
                test_results["error_messages"].append(f"Agent factory fallback failed: {e}")
            
            test_results["overall_compatibility"] = all([
                test_results["legacy_agent_creation"],
                test_results["generate_content_config"],
                test_results["sub_agent_patterns"]
            ])
            
            return test_results
        
        return test_compatibility
    
    def _create_migration_assistant(self) -> Callable:
        """Create a migration assistant utility."""
        
        def migration_assistant(target_mode: str = "enhanced_adk") -> Dict[str, Any]:
            """Assist with migration to multi-provider setup."""
            
            try:
                target_mode_enum = CompatibilityMode(target_mode)
            except ValueError:
                return {"error": f"Invalid target mode: {target_mode}"}
            
            plan = self.get_migration_plan(target_mode_enum)
            
            # Check prerequisites
            prerequisites_met = True
            prerequisite_checks = {}
            
            # Check if Google environment is properly set up
            google_setup = self.legacy_config.is_valid_legacy_setup()
            prerequisite_checks["google_setup"] = google_setup
            if not google_setup:
                prerequisites_met = False
            
            # Check if Poetry is available for dependency management
            poetry_available = self._check_poetry_available()
            prerequisite_checks["poetry_available"] = poetry_available
            if not poetry_available:
                prerequisites_met = False
            
            return {
                "migration_plan": {
                    "current_mode": plan.current_mode.value,
                    "target_mode": plan.target_mode.value,
                    "steps": plan.migration_steps,
                    "risk_level": plan.risk_level,
                    "estimated_effort": plan.estimated_effort,
                    "required_changes": plan.required_changes,
                    "rollback_plan": plan.rollback_plan
                },
                "prerequisites_met": prerequisites_met,
                "prerequisite_checks": prerequisite_checks,
                "next_steps": plan.migration_steps[:3] if prerequisites_met else [
                    "Fix prerequisite issues before proceeding with migration"
                ]
            }
        
        return migration_assistant
    
    def _test_legacy_agent_creation(self):
        """Test that legacy agent creation patterns still work."""
        
        # Test basic agent creation
        test_agent = agents.Agent(
            model=self.legacy_config.root_agent_model or "gemini-1.5-pro",
            name="test_legacy_agent",
            instruction="This is a test instruction",
            description="Test agent for backward compatibility"
        )
        
        # Verify agent was created successfully
        assert test_agent.name == "test_legacy_agent"
        assert test_agent.instruction == "This is a test instruction"
    
    def _test_legacy_generate_config(self):
        """Test that legacy GenerateContentConfig patterns still work."""
        
        # Test generate content config creation
        config = types.GenerateContentConfig(
            temperature=self.legacy_config.temperature,
            max_output_tokens=self.legacy_config.max_output_tokens
        )
        
        # Verify config was created successfully
        assert config.temperature == self.legacy_config.temperature
    
    def _test_legacy_sub_agent_patterns(self):
        """Test that legacy sub-agent patterns still work."""
        
        # Create a simple sub-agent
        sub_agent = agents.Agent(
            model=self.legacy_config.root_agent_model or "gemini-1.5-pro",
            name="test_sub_agent",
            instruction="Sub-agent test instruction"
        )
        
        # Create parent agent with sub-agent
        parent_agent = agents.Agent(
            model=self.legacy_config.root_agent_model or "gemini-1.5-pro",
            name="test_parent_agent",
            instruction="Parent agent test instruction",
            sub_agents=[sub_agent]
        )
        
        # Verify sub-agent relationship
        assert len(parent_agent.sub_agents) == 1
        assert parent_agent.sub_agents[0].name == "test_sub_agent"
    
    def _check_installed_packages(self) -> Dict[str, bool]:
        """Check which optional packages are installed."""
        
        packages_to_check = [
            "openai", "anthropic", "groq", "cohere", "mistralai",
            "google-cloud-aiplatform", "google-generativeai"
        ]
        
        installed = {}
        
        for package in packages_to_check:
            try:
                __import__(package.replace("-", "_"))
                installed[package] = True
            except ImportError:
                installed[package] = False
        
        return installed
    
    def _check_poetry_available(self) -> bool:
        """Check if Poetry is available for dependency management."""
        
        try:
            import subprocess
            result = subprocess.run(["poetry", "--version"], capture_output=True, text=True)
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _show_compatibility_warning(self, warning_type: str, message: str):
        """Show a compatibility warning once per session."""
        
        if warning_type not in self._compatibility_warnings_shown:
            warnings.warn(
                f"Backward Compatibility Warning ({warning_type}): {message}",
                UserWarning,
                stacklevel=3
            )
            self._compatibility_warnings_shown.add(warning_type)
    
    def get_compatibility_status(self) -> Dict[str, Any]:
        """Get current compatibility status."""
        
        return {
            "compatibility_mode": self.compatibility_mode.value,
            "legacy_config_valid": self.legacy_config.is_valid_legacy_setup(),
            "adk_compatible": self.ensure_adk_compatibility(),
            "vertex_ai_compatible": self.ensure_vertex_ai_compatibility(),
            "environment": {
                "google_cloud_project": bool(os.getenv("GOOGLE_CLOUD_PROJECT")),
                "vertex_ai_enabled": os.getenv("GOOGLE_GENAI_USE_VERTEXAI") == "true",
                "root_agent_model": bool(os.getenv("ROOT_AGENT_MODEL")),
                "multi_provider_vars": len([
                    v for v in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY"] 
                    if os.getenv(v)
                ])
            },
            "migration_available": self.compatibility_mode != CompatibilityMode.FULL_MULTI_PROVIDER
        }


# Global backward compatibility manager instance
_compatibility_manager = None


def get_backward_compatibility_manager() -> BackwardCompatibilityManager:
    """Get the global backward compatibility manager."""
    global _compatibility_manager
    if _compatibility_manager is None:
        _compatibility_manager = BackwardCompatibilityManager()
    return _compatibility_manager


def create_legacy_compatible_agent(
    name: str,
    instruction: Union[str, Callable],
    model: Optional[str] = None,
    **kwargs
) -> agents.Agent:
    """Create an agent with full backward compatibility."""
    manager = get_backward_compatibility_manager()
    return manager.create_legacy_compatible_agent(name, instruction, model, **kwargs)


def ensure_deployment_compatibility() -> bool:
    """Ensure compatibility with existing deployment pipelines."""
    manager = get_backward_compatibility_manager()
    
    adk_compatible = manager.ensure_adk_compatibility()
    vertex_ai_compatible = manager.ensure_vertex_ai_compatibility()
    
    return adk_compatible and vertex_ai_compatible


def get_migration_plan(target_mode: str = "enhanced_adk") -> Dict[str, Any]:
    """Get a migration plan for upgrading to multi-provider support."""
    manager = get_backward_compatibility_manager()
    
    try:
        target_mode_enum = CompatibilityMode(target_mode)
        plan = manager.get_migration_plan(target_mode_enum)
        
        return {
            "current_mode": plan.current_mode.value,
            "target_mode": plan.target_mode.value,
            "migration_steps": plan.migration_steps,
            "required_changes": plan.required_changes,
            "risk_level": plan.risk_level,
            "estimated_effort": plan.estimated_effort,
            "rollback_plan": plan.rollback_plan
        }
    except ValueError:
        return {"error": f"Invalid target mode: {target_mode}"}


def get_migration_utilities() -> Dict[str, Callable]:
    """Get migration utilities for helping with upgrades."""
    manager = get_backward_compatibility_manager()
    return manager.create_migration_utilities()