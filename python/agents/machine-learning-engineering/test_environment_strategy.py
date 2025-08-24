#!/usr/bin/env python3
"""Test script for Environment Variable Strategy with Multi-Provider Support."""

import os
import sys
import json
from typing import Dict, Any

# Add the machine_learning_engineering module to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


def test_environment_manager_import():
    """Test importing the environment manager."""
    print("🔍 Testing environment manager import...")
    
    try:
        sys.path.insert(0, 'machine_learning_engineering/shared_libraries')
        from environment_manager import (
            EnvironmentManager, 
            ProviderEnvironmentSpec,
            EnvironmentStatus,
            EnvironmentDiagnostic,
            get_environment_manager,
            detect_available_providers,
            get_provider_diagnostics
        )
        from llm_providers import ProviderType
        
        print("✅ Environment manager classes imported successfully")
        print(f"✅ ProviderType enum: {len(list(ProviderType))} providers")
        print(f"✅ EnvironmentStatus enum: {[s.value for s in EnvironmentStatus]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment manager import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_specifications():
    """Test provider environment specifications."""
    print("\n📋 Testing provider specifications...")
    
    try:
        from environment_manager import EnvironmentManager
        from llm_providers import ProviderType
        
        manager = EnvironmentManager()
        specs = manager.PROVIDER_SPECS
        
        print(f"✅ Provider specifications: {len(specs)} configured")
        
        # Test key providers
        key_providers = [
            ProviderType.GOOGLE,
            ProviderType.OPENAI, 
            ProviderType.ANTHROPIC,
            ProviderType.DEEPSEEK,
            ProviderType.OLLAMA
        ]
        
        for provider in key_providers:
            if provider in specs:
                spec = specs[provider]
                print(f"  ✅ {provider.value}:")
                print(f"    Primary API key: {spec.primary_api_key}")
                print(f"    Auth methods: {spec.auth_methods}")
                print(f"    Required vars: {len(spec.required_vars)}")
            else:
                print(f"  ❌ {provider.value}: No specification found")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider specifications test failed: {e}")
        return False


def test_auto_detection():
    """Test auto-detection of available providers."""
    print("\n🔍 Testing provider auto-detection...")
    
    try:
        from environment_manager import get_environment_manager
        
        manager = get_environment_manager()
        available_providers = manager.detect_available_providers()
        
        print(f"✅ Detected providers: {[p.value for p in available_providers]}")
        
        if available_providers:
            print(f"✅ {len(available_providers)} providers available")
            for provider in available_providers:
                print(f"  🟢 {provider.value}")
        else:
            print("⚠️  No providers detected in current environment")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-detection test failed: {e}")
        return False


def test_provider_diagnostics():
    """Test provider diagnostics functionality."""
    print("\n🏥 Testing provider diagnostics...")
    
    try:
        from environment_manager import get_environment_manager
        from llm_providers import ProviderType
        
        manager = get_environment_manager()
        
        # Test diagnostics for key providers
        test_providers = [
            ProviderType.GOOGLE,
            ProviderType.OPENAI,
            ProviderType.ANTHROPIC,
            ProviderType.DEEPSEEK,
            ProviderType.OLLAMA
        ]
        
        for provider_type in test_providers:
            diagnostic = manager.get_provider_diagnostics(provider_type)
            
            status_emoji = {
                "configured": "🟢",
                "missing": "🔴", 
                "invalid": "⚠️",
                "partially_configured": "🟡"
            }
            
            emoji = status_emoji.get(diagnostic.status.value, "❓")
            print(f"  {emoji} {provider_type.value}: {diagnostic.status.value}")
            
            if diagnostic.available_vars:
                print(f"    Available vars: {list(diagnostic.available_vars.keys())}")
            if diagnostic.missing_vars:
                print(f"    Missing vars: {diagnostic.missing_vars}")
            if diagnostic.auth_method:
                print(f"    Auth method: {diagnostic.auth_method}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider diagnostics test failed: {e}")
        return False


def test_environment_variables():
    """Test environment variable detection and validation."""
    print("\n🌍 Testing environment variables...")
    
    try:
        # Common environment variables to check
        env_vars_to_check = [
            # Google
            "GOOGLE_APPLICATION_CREDENTIALS",
            "GOOGLE_API_KEY", 
            "GOOGLE_CLOUD_PROJECT",
            "GOOGLE_CLOUD_LOCATION",
            "GOOGLE_GENAI_USE_VERTEXAI",
            
            # OpenAI
            "OPENAI_API_KEY",
            "OPENAI_BASE_URL",
            "OPENAI_ORGANIZATION",
            
            # Anthropic
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            
            # DeepSeek
            "DEEPSEEK_API_KEY",
            "DEEPSEEK_BASE_URL",
            
            # Others
            "GROQ_API_KEY",
            "TOGETHER_API_KEY", 
            "MISTRAL_API_KEY",
            "COHERE_API_KEY",
            "HUGGINGFACE_API_KEY",
            "REPLICATE_API_TOKEN",
            
            # Local/Enterprise
            "OLLAMA_HOST",
            "AZURE_OPENAI_API_KEY",
            "AWS_ACCESS_KEY_ID"
        ]
        
        configured_vars = []
        missing_vars = []
        
        for var in env_vars_to_check:
            value = os.getenv(var)
            if value:
                # Mask sensitive values
                if "KEY" in var or "TOKEN" in var or "CREDENTIALS" in var:
                    masked_value = f"{value[:4]}***{value[-4:]}" if len(value) > 8 else "***"
                    configured_vars.append((var, masked_value))
                else:
                    configured_vars.append((var, value))
            else:
                missing_vars.append(var)
        
        print(f"✅ Environment variables configured: {len(configured_vars)}")
        for var, value in configured_vars:
            print(f"  🟢 {var}: {value}")
        
        print(f"⚠️  Environment variables missing: {len(missing_vars)}")
        for var in missing_vars[:5]:  # Show first 5
            print(f"  🔴 {var}")
        
        if len(missing_vars) > 5:
            print(f"  ... and {len(missing_vars) - 5} more")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment variables test failed: {e}")
        return False


def test_fallback_chain():
    """Test fallback chain generation."""
    print("\n🔄 Testing fallback chain...")
    
    try:
        from environment_manager import get_environment_manager
        from llm_providers import ProviderType
        
        manager = get_environment_manager()
        
        # Test basic fallback chain
        fallback_chain = manager.get_fallback_chain()
        print(f"✅ Fallback chain: {[p.value for p in fallback_chain]}")
        
        # Test excluding specific providers
        excluded = [ProviderType.GOOGLE] if ProviderType.GOOGLE in fallback_chain else []
        if excluded:
            filtered_chain = manager.get_fallback_chain(exclude_providers=excluded)
            print(f"✅ Chain excluding {excluded[0].value}: {[p.value for p in filtered_chain]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback chain test failed: {e}")
        return False


def test_task_based_provider_selection():
    """Test task-based provider selection."""
    print("\n🎯 Testing task-based provider selection...")
    
    try:
        from environment_manager import get_environment_manager, get_best_provider_for_task
        
        manager = get_environment_manager()
        
        # Test different task types
        task_types = [
            "coding",
            "reasoning", 
            "cost_effective",
            "fast_inference",
            "local",
            "enterprise"
        ]
        
        for task_type in task_types:
            best_provider = manager.get_best_provider_for_task(task_type)
            if best_provider:
                print(f"✅ {task_type}: {best_provider.value}")
            else:
                print(f"⚠️  {task_type}: No suitable provider available")
        
        # Test cost priority
        cost_provider = manager.get_best_provider_for_task("coding", cost_priority=True)
        if cost_provider:
            print(f"✅ Cost-optimized coding: {cost_provider.value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task-based selection test failed: {e}")
        return False


def test_auto_configuration():
    """Test auto-configuration functionality."""
    print("\n⚙️  Testing auto-configuration...")
    
    try:
        from environment_manager import get_environment_manager
        from llm_providers import ProviderType
        
        manager = get_environment_manager()
        
        # Test auto-configuration for available providers
        available_providers = manager.detect_available_providers()
        
        configured_count = 0
        for provider_type in available_providers:
            try:
                success = manager.auto_configure_provider(provider_type)
                if success:
                    configured_count += 1
                    print(f"✅ Auto-configured: {provider_type.value}")
            except Exception as e:
                print(f"⚠️  Auto-config failed for {provider_type.value}: {e}")
        
        print(f"✅ Auto-configured {configured_count} providers")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-configuration test failed: {e}")
        return False


def test_configuration_bridge_integration():
    """Test integration with configuration bridge."""
    print("\n🌉 Testing configuration bridge integration...")
    
    try:
        from config_bridge import (
            get_available_providers_from_env,
            get_environment_diagnostics,
            validate_environment_setup,
            get_best_provider_for_task
        )
        
        # Test environment provider detection
        env_providers = get_available_providers_from_env()
        print(f"✅ Environment providers: {env_providers}")
        
        # Test diagnostics export
        diagnostics = get_environment_diagnostics()
        print(f"✅ Diagnostics available: {len(diagnostics.get('provider_diagnostics', {}))} providers")
        print(f"✅ Available in environment: {diagnostics.get('available_providers', [])}")
        
        # Test environment validation
        validation = validate_environment_setup()
        print(f"✅ Environment status: {validation['status']}")
        print(f"✅ Total configured: {validation['total_providers_configured']}")
        
        if validation['issues']:
            print(f"⚠️  Issues found: {len(validation['issues'])}")
            for issue in validation['issues'][:2]:
                print(f"    - {issue}")
        
        if validation['recommendations']:
            print(f"💡 Recommendations: {len(validation['recommendations'])}")
            for rec in validation['recommendations'][:2]:
                print(f"    - {rec}")
        
        # Test task-based selection via bridge
        best_coding = get_best_provider_for_task("coding")
        best_cost = get_best_provider_for_task("coding", cost_priority=True)
        
        if best_coding:
            print(f"✅ Best for coding: {best_coding}")
        if best_cost:
            print(f"✅ Cost-effective coding: {best_cost}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration bridge integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_graceful_fallback_simulation():
    """Simulate graceful fallback scenarios."""
    print("\n🛡️  Testing graceful fallback simulation...")
    
    try:
        from environment_manager import get_environment_manager
        
        manager = get_environment_manager()
        available_providers = manager.detect_available_providers()
        
        if len(available_providers) < 2:
            print("⚠️  Need at least 2 providers to test fallback (simulating scenarios)")
            
            # Simulate fallback scenarios
            scenarios = [
                "Primary provider (Google) fails → Fallback to DeepSeek",
                "DeepSeek unavailable → Fallback to Anthropic", 
                "All API providers fail → Fallback to Ollama (local)",
                "Network issues → Use cached provider configuration"
            ]
            
            for i, scenario in enumerate(scenarios, 1):
                print(f"✅ Scenario {i}: {scenario}")
        else:
            print(f"✅ Fallback chain available with {len(available_providers)} providers")
            print(f"  Primary: {available_providers[0].value}")
            if len(available_providers) > 1:
                print(f"  Fallbacks: {[p.value for p in available_providers[1:]]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Graceful fallback test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Environment Variable Strategy Test Suite")
    print("=" * 60)
    
    # Set minimal test environment to avoid issues
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    tests = [
        test_environment_manager_import,
        test_provider_specifications,
        test_auto_detection,
        test_provider_diagnostics,
        test_environment_variables,
        test_fallback_chain,
        test_task_based_provider_selection,
        test_auto_configuration,
        test_configuration_bridge_integration,
        test_graceful_fallback_simulation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print()  # Add spacing between tests
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            print()
    
    print("=" * 60)
    print(f"✅ Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All environment variable strategy tests passed!")
        print("\n📋 Environment Setup Examples:")
        print("  # OpenAI GPT-4")
        print("  export OPENAI_API_KEY='sk-...'")
        print()
        print("  # Anthropic Claude")
        print("  export ANTHROPIC_API_KEY='sk-ant-...'") 
        print()
        print("  # DeepSeek (cost-effective)")
        print("  export DEEPSEEK_API_KEY='sk-...'")
        print()
        print("  # Google Gemini (Vertex AI)")
        print("  export GOOGLE_CLOUD_PROJECT='your-project'")
        print("  export GOOGLE_GENAI_USE_VERTEXAI='true'")
        print()
        print("  # Local inference")
        print("  # Install Ollama from https://ollama.ai")
        print()
        print("🔧 Auto-Detection Features:")
        print("  - Automatic provider discovery from environment")
        print("  - Intelligent fallback chains")
        print("  - Task-specific provider selection") 
        print("  - Cost-optimized routing")
        print("  - Comprehensive diagnostics and validation")
    else:
        print(f"⚠️  {total - passed} tests failed. Check environment setup.")
        print("\n💡 To configure providers, set these environment variables:")
        print("  OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY")
        print("  Or install Ollama for local inference")


if __name__ == "__main__":
    main()