#!/usr/bin/env python3
"""Test script for enhanced configuration with multi-provider support."""

import os
import sys

# Add the machine_learning_engineering module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


def test_basic_configuration():
    """Test basic configuration loading and backward compatibility."""
    print("🔧 Testing basic configuration...")
    
    try:
        # Import config directly to avoid module initialization issues
        sys.path.insert(0, 'machine_learning_engineering/shared_libraries')
        
        from config import DefaultConfig, ProviderStrategy, ProviderConfig, CONFIG
        
        print("✅ Configuration classes imported successfully")
        print(f"  Default agent model: {CONFIG.agent_model}")
        print(f"  Provider type: {CONFIG.provider_type}")
        print(f"  Provider strategy: {CONFIG.provider_strategy}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic configuration test failed: {e}")
        return False


def test_provider_configurations():
    """Test provider-specific configurations."""
    print("\n🔌 Testing provider configurations...")
    
    try:
        from config import CONFIG
        
        print(f"✅ Provider configs: {len(CONFIG.provider_configs)} configured")
        
        for name, config in CONFIG.provider_configs.items():
            status = "🟢" if config.enabled else "🔴"
            print(f"  {status} {name}: {config.model_name} (priority: {config.priority})")
        
        # Test API key mapping
        print(f"✅ API key mappings: {len(CONFIG.api_key_mapping)} providers")
        for provider, env_var in list(CONFIG.api_key_mapping.items())[:3]:
            print(f"  {provider} → {env_var}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider configuration test failed: {e}")
        return False


def test_backward_compatibility():
    """Test backward compatibility with ROOT_AGENT_MODEL."""
    print("\n🔄 Testing backward compatibility...")
    
    try:
        from config import CONFIG
        
        # Test ROOT_AGENT_MODEL detection
        original_model = CONFIG.agent_model
        print(f"✅ Original agent model: {original_model}")
        
        # Test provider inference
        provider_type = CONFIG._infer_provider_from_model("gpt-4o")
        print(f"✅ GPT-4o inferred as: {provider_type}")
        
        provider_type = CONFIG._infer_provider_from_model("claude-3-sonnet")
        print(f"✅ Claude inferred as: {provider_type}")
        
        provider_type = CONFIG._infer_provider_from_model("deepseek-chat")
        print(f"✅ DeepSeek inferred as: {provider_type}")
        
        # Test model-for-task functionality
        coding_model = CONFIG.get_model_for_task("coding")
        reasoning_model = CONFIG.get_model_for_task("reasoning")
        default_model = CONFIG.get_model_for_task()
        
        print(f"✅ Task-specific models:")
        print(f"  Coding: {coding_model}")
        print(f"  Reasoning: {reasoning_model}")
        print(f"  Default: {default_model}")
        
        return True
        
    except Exception as e:
        print(f"❌ Backward compatibility test failed: {e}")
        return False


def test_provider_management():
    """Test provider management functions."""
    print("\n⚙️  Testing provider management...")
    
    try:
        from config import CONFIG
        
        # Test enabled providers
        enabled = CONFIG.get_enabled_providers()
        print(f"✅ Enabled providers: {enabled}")
        
        # Test fallback chain
        fallback = CONFIG.get_fallback_chain()
        print(f"✅ Fallback chain: {fallback}")
        
        # Test provider configuration check
        configured_providers = []
        for provider in ["google", "openai", "anthropic", "deepseek", "ollama"]:
            is_configured = CONFIG.is_provider_configured(provider)
            status = "✅" if is_configured else "❌"
            print(f"  {status} {provider}: {'configured' if is_configured else 'not configured'}")
            if is_configured:
                configured_providers.append(provider)
        
        print(f"✅ {len(configured_providers)} providers properly configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider management test failed: {e}")
        return False


def test_strategy_updates():
    """Test provider strategy updates."""
    print("\n🎯 Testing strategy updates...")
    
    try:
        from config import CONFIG, ProviderStrategy
        
        original_strategy = CONFIG.provider_strategy
        original_fallback = CONFIG.fallback_providers.copy()
        
        # Test strategy changes
        strategies = [
            ProviderStrategy.COST_OPTIMIZED,
            ProviderStrategy.CODING,
            ProviderStrategy.FAST_INFERENCE
        ]
        
        for strategy in strategies:
            CONFIG.update_provider_strategy(strategy)
            print(f"✅ {strategy.value}: {CONFIG.fallback_providers[:3]}")
        
        # Restore original
        CONFIG.provider_strategy = original_strategy
        CONFIG.fallback_providers = original_fallback
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy update test failed: {e}")
        return False


def test_configuration_bridge():
    """Test configuration bridge integration."""
    print("\n🌉 Testing configuration bridge...")
    
    try:
        from config_bridge import (
            ConfigurationManager, 
            BackwardCompatibilityAdapter,
            get_agent_model,
            ConfiguredProviderFactory
        )
        
        # Test backward compatibility adapter
        agent_model = BackwardCompatibilityAdapter.get_agent_model()
        print(f"✅ Backward compatible agent model: {agent_model}")
        
        coding_model = BackwardCompatibilityAdapter.get_agent_model_for_task("coding")
        print(f"✅ Coding model: {coding_model}")
        
        # Test configuration manager
        summary = ConfigurationManager.get_configuration_summary()
        print(f"✅ Configuration summary: {len(summary)} fields")
        print(f"  Primary: {summary.get('primary_provider')}")
        print(f"  Strategy: {summary.get('provider_strategy')}")
        print(f"  Enabled: {len(summary.get('enabled_providers', []))}")
        
        # Test validation
        issues = ConfigurationManager.validate_configuration()
        if issues:
            print(f"⚠️  Configuration issues: {len(issues)}")
            for issue in issues[:2]:
                print(f"    - {issue}")
        else:
            print("✅ Configuration validation passed")
        
        # Test auto-configuration
        configured_count = ConfigurationManager.auto_configure_from_environment()
        print(f"✅ Auto-configured {configured_count} providers from environment")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment_integration():
    """Test environment variable integration."""
    print("\n🌍 Testing environment integration...")
    
    try:
        from config import CONFIG
        
        # Test current environment
        env_vars = [
            "ROOT_AGENT_MODEL",
            "GOOGLE_GENAI_USE_VERTEXAI", 
            "GOOGLE_CLOUD_PROJECT",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY"
        ]
        
        print("✅ Environment variables:")
        for var in env_vars:
            value = os.environ.get(var, "not set")
            status = "🟢" if value != "not set" else "🔴"
            print(f"  {status} {var}: {value if value != 'not set' else 'not set'}")
        
        # Test API key retrieval
        for provider in ["google", "openai", "anthropic", "deepseek"]:
            env_var = CONFIG.get_api_key_for_provider(provider)
            print(f"✅ {provider} API key env: {env_var}")
        
        return True
        
    except Exception as e:
        print(f"❌ Environment integration test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Enhanced Configuration Test Suite")
    print("=" * 60)
    
    # Set test environment to avoid issues
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    tests = [
        test_basic_configuration,
        test_provider_configurations,
        test_backward_compatibility,
        test_provider_management,
        test_strategy_updates,
        test_configuration_bridge,
        test_environment_integration
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
        print("🎉 All configuration enhancement tests passed!")
        print("\n📋 Usage examples:")
        print("  # Use cost-optimized providers")
        print("  CONFIG.update_provider_strategy('cost_optimized')")
        print("  ")
        print("  # Get coding-specific model")
        print("  model = CONFIG.get_model_for_task('coding')")
        print("  ")
        print("  # Enable/disable providers")
        print("  ConfigurationManager.enable_provider('openai', True)")
    else:
        print(f"⚠️  {total - passed} tests failed. Check configuration setup.")


if __name__ == "__main__":
    main()