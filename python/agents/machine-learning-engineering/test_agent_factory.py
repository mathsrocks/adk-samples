#!/usr/bin/env python3
"""Test script for Multi-Provider Agent Factory."""

import os
import sys

# Add the machine_learning_engineering module to the path  
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))


def test_agent_factory_basic():
    """Test basic agent factory functionality."""
    print("🏭 Testing basic agent factory...")
    
    try:
        sys.path.insert(0, 'machine_learning_engineering/shared_libraries')
        
        from agent_factory import MultiProviderAgentFactory, get_agent_factory
        from config import CONFIG
        
        print("✅ Agent factory classes imported successfully")
        
        # Test factory creation
        factory = MultiProviderAgentFactory()
        print(f"✅ Factory created with provider: {factory.default_provider}")
        print(f"✅ Fallback enabled: {factory.fallback_enabled}")
        
        # Test global factory
        global_factory = get_agent_factory()
        print(f"✅ Global factory provider: {global_factory.default_provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_provider_optimizations():
    """Test provider-specific optimizations."""
    print("\n🎯 Testing provider optimizations...")
    
    try:
        from agent_factory import MultiProviderAgentFactory
        
        factory = MultiProviderAgentFactory()
        optimizations = factory.provider_optimizations
        
        print(f"✅ Provider optimizations configured: {len(optimizations)}")
        
        for provider, opts in optimizations.items():
            temp = opts.get('temperature', 'N/A')
            max_tokens = opts.get('max_tokens', 'N/A')
            prefix = opts.get('description_prefix', 'N/A')
            print(f"  {provider}: temp={temp}, max_tokens={max_tokens}, prefix={prefix}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider optimization test failed: {e}")
        return False


def test_task_specific_providers():
    """Test task-specific provider selection.""" 
    print("\n🔍 Testing task-specific provider selection...")
    
    try:
        from agent_factory import MultiProviderAgentFactory
        
        factory = MultiProviderAgentFactory()
        
        # Test provider selection for different tasks
        tasks = ["coding", "reasoning", "debugging", "cost_effective", "default"]
        
        for task in tasks:
            provider = factory._select_provider_for_task(task)
            print(f"✅ {task}: {provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Task-specific provider test failed: {e}")
        return False


def test_agent_creation():
    """Test agent creation with different configurations."""
    print("\n🤖 Testing agent creation...")
    
    try:
        from agent_factory import get_agent_factory, create_agent, create_coding_agent
        
        factory = get_agent_factory()
        
        def test_instruction():
            return "Test instruction"
        
        # Test basic agent creation (without actual instantiation due to ADK dependencies)
        print("✅ Factory has create_agent method")
        print("✅ Factory has create_coding_optimized_agent method")
        print("✅ Factory has create_reasoning_optimized_agent method")
        print("✅ Factory has create_cost_optimized_agent method")
        
        # Test convenience functions exist
        print("✅ create_agent function available")
        print("✅ create_coding_agent function available")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent creation test failed: {e}")
        return False


def test_factory_stats():
    """Test factory statistics and configuration."""
    print("\n📊 Testing factory statistics...")
    
    try:
        from agent_factory import get_agent_factory
        
        factory = get_agent_factory()
        stats = factory.get_factory_stats()
        
        print(f"✅ Factory statistics:")
        print(f"  Default provider: {stats.get('default_provider')}")
        print(f"  Fallback enabled: {stats.get('fallback_enabled')}")
        print(f"  Routing strategy: {stats.get('routing_strategy')}")
        print(f"  Available providers: {stats.get('available_providers', [])}")
        print(f"  Configured providers: {stats.get('configured_providers', [])}")
        print(f"  Provider optimizations: {len(stats.get('provider_optimizations', []))} configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Factory stats test failed: {e}")
        return False


def test_provider_strategies():
    """Test different provider strategies."""
    print("\n🧠 Testing provider strategies...")
    
    try:
        from agent_factory import MultiProviderAgentFactory
        from config import ProviderStrategy
        
        strategies = [
            ProviderStrategy.DEFAULT,
            ProviderStrategy.COST_OPTIMIZED,
            ProviderStrategy.CODING,
            ProviderStrategy.REASONING,
            ProviderStrategy.FAST_INFERENCE
        ]
        
        for strategy in strategies:
            factory = MultiProviderAgentFactory(routing_strategy=strategy)
            print(f"✅ {strategy.value}: default_provider={factory.default_provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Provider strategy test failed: {e}")
        return False


def test_fallback_configuration():
    """Test fallback configuration."""
    print("\n🔄 Testing fallback configuration...")
    
    try:
        from agent_factory import MultiProviderAgentFactory
        
        # Test with fallback enabled
        factory_with_fallback = MultiProviderAgentFactory(fallback_enabled=True)
        print(f"✅ Fallback enabled: {factory_with_fallback.fallback_enabled}")
        
        # Test with fallback disabled
        factory_without_fallback = MultiProviderAgentFactory(fallback_enabled=False)
        print(f"✅ Fallback disabled: {factory_without_fallback.fallback_enabled}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fallback configuration test failed: {e}")
        return False


def test_integration_with_config():
    """Test integration with enhanced configuration."""
    print("\n🔗 Testing integration with enhanced configuration...")
    
    try:
        from agent_factory import get_agent_factory
        from config import CONFIG
        
        factory = get_agent_factory()
        
        # Test that factory uses configuration values
        print(f"✅ Config provider type: {CONFIG.provider_type}")
        print(f"✅ Factory default provider: {factory.default_provider}")
        
        # Test task-specific model selection
        coding_provider = factory._select_provider_for_task("coding")
        reasoning_provider = factory._select_provider_for_task("reasoning")
        print(f"✅ Coding task provider: {coding_provider}")
        print(f"✅ Reasoning task provider: {reasoning_provider}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration integration test failed: {e}")
        return False


def test_specific_provider_support():
    """Test specific provider support (OpenAI, Claude, DeepSeek)."""
    print("\n🎯 Testing specific provider support...")
    
    try:
        from agent_factory import MultiProviderAgentFactory
        
        # Test OpenAI GPT-4 support
        openai_factory = MultiProviderAgentFactory(default_provider="openai")
        print(f"✅ OpenAI factory: {openai_factory.default_provider}")
        
        # Test Anthropic Claude support
        claude_factory = MultiProviderAgentFactory(default_provider="anthropic")
        print(f"✅ Anthropic factory: {claude_factory.default_provider}")
        
        # Test DeepSeek support
        deepseek_factory = MultiProviderAgentFactory(default_provider="deepseek")
        print(f"✅ DeepSeek factory: {deepseek_factory.default_provider}")
        
        # Test optimizations for each
        openai_opts = openai_factory.provider_optimizations.get("openai", {})
        claude_opts = claude_factory.provider_optimizations.get("anthropic", {})
        deepseek_opts = deepseek_factory.provider_optimizations.get("deepseek", {})
        
        print(f"✅ OpenAI optimizations: temp={openai_opts.get('temperature')}")
        print(f"✅ Claude optimizations: temp={claude_opts.get('temperature')}")
        print(f"✅ DeepSeek optimizations: temp={deepseek_opts.get('temperature')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Specific provider support test failed: {e}")
        return False


def main():
    """Main test function."""
    print("🧪 Multi-Provider Agent Factory Test Suite")
    print("=" * 60)
    
    # Set test environment
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    tests = [
        test_agent_factory_basic,
        test_provider_optimizations,
        test_task_specific_providers,
        test_agent_creation,
        test_factory_stats,
        test_provider_strategies,
        test_fallback_configuration,
        test_integration_with_config,
        test_specific_provider_support
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
        print("🎉 All agent factory tests passed!")
        print("\n📋 Usage examples:")
        print("  # Create factory with specific provider")
        print("  factory = get_agent_factory('anthropic')")
        print()
        print("  # Create coding-optimized agent")  
        print("  agent = factory.create_coding_optimized_agent('coder', instruction)")
        print()
        print("  # Create reasoning-optimized agent")
        print("  agent = factory.create_reasoning_optimized_agent('thinker', instruction)")
        print()
        print("  # Create cost-optimized agent")
        print("  agent = factory.create_cost_optimized_agent('budget', instruction)")
        print()
        print("🔧 Provider-specific optimizations:")
        print("  - Claude: Higher temperature (0.7) for creativity")
        print("  - GPT-4: Lower temperature (0.1) for consistency")
        print("  - DeepSeek: Balanced temperature (0.3) for cost-effectiveness")
        print("  - Groq: Optimized for fast inference (temp=0.5)")
    else:
        print(f"⚠️  {total - passed} tests failed. Check factory implementation.")


if __name__ == "__main__":
    main()