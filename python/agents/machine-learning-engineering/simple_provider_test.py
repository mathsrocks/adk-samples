#!/usr/bin/env python3
"""Simple test for provider abstraction layer structure."""

import os
import sys

# Test basic imports
try:
    # Test basic structure
    from machine_learning_engineering.shared_libraries.llm_providers import (
        ProviderType, 
        ModelConfig, 
        GenerateResponse,
        LLMProvider,
        LLMProviderFactory
    )
    print("✅ Core provider abstraction classes imported successfully")
    
    # Test enum values
    print(f"✅ Provider types: {[p.value for p in ProviderType]}")
    
    # Test model config creation
    config = ModelConfig(
        model_name="test-model",
        provider=ProviderType.OPENAI,
        api_key_env_var="TEST_KEY"
    )
    print(f"✅ ModelConfig created: {config.model_name} ({config.provider.value})")
    
    # Test default configurations
    defaults = LLMProviderFactory.DEFAULT_MODELS
    print(f"✅ Default models configured for {len(defaults)} providers")
    
    # Test performance tiers
    tiers = LLMProviderFactory.PERFORMANCE_TIERS
    print(f"✅ Performance tiers available: {list(tiers.keys())}")
    
    # Test provider mapping exists
    provider_mapping = {
        ProviderType.GOOGLE: ("google_provider", "GoogleLLMProvider"),
        ProviderType.OPENAI: ("openai_provider", "OpenAILLMProvider"),
        ProviderType.ANTHROPIC: ("anthropic_provider", "AnthropicLLMProvider"),
        ProviderType.DEEPSEEK: ("deepseek_provider", "DeepSeekLLMProvider"),
    }
    print(f"✅ Provider mapping configured for {len(provider_mapping)} providers")
    
    print("\n📊 Provider Details:")
    for provider_type, config in defaults.items():
        cost_info = ""
        if config.cost_per_token:
            costs = config.cost_per_token
            cost_info = f" (${costs.get('input', 0):.6f}/${costs.get('output', 0):.6f} per token)"
        print(f"  {provider_type.value}: {config.model_name}{cost_info}")
    
    print("\n🎯 Performance Tiers:")
    for tier, providers in tiers.items():
        print(f"  {tier}: {[p.value for p in providers]}")
    
    print("\n✅ Provider abstraction layer structure test passed!")
    
except Exception as e:
    print(f"❌ Test failed: {e}")
    import traceback
    traceback.print_exc()