#!/usr/bin/env python3
"""Test script for LLM provider abstraction layer."""

import os
import sys
from typing import List

# Add the machine_learning_engineering module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

from machine_learning_engineering.shared_libraries.llm_providers import (
    LLMProviderFactory, 
    MultiProviderLLM, 
    ProviderType
)


def test_provider_availability():
    """Test which providers are available based on environment configuration."""
    print("🔍 Testing provider availability...")
    
    available_providers = LLMProviderFactory.get_available_providers()
    print(f"✅ Available providers: {[p.value for p in available_providers]}")
    
    for provider_type in ProviderType:
        try:
            provider = LLMProviderFactory.create_provider(provider_type)
            is_available = provider.is_available()
            status = "✅" if is_available else "❌"
            print(f"  {status} {provider_type.value}: {provider.get_model_name()}")
        except Exception as e:
            print(f"  ❌ {provider_type.value}: Failed to create - {str(e)[:80]}...")


def test_default_provider():
    """Test default provider creation."""
    print("\n🔧 Testing default provider creation...")
    
    try:
        provider = LLMProviderFactory.create_default_provider()
        print(f"✅ Default provider: {provider.get_provider_type().value} ({provider.get_model_name()})")
        return provider
    except Exception as e:
        print(f"❌ Failed to create default provider: {e}")
        return None


def test_multi_provider_llm():
    """Test multi-provider LLM with fallback."""
    print("\n🚀 Testing MultiProviderLLM...")
    
    # Test different routing strategies
    strategies = ["default", "cost_optimized", "coding", "fast_inference"]
    
    for strategy in strategies:
        try:
            multi_llm = MultiProviderLLM(routing_strategy=strategy)
            primary = multi_llm.get_primary_provider()
            available = multi_llm.get_available_providers()
            
            print(f"  {strategy}: Primary={primary.get_provider_type().value if primary else 'None'}, "
                  f"Available={[p.value for p in available]}")
        except Exception as e:
            print(f"  {strategy}: ❌ {str(e)[:80]}...")


def test_performance_tiers():
    """Test performance tier provider selection."""
    print("\n📊 Testing performance tiers...")
    
    tiers = ["coding", "reasoning", "cost_effective", "fast_inference", "long_context", "open_source"]
    
    for tier in tiers:
        providers = LLMProviderFactory.get_providers_by_tier(tier)
        print(f"  {tier}: {[p.value for p in providers]}")


def test_cost_optimization():
    """Test cost-optimized provider creation."""
    print("\n💰 Testing cost-optimized provider...")
    
    try:
        provider = LLMProviderFactory.create_cost_optimized_provider()
        print(f"✅ Cost-optimized provider: {provider.get_provider_type().value} ({provider.get_model_name()})")
        
        # Show cost information if available
        if provider.config.cost_per_token:
            costs = provider.config.cost_per_token
            print(f"  💵 Costs: Input=${costs.get('input', 0):.6f}, Output=${costs.get('output', 0):.6f} per token")
        
    except Exception as e:
        print(f"❌ Failed to create cost-optimized provider: {e}")


def test_provider_configuration():
    """Test provider configuration details."""
    print("\n⚙️  Testing provider configurations...")
    
    for provider_type in [ProviderType.GOOGLE, ProviderType.OPENAI, ProviderType.ANTHROPIC, 
                         ProviderType.DEEPSEEK, ProviderType.GROQ, ProviderType.OLLAMA]:
        try:
            config = LLMProviderFactory.DEFAULT_MODELS.get(provider_type)
            if config:
                print(f"  {provider_type.value}:")
                print(f"    Model: {config.model_name}")
                print(f"    Context: {config.context_length:,} tokens" if config.context_length else "    Context: N/A")
                if config.cost_per_token:
                    costs = config.cost_per_token
                    print(f"    Cost: ${costs.get('input', 0):.6f}/${costs.get('output', 0):.6f} per token")
                print(f"    API Key: {config.api_key_env_var}")
        except Exception as e:
            print(f"  {provider_type.value}: ❌ {e}")


def main():
    """Main test function."""
    print("🧪 LLM Provider Abstraction Layer Test\n")
    print("=" * 50)
    
    # Set test environment variables to avoid actual API calls
    os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "true")
    os.environ.setdefault("GOOGLE_CLOUD_PROJECT", "test-project")
    os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    test_provider_availability()
    test_default_provider()
    test_multi_provider_llm()
    test_performance_tiers()
    test_cost_optimization()
    test_provider_configuration()
    
    print("\n" + "=" * 50)
    print("✅ Provider abstraction layer test completed!")
    print("\n📋 Installation commands for providers:")
    print("  All providers:    poetry install --extras all-providers")
    print("  Cost-effective:   poetry install --extras cost-effective")
    print("  Open-source:      poetry install --extras open-source")
    print("  Specific:         poetry install --extras 'openai anthropic deepseek'")


if __name__ == "__main__":
    main()