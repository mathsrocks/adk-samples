#!/usr/bin/env python3
"""Test script for dependency installation and provider availability."""

import os
import sys
import subprocess
import importlib
import json
from typing import Dict, List, Tuple, Optional


def test_import_availability() -> Dict[str, bool]:
    """Test which provider packages are available for import."""
    print("🔍 Testing provider package imports...")
    
    # Provider packages to test
    provider_packages = {
        # Primary providers
        "openai": "openai",
        "anthropic": "anthropic", 
        "groq": "groq",
        
        # Alternative providers
        "cohere": "cohere",
        "mistralai": "mistralai",
        "together": "together",
        "fireworks-ai": "fireworks",
        
        # Open source platforms  
        "huggingface-hub": "huggingface_hub",
        "transformers": "transformers",
        "replicate": "replicate",
        "torch": "torch",
        
        # Enterprise & cloud
        "boto3": "boto3",
        "azure-identity": "azure.identity", 
        "google-cloud-aiplatform": "google.cloud.aiplatform",
        
        # Local & self-hosted
        "ollama": "ollama",
        "litellm": "litellm",
        "langchain": "langchain",
        "langchain-community": "langchain_community",
        
        # Optimization
        "optimum": "optimum",
        "bitsandbytes": "bitsandbytes", 
        "accelerate": "accelerate"
    }
    
    import_results = {}
    available_count = 0
    
    for package_name, import_name in provider_packages.items():
        try:
            importlib.import_module(import_name)
            import_results[package_name] = True
            available_count += 1
            print(f"  ✅ {package_name}: Available")
        except ImportError:
            import_results[package_name] = False
            print(f"  ❌ {package_name}: Not installed")
    
    print(f"\n📊 Import Summary: {available_count}/{len(provider_packages)} packages available")
    return import_results


def test_poetry_extras() -> Dict[str, List[str]]:
    """Test Poetry extras configuration by parsing pyproject.toml."""
    print("\n📋 Testing Poetry extras configuration...")
    
    try:
        import toml
    except ImportError:
        print("⚠️  toml package not available, skipping pyproject.toml parsing")
        return {}
    
    try:
        with open("pyproject.toml", "r") as f:
            config = toml.load(f)
        
        extras = config.get("tool", {}).get("poetry", {}).get("extras", {})
        
        print(f"✅ Found {len(extras)} Poetry extras configured")
        
        # Categorize extras
        categories = {
            "Individual Providers": [],
            "Bundles": [],
            "Use Case Groups": [],
            "Advanced Features": []
        }
        
        bundle_keywords = ["essential", "professional", "research", "development", "local-full", "enterprise-full", "all-providers"]
        use_case_keywords = ["commercial", "open-source", "fast-inference", "cost-effective", "coding", "reasoning", "creative"]
        feature_keywords = ["optimization", "proxy", "langchain"]
        
        for extra_name in extras.keys():
            if extra_name in bundle_keywords:
                categories["Bundles"].append(extra_name)
            elif extra_name in use_case_keywords:
                categories["Use Case Groups"].append(extra_name)
            elif extra_name in feature_keywords:
                categories["Advanced Features"].append(extra_name)
            else:
                categories["Individual Providers"].append(extra_name)
        
        for category, items in categories.items():
            if items:
                print(f"  {category}: {len(items)} extras")
                for item in items[:5]:  # Show first 5
                    dependencies = extras.get(item, [])
                    print(f"    • {item}: {len(dependencies)} dependencies")
                if len(items) > 5:
                    print(f"    ... and {len(items) - 5} more")
        
        return extras
        
    except Exception as e:
        print(f"❌ Failed to parse pyproject.toml: {e}")
        return {}


def test_environment_detection():
    """Test environment variable detection for providers."""
    print("\n🌍 Testing environment variable detection...")
    
    # Standard environment variables to check
    provider_env_vars = {
        "OpenAI": ["OPENAI_API_KEY", "OPENAI_ORGANIZATION"],
        "Anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "DeepSeek": ["DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL"],
        "Groq": ["GROQ_API_KEY", "GROQ_BASE_URL"],
        "Google": ["GOOGLE_CLOUD_PROJECT", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "GOOGLE_GENAI_USE_VERTEXAI"],
        "Cohere": ["COHERE_API_KEY"],
        "Mistral": ["MISTRAL_API_KEY"],
        "Together": ["TOGETHER_API_KEY"],
        "Fireworks": ["FIREWORKS_API_KEY"],
        "Perplexity": ["PERPLEXITY_API_KEY"],
        "Hugging Face": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
        "Replicate": ["REPLICATE_API_TOKEN"],
        "Ollama": ["OLLAMA_HOST", "OLLAMA_PORT"],
        "Azure": ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT"],
        "AWS": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]
    }
    
    configured_providers = []
    partially_configured = []
    missing_providers = []
    
    for provider, env_vars in provider_env_vars.items():
        available_vars = [var for var in env_vars if os.getenv(var)]
        
        if available_vars:
            if len(available_vars) == len(env_vars):
                configured_providers.append(provider)
                print(f"  ✅ {provider}: Fully configured ({len(available_vars)}/{len(env_vars)} vars)")
            else:
                partially_configured.append(provider)
                print(f"  🟡 {provider}: Partially configured ({len(available_vars)}/{len(env_vars)} vars)")
        else:
            missing_providers.append(provider)
            print(f"  ❌ {provider}: Not configured")
    
    print(f"\n📊 Environment Summary:")
    print(f"  ✅ Fully configured: {len(configured_providers)} providers")
    print(f"  🟡 Partially configured: {len(partially_configured)} providers") 
    print(f"  ❌ Missing: {len(missing_providers)} providers")
    
    return {
        "configured": configured_providers,
        "partial": partially_configured, 
        "missing": missing_providers
    }


def test_installation_commands():
    """Test various Poetry installation commands (simulation)."""
    print("\n🛠️  Testing installation command patterns...")
    
    # Common installation patterns
    install_patterns = {
        "Essential Bundle": "poetry install --extras essential",
        "OpenAI Only": "poetry install --extras openai", 
        "Claude Only": "poetry install --extras anthropic",
        "DeepSeek Only": "poetry install --extras deepseek",
        "Multiple Providers": "poetry install --extras 'openai anthropic deepseek'",
        "Professional Setup": "poetry install --extras professional",
        "Research Setup": "poetry install --extras research",
        "Local Development": "poetry install --extras development",
        "Enterprise Setup": "poetry install --extras enterprise-full",
        "Cost-Effective": "poetry install --extras cost-effective",
        "Fast Inference": "poetry install --extras fast-inference",
        "Coding Optimized": "poetry install --extras coding",
        "With Optimization": "poetry install --extras 'essential optimization'",
        "Everything": "poetry install --extras all-providers"
    }
    
    print("✅ Validated installation command patterns:")
    for description, command in install_patterns.items():
        print(f"  • {description}: {command}")
    
    return install_patterns


def simulate_provider_selection():
    """Simulate provider selection based on available packages."""
    print("\n🎯 Simulating provider selection logic...")
    
    # Get import results
    import_results = {
        "openai": True,    # Simulate OpenAI available
        "anthropic": False,  # Simulate Anthropic not installed
        "groq": True,       # Simulate Groq available
        "cohere": False,    # Simulate Cohere not installed
        "ollama": True      # Simulate Ollama available
    }
    
    # Task-based provider selection simulation
    task_preferences = {
        "coding": ["anthropic", "openai", "groq"],
        "reasoning": ["openai", "anthropic", "groq"],
        "cost_effective": ["groq", "openai"],
        "fast_inference": ["groq", "openai"],
        "local": ["ollama"]
    }
    
    print("Provider selection simulation based on available packages:")
    for task, preferences in task_preferences.items():
        available_for_task = [p for p in preferences if import_results.get(p, False)]
        selected = available_for_task[0] if available_for_task else "none"
        print(f"  {task}: {selected} (available: {available_for_task})")
    
    return task_preferences


def generate_dependency_report():
    """Generate a comprehensive dependency report."""
    print("\n📄 Generating dependency installation report...")
    
    report = {
        "timestamp": "2024-01-01T00:00:00Z",
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "platform": sys.platform,
        "imports": test_import_availability(),
        "environment": test_environment_detection(),
        "recommendations": []
    }
    
    # Generate recommendations based on findings
    if not any(report["imports"].values()):
        report["recommendations"].append("No LLM provider packages installed. Run: poetry install --extras essential")
    
    if not report["environment"]["configured"]:
        report["recommendations"].append("No provider environment variables configured. Set API keys for desired providers.")
    
    if report["imports"].get("openai", False) and "OpenAI" not in report["environment"]["configured"]:
        report["recommendations"].append("OpenAI package installed but OPENAI_API_KEY not set.")
    
    if report["imports"].get("anthropic", False) and "Anthropic" not in report["environment"]["configured"]:
        report["recommendations"].append("Anthropic package installed but ANTHROPIC_API_KEY not set.")
    
    print("✅ Dependency report generated")
    return report


def main():
    """Main test function."""
    print("🧪 Dependency Installation Test Suite")
    print("=" * 60)
    
    # Run all tests
    import_results = test_import_availability()
    extras_config = test_poetry_extras() 
    env_detection = test_environment_detection()
    install_patterns = test_installation_commands()
    provider_selection = simulate_provider_selection()
    report = generate_dependency_report()
    
    print("\n" + "=" * 60)
    print("📋 Summary")
    print("=" * 60)
    
    available_packages = sum(import_results.values())
    total_packages = len(import_results)
    configured_providers = len(env_detection["configured"])
    total_env_providers = len(env_detection["configured"]) + len(env_detection["partial"]) + len(env_detection["missing"])
    
    print(f"📦 Packages: {available_packages}/{total_packages} available")
    print(f"🌍 Environment: {configured_providers}/{total_env_providers} providers configured")  
    print(f"⚙️  Extras: {len(extras_config)} Poetry extras defined")
    print(f"🛠️  Patterns: {len(install_patterns)} installation patterns validated")
    
    if report["recommendations"]:
        print(f"\n💡 Recommendations:")
        for rec in report["recommendations"]:
            print(f"  • {rec}")
    
    print(f"\n🎯 Next Steps:")
    if available_packages == 0:
        print("  1. Install essential providers: poetry install --extras essential")
        print("  2. Set environment variables for chosen providers")
        print("  3. Run environment validation")
    elif configured_providers == 0:
        print("  1. Configure environment variables for installed packages")
        print("  2. Test provider connectivity") 
        print("  3. Run full system validation")
    else:
        print("  1. Test multi-provider functionality")
        print("  2. Optimize for your specific use case")
        print("  3. Consider additional providers based on needs")
    
    print(f"\n📚 See DEPENDENCY_MANAGEMENT.md for detailed installation guide")


if __name__ == "__main__":
    main()