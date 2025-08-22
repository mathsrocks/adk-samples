# Dependency Management Guide

This guide explains how to install MLE-STAR with different LLM provider combinations using Poetry's optional dependencies and extras system.

## Quick Start

### Essential Installation (Recommended)
Install the most commonly used LLM providers:
```bash
poetry install --extras essential
```
Includes: OpenAI GPT-4, Anthropic Claude, Groq (+ DeepSeek compatibility)

### Individual Provider Installation
Install specific providers as needed:
```bash
# OpenAI GPT-4, GPT-4o, o1
poetry install --extras openai

# Anthropic Claude 3.5 Sonnet, Claude 3 Opus
poetry install --extras anthropic

# DeepSeek (uses OpenAI-compatible API)
poetry install --extras deepseek

# Multiple providers
poetry install --extras "openai anthropic deepseek"
```

## Available Provider Extras

### 🏢 Primary Commercial Providers
| Extra | Provider | Models | Use Case |
|-------|----------|--------|----------|
| `openai` | OpenAI | GPT-4, GPT-4o, o1-preview | General purpose, reasoning |
| `anthropic` | Anthropic | Claude 3.5 Sonnet, Claude 3 Opus | Coding, analysis |
| `deepseek` | DeepSeek | DeepSeek-Chat, DeepSeek-Coder | Cost-effective, coding |

### ⚡ Alternative & Specialized Providers  
| Extra | Provider | Models | Use Case |
|-------|----------|--------|----------|
| `groq` | Groq | Llama 3.3, Mixtral | Fast inference |
| `cohere` | Cohere | Command R+, Command | Enterprise NLP |
| `mistral` | Mistral | Mistral Large, Mixtral | European alternative |
| `together` | Together AI | Open source models | Community models |
| `fireworks` | Fireworks AI | Various models | High-speed inference |
| `perplexity` | Perplexity | Sonar models | Web search integration |

### 🌐 Open Source Platforms
| Extra | Provider | Description | Use Case |
|-------|----------|-------------|----------|
| `huggingface` | Hugging Face | Transformers ecosystem | Research, fine-tuning |
| `replicate` | Replicate | Model hosting platform | Experimentation |
| `local` | Local inference | Ollama + transformers | Offline development |

### ☁️ Enterprise & Cloud
| Extra | Provider | Description | Use Case |
|-------|----------|-------------|----------|
| `aws` | AWS Bedrock | Enterprise AI service | Corporate deployment |
| `azure` | Azure OpenAI | Microsoft cloud AI | Enterprise Microsoft stack |
| `google-extended` | Google AI Platform | Extended Google features | Advanced Google integration |

### 🔧 Advanced Features
| Extra | Description | Packages | Use Case |
|-------|-------------|----------|----------|
| `optimization` | Model optimization | optimum, bitsandbytes, accelerate | Performance tuning |
| `proxy` | Universal LLM proxy | litellm | Multi-provider abstraction |
| `langchain` | LangChain integration | langchain, langchain-community | Framework integration |

## Installation Bundles

### 📦 Pre-configured Bundles

#### Essential Bundle
Most commonly used providers for general development:
```bash
poetry install --extras essential
```
**Includes:** OpenAI, Anthropic, Groq (+ DeepSeek compatibility)

#### Professional Bundle  
Production-ready setup with enterprise providers:
```bash
poetry install --extras professional
```
**Includes:** OpenAI, Anthropic, Groq, Cohere, Mistral, AWS Bedrock, Azure OpenAI

#### Research Bundle
For ML research and experimentation:
```bash
poetry install --extras research  
```
**Includes:** OpenAI, Anthropic, Hugging Face, Transformers, PyTorch, Perplexity, Replicate

#### Development Bundle
For development and testing:
```bash
poetry install --extras development
```
**Includes:** OpenAI, Anthropic, Groq, Ollama, LiteLLM

#### Local Bundle
For offline and self-hosted setups:
```bash
poetry install --extras local-full
```
**Includes:** Ollama, Hugging Face, Transformers, PyTorch, Accelerate, Optimum, BitsAndBytes

#### Enterprise Bundle
For enterprise deployment:
```bash
poetry install --extras enterprise-full
```
**Includes:** OpenAI, Anthropic, AWS Bedrock, Azure OpenAI + AI Services, Google AI Platform, LiteLLM

### 🎯 Use Case Based Installation

#### By Performance Characteristics
```bash
# Fast inference optimized
poetry install --extras fast-inference

# Cost-effective providers  
poetry install --extras cost-effective

# High-quality premium models
poetry install --extras high-quality

# Large context window support
poetry install --extras long-context
```

#### By Business Model
```bash
# Commercial APIs only
poetry install --extras commercial

# Open source models only
poetry install --extras open-source

# Providers with free tiers
poetry install --extras freemium
```

#### By Deployment Type
```bash
# Cloud-hosted providers
poetry install --extras cloud-hosted

# Self-hosted solutions
poetry install --extras self-hosted

# Enterprise-grade providers
poetry install --extras enterprise
```

#### By Task Specialization
```bash
# Best for coding tasks
poetry install --extras coding

# Best for reasoning tasks
poetry install --extras reasoning

# Best for creative tasks
poetry install --extras creative

# With web search capabilities
poetry install --extras research
```

## Advanced Installation Examples

### Custom Combinations
Mix and match providers for specific needs:

```bash
# Coding + Cost-effective
poetry install --extras "anthropic deepseek groq"

# Research + Local development
poetry install --extras "openai huggingface local"

# Enterprise + Optimization
poetry install --extras "professional optimization"

# Everything for development
poetry install --extras "development langchain proxy"
```

### Development vs Production

**Development Setup:**
```bash
poetry install --extras "development optimization" --with dev
```

**Production Setup:**
```bash
poetry install --extras professional --without dev
```

**Research Setup:**
```bash
poetry install --extras "research optimization langchain" --with dev
```

## Environment Variable Setup

After installing providers, configure environment variables:

### OpenAI
```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_ORGANIZATION="org-..."  # Optional
```

### Anthropic Claude
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

### DeepSeek  
```bash
export DEEPSEEK_API_KEY="sk-..."
```

### Groq
```bash
export GROQ_API_KEY="gsk_..."
```

### Google (Existing)
```bash
export GOOGLE_CLOUD_PROJECT="your-project"
export GOOGLE_GENAI_USE_VERTEXAI="true"
```

### Local Setup (Ollama)
```bash
# Install Ollama first: https://ollama.ai
export OLLAMA_HOST="localhost"  # Optional
export OLLAMA_PORT="11434"      # Optional
```

### Enterprise Providers
```bash
# AWS Bedrock
export AWS_ACCESS_KEY_ID="..."
export AWS_SECRET_ACCESS_KEY="..."
export AWS_DEFAULT_REGION="us-east-1"

# Azure OpenAI
export AZURE_OPENAI_API_KEY="..."
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com"
```

## Dependency Management Best Practices

### 1. Start Minimal
Begin with essential providers and add more as needed:
```bash
# Start with essentials
poetry install --extras essential

# Add providers as required
poetry install --extras "essential cohere mistral"
```

### 2. Use Bundles for Common Setups
Leverage pre-configured bundles for typical use cases:
```bash
# For most development work
poetry install --extras professional

# For research projects
poetry install --extras research
```

### 3. Separate Development and Production
Use different dependency sets:
```bash
# Development (with dev tools and extras)
poetry install --extras "development optimization" --with dev

# Production (minimal, optimized)  
poetry install --extras essential --without dev --without deployment
```

### 4. Pin Versions for Production
For production deployments, consider pinning specific versions in pyproject.toml.

### 5. Monitor Dependencies
Keep track of installed packages:
```bash
# List installed packages
poetry show

# Check for updates
poetry show --outdated

# Update specific extra
poetry install --extras openai  # Updates openai package
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the provider extra is installed
   ```bash
   poetry install --extras openai  # If getting openai import errors
   ```

2. **Version Conflicts**: Check compatibility
   ```bash
   poetry check
   poetry install --verbose  # See detailed dependency resolution
   ```

3. **Missing Optional Dependencies**: Install the specific extra
   ```bash
   poetry install --extras "provider-name"
   ```

### Verification
Test your installation:
```python
# Test provider availability
from machine_learning_engineering.shared_libraries.config_bridge import validate_environment_setup
status = validate_environment_setup()
print(f"Status: {status['status']}")
print(f"Available providers: {status['available_providers']}")
```

## Migration Guide

### From Basic to Advanced Setup

1. **Assess Current Usage**
   ```bash
   poetry show | grep -E "(openai|anthropic|groq)"
   ```

2. **Install Additional Providers**
   ```bash
   poetry install --extras "current-extras new-provider"
   ```

3. **Update Environment Variables**
   Add new provider API keys as needed.

4. **Test Configuration**
   ```python
   from machine_learning_engineering.shared_libraries.environment_manager import detect_available_providers
   print(f"Available: {[p.value for p in detect_available_providers()]}")
   ```

The dependency management system is designed to be flexible, allowing you to install exactly what you need while providing convenient bundles for common use cases. Choose the installation method that best fits your specific requirements and use case.