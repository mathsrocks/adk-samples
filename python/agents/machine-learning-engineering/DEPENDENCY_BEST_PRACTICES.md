# Dependency Management Best Practices

This guide outlines best practices for managing LLM provider dependencies in the MLE-STAR project using Poetry's optional dependencies and extras system.

## 🎯 Core Principles

### 1. **Minimal Core, Optional Extensions**
- Keep core dependencies minimal for faster installation
- Make all LLM providers optional to avoid bloat
- Allow users to install only what they need

### 2. **Use Case Driven Organization**
- Group dependencies by actual usage patterns
- Provide bundles for common scenarios
- Enable selective installation based on requirements

### 3. **Version Compatibility**
- Pin compatible version ranges for stability
- Test major version updates before releasing
- Document breaking changes and migration paths

## 📦 Dependency Organization Strategy

### Core Dependencies (Always Installed)
```toml
[tool.poetry.dependencies]
python = "^3.12"
google-adk = "^1.5.0"          # Framework foundation
google-genai = "^1.9.0"        # Default provider (backward compatibility)
requests = "^2.31.0"           # HTTP client for providers
pydantic = "^2.10.6"          # Data validation
python-dotenv = "^1.0.1"      # Environment variable management
```

### Optional Provider Dependencies
```toml
# Primary LLM Providers  
openai = {version = "^1.50.0", optional = true}
anthropic = {version = "^0.40.0", optional = true}

# Specialized Providers
groq = {version = "^0.13.0", optional = true}
cohere = {version = "^5.18.0", optional = true}
```

### Poetry Extras Strategy
```toml
[tool.poetry.extras]
# Individual providers
openai = ["openai"]
anthropic = ["anthropic"]

# Use case bundles
essential = ["openai", "anthropic", "groq"]
professional = ["openai", "anthropic", "groq", "cohere", "mistralai", "boto3"]
```

## 🔧 Installation Best Practices

### For End Users

#### 1. **Start with Essential Bundle**
```bash
# Recommended for most users
poetry install --extras essential
```
- Includes most commonly used providers
- Balanced between functionality and dependency size
- Good starting point for experimentation

#### 2. **Add Providers as Needed**
```bash
# Start minimal
poetry install --extras openai

# Add more providers later
poetry install --extras "openai anthropic"

# Upgrade to bundle
poetry install --extras professional
```

#### 3. **Use Task-Specific Bundles**
```bash
# For coding tasks
poetry install --extras coding

# For research work
poetry install --extras research

# For enterprise deployment
poetry install --extras enterprise-full
```

### For Development Teams

#### 1. **Standardize Team Setup**
```bash
# Create shared setup script
echo "poetry install --extras professional" > setup.sh
```

#### 2. **Environment-Specific Installations**
```bash
# Development environment
poetry install --extras "development optimization" --with dev

# Production environment  
poetry install --extras essential --without dev --without deployment

# CI/CD environment
poetry install --extras "essential optimization" --without dev
```

#### 3. **Document Team Requirements**
```markdown
## Required Setup
- Essential providers: `poetry install --extras essential`
- Set environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- Verify setup: `python test_dependency_installation.py`
```

## 🏗️ Architecture Best Practices

### 1. **Provider Abstraction**
```python
# ✅ Good: Use factory pattern
from machine_learning_engineering.shared_libraries.agent_factory import get_agent_factory
factory = get_agent_factory()
agent = factory.create_agent(name, instruction)

# ❌ Avoid: Direct provider imports in core logic  
import openai  # Don't do this in core modules
```

### 2. **Graceful Degradation**
```python
# ✅ Good: Handle missing dependencies gracefully
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

def create_openai_client():
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not installed. Run: poetry install --extras openai")
```

### 3. **Environment Detection**
```python
# ✅ Good: Auto-detect available providers
from machine_learning_engineering.shared_libraries.environment_manager import detect_available_providers
available = detect_available_providers()
```

## 📋 Version Management

### 1. **Semantic Versioning**
```toml
# ✅ Recommended: Use caret ranges for stability
openai = {version = "^1.50.0", optional = true}    # 1.50.0 <= version < 2.0.0
anthropic = {version = "^0.40.0", optional = true}  # 0.40.0 <= version < 1.0.0

# ❌ Avoid: Too restrictive
openai = {version = "1.50.0", optional = true}      # Exact version only

# ❌ Avoid: Too permissive  
openai = {version = "*", optional = true}           # Any version
```

### 2. **Regular Updates**
```bash
# Check for updates
poetry show --outdated

# Update specific provider
poetry add openai@^1.55.0 --optional

# Update all dependencies
poetry update
```

### 3. **Breaking Change Management**
- Test major version updates in separate branch
- Document migration requirements
- Provide backward compatibility period
- Update installation guides

## 🧪 Testing Strategy

### 1. **Dependency Matrix Testing**
```yaml
# Example GitHub Actions matrix
strategy:
  matrix:
    extras:
      - "essential"
      - "professional" 
      - "research"
      - "development"
```

### 2. **Import Testing**
```python
# Test that extras work correctly
@pytest.mark.parametrize("provider", ["openai", "anthropic", "groq"])
def test_provider_import(provider):
    # Test imports only if package is installed
    pass
```

### 3. **Environment Testing**
```bash
# Test different installation combinations
poetry install --extras essential
python test_dependency_installation.py

poetry install --extras professional  
python test_dependency_installation.py
```

## 🚀 Performance Optimization

### 1. **Lazy Loading**
```python
# ✅ Good: Import providers only when needed
def get_openai_client():
    if not hasattr(get_openai_client, '_client'):
        from openai import OpenAI
        get_openai_client._client = OpenAI()
    return get_openai_client._client
```

### 2. **Optional Feature Detection**
```python
# ✅ Good: Feature flags based on availability
FEATURES = {
    'openai_support': False,
    'anthropic_support': False,
    'local_inference': False
}

try:
    import openai
    FEATURES['openai_support'] = True
except ImportError:
    pass
```

### 3. **Dependency Grouping**
```toml
# Group related dependencies together
huggingface = ["huggingface-hub", "transformers", "torch"]  # Complete HF stack
optimization = ["optimum", "bitsandbytes", "accelerate"]    # Optimization tools
```

## 📚 Documentation Standards

### 1. **Clear Installation Instructions**
```markdown
## Quick Start
```bash
# Essential setup (recommended)
poetry install --extras essential

# Professional setup
poetry install --extras professional
```

### 2. **Dependency Explanations**
```markdown
## Provider Packages
- `openai`: OpenAI GPT-4, GPT-4o access
- `anthropic`: Claude 3.5 Sonnet, Claude 3 Opus
- `groq`: Fast inference with Llama, Mixtral
```

### 3. **Troubleshooting Guide**
```markdown
## Common Issues
1. **Import Error**: Install missing provider with `poetry install --extras provider-name`
2. **Version Conflict**: Check compatibility with `poetry check`
3. **Environment Missing**: Set required API keys
```

## 🔒 Security Considerations

### 1. **Dependency Security**
```bash
# Regular security audits
poetry audit

# Check for vulnerabilities
pip-audit  # If available
```

### 2. **API Key Management**
```bash
# ✅ Good: Use environment variables
export OPENAI_API_KEY="sk-..."

# ❌ Avoid: Hardcoding in dependencies
# Never put API keys in pyproject.toml
```

### 3. **Supply Chain Security**
- Pin dependency versions in production
- Use Poetry's lock file (`poetry.lock`)
- Regularly update dependencies
- Monitor security advisories

## 🔄 Migration Strategies

### 1. **Adding New Providers**
```toml
# Add to pyproject.toml
new-provider = {version = "^1.0.0", optional = true}

# Add to extras
[tool.poetry.extras]
new-provider = ["new-provider"]
professional = ["openai", "anthropic", "new-provider"]  # Update bundles
```

### 2. **Deprecating Providers**
```toml
# Mark as deprecated in comments
old-provider = {version = "^0.5.0", optional = true}  # DEPRECATED: Use new-provider instead
```

### 3. **Version Upgrades**
```bash
# Test upgrade in isolation
poetry add provider@^2.0.0 --optional

# Update extras if needed
# Test compatibility
# Update documentation
```

## 📊 Monitoring and Maintenance

### 1. **Dependency Health**
- Monitor download statistics
- Track issue reports for specific providers
- Assess community adoption and support

### 2. **Usage Analytics**
```python
# Optional: Track which providers are actually used
def log_provider_usage(provider_name):
    # Log for analytics (privacy compliant)
    pass
```

### 3. **Regular Reviews**
- Quarterly dependency review
- Annual provider ecosystem assessment  
- User feedback incorporation

## 🎯 Success Metrics

### 1. **Installation Success Rate**
- Measure successful `poetry install` completion
- Track common installation failures
- Monitor dependency resolution time

### 2. **Developer Experience**
- Setup time for new team members
- Frequency of dependency-related issues
- User satisfaction with provider selection

### 3. **System Performance**
- Installation size optimization
- Import time measurement
- Runtime performance impact

By following these best practices, teams can maintain a clean, performant, and user-friendly dependency management system that scales with the project's needs while providing flexibility for different use cases and deployment scenarios.