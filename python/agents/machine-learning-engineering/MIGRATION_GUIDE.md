# MLE-STAR Multi-Provider Migration Guide

This guide provides clear upgrade paths for migrating from the original Google-only MLE-STAR implementation to the enhanced multi-provider system while maintaining full backward compatibility.

## Migration Overview

The MLE-STAR system now supports multiple LLM providers while maintaining 100% backward compatibility with existing Google ADK and Vertex AI deployments. You can adopt new features gradually without breaking existing workflows.

## Compatibility Modes

### 1. STRICT_ADK (Default for existing deployments)
- **Description**: Original Google-only behavior, fully compatible with existing deployments
- **Use case**: Existing production deployments that don't need multi-provider features
- **Requirements**: Google Cloud environment variables only
- **Features**: Original MLE-STAR functionality with Google models

### 2. ENHANCED_ADK (Recommended upgrade)
- **Description**: Google ADK compatible with optional multi-provider support
- **Use case**: Gradual adoption of multi-provider features while maintaining ADK compatibility
- **Requirements**: Google Cloud setup + optional provider API keys
- **Features**: Original functionality + optional cost optimization and performance routing

### 3. GRADUAL_MIGRATION
- **Description**: Gradual transition to multi-provider with migration utilities
- **Use case**: Large deployments that need careful migration planning
- **Requirements**: Google Cloud setup + migration planning
- **Features**: Step-by-step migration with rollback capabilities

### 4. FULL_MULTI_PROVIDER
- **Description**: Full multi-provider capabilities with all optimization features
- **Use case**: New deployments or fully migrated systems
- **Requirements**: Multiple provider API keys + enhanced configuration
- **Features**: All multi-provider features including cost optimization and performance routing

## Quick Start Migration Paths

### Path 1: Zero-Change Upgrade (Recommended for most users)

**Current State**: Existing MLE-STAR deployment with Google models  
**Target**: Enhanced capabilities with zero changes to existing code  
**Risk Level**: Low  
**Effort**: Minimal (5 minutes)

```bash
# 1. Update dependencies (optional providers)
poetry install --extras essential

# 2. Set optional provider API keys (if desired)
export OPENAI_API_KEY="your-key-here"        # Optional
export ANTHROPIC_API_KEY="your-key-here"     # Optional
export DEEPSEEK_API_KEY="your-key-here"      # Optional

# 3. Your existing code continues to work unchanged
# No code changes required - enhanced features are opt-in
```

**What you get**:
- Existing functionality works exactly as before
- Optional access to cost optimization features
- Automatic DeepSeek routing for budget-conscious tasks
- Fallback chains for improved reliability

### Path 2: Cost Optimization Upgrade

**Current State**: Working MLE-STAR deployment  
**Target**: Cost savings with DeepSeek integration  
**Risk Level**: Low  
**Effort**: Moderate (30 minutes)

```bash
# 1. Install cost-effective providers
poetry install --extras "deepseek groq"

# 2. Set up cost-effective provider keys
export DEEPSEEK_API_KEY="your-deepseek-key"
export GROQ_API_KEY="your-groq-key"         # Optional for fast inference

# 3. Update agent creation for cost savings (optional)
```

**Code changes** (optional - existing code still works):
```python
# Before (still works)
from google.adk import agents
agent = agents.Agent(model="gemini-1.5-pro", name="my_agent", instruction="...")

# After (for cost optimization)
from machine_learning_engineering.shared_libraries.agent_factory import create_budget_conscious_agent
agent = create_budget_conscious_agent(
    name="my_agent", 
    instruction="...",
    task_type="coding"  # Automatically routes to DeepSeek for cost savings
)
```

**What you get**:
- Significant cost savings (up to 99% reduction with DeepSeek)
- Quality thresholds maintained
- Automatic fallback to Google if needed

### Path 3: Full Multi-Provider Upgrade

**Current State**: Any MLE-STAR deployment  
**Target**: Full multi-provider capabilities  
**Risk Level**: Medium  
**Effort**: Significant (2-4 hours)

```bash
# 1. Install all providers
poetry install --extras professional

# 2. Set up multiple provider API keys
export OPENAI_API_KEY="your-openai-key"
export ANTHROPIC_API_KEY="your-anthropic-key"
export DEEPSEEK_API_KEY="your-deepseek-key"
export GROQ_API_KEY="your-groq-key"

# 3. Configure provider strategy
export PROVIDER_STRATEGY="cost_optimized"
```

**Enhanced agent creation**:
```python
from machine_learning_engineering.shared_libraries.agent_factory import (
    create_coding_optimized_agent,
    create_reasoning_optimized_agent,
    create_budget_conscious_agent
)

# Task-specific optimization
coding_agent = create_coding_optimized_agent(
    name="coding_agent",
    instruction="Generate high-quality code",
    cost_priority=True  # Will prefer DeepSeek for cost savings
)

reasoning_agent = create_reasoning_optimized_agent(
    name="reasoning_agent", 
    instruction="Solve complex problems",
    # Automatically uses best reasoning model (OpenAI o1-mini)
)
```

**What you get**:
- Intelligent model routing based on task type
- Cost optimization with quality thresholds
- Performance-based provider selection
- Comprehensive fallback chains
- Monthly cost estimation and budget controls

## Migration Utilities

### Configuration Validator

Check your current setup and get migration recommendations:

```python
from machine_learning_engineering.shared_libraries.backward_compatibility import get_migration_utilities

utilities = get_migration_utilities()
validator = utilities["configuration_validator"]

# Check current configuration
status = validator()
print(f"Legacy config valid: {status['legacy_config_valid']}")
print(f"Compatibility mode: {status['compatibility_mode']}")
print(f"Recommendations: {status['recommendations']}")
```

### Environment Checker

Verify your environment is ready for migration:

```python
env_checker = utilities["environment_checker"]
env_status = env_checker()

print("Google Environment:")
for var, value in env_status["google_environment"].items():
    print(f"  {var}: {value}")

print("Provider Environment:")
for var, enabled in env_status["provider_environment"].items():
    print(f"  {var}: {'✓' if enabled else '✗'}")

print(f"Migration ready: {env_status['migration_ready']}")
```

### Migration Assistant

Get a personalized migration plan:

```python
migration_assistant = utilities["migration_assistant"]
plan = migration_assistant(target_mode="enhanced_adk")

print(f"Current mode: {plan['migration_plan']['current_mode']}")
print(f"Target mode: {plan['migration_plan']['target_mode']}")
print(f"Risk level: {plan['migration_plan']['risk_level']}")

print("Steps:")
for step in plan["migration_plan"]["steps"]:
    print(f"  {step}")

print("Next steps:")
for step in plan["next_steps"]:
    print(f"  {step}")
```

### Compatibility Tester

Test that existing patterns still work:

```python
compatibility_tester = utilities["compatibility_tester"]
test_results = compatibility_tester()

print(f"Overall compatibility: {test_results['overall_compatibility']}")
print(f"Legacy agent creation: {test_results['legacy_agent_creation']}")
print(f"Generate content config: {test_results['generate_content_config']}")

if test_results["error_messages"]:
    print("Issues found:")
    for error in test_results["error_messages"]:
        print(f"  ⚠️ {error}")
```

## Deployment Pipeline Compatibility

### Vertex AI Deployment

The multi-provider system maintains full compatibility with Vertex AI deployments:

```bash
# Required Vertex AI environment variables (unchanged)
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export ROOT_AGENT_MODEL=gemini-1.5-pro

# Optional: Add multi-provider support
export DEEPSEEK_API_KEY=your-deepseek-key  # For cost optimization

# Deploy using existing pipeline
python3 deployment/deploy.py --create
```

**Verification**:
```python
from machine_learning_engineering.shared_libraries.backward_compatibility import ensure_deployment_compatibility

# Verify deployment compatibility
is_compatible = ensure_deployment_compatibility()
print(f"Deployment compatible: {is_compatible}")
```

### Docker Deployment

Update your Dockerfile to include optional dependencies:

```dockerfile
# Existing Dockerfile works unchanged
FROM python:3.12-slim
COPY . /app
WORKDIR /app
RUN poetry install

# Optional: Add multi-provider support
# RUN poetry install --extras professional

# Environment variables
ENV GOOGLE_GENAI_USE_VERTEXAI=true
ENV GOOGLE_CLOUD_PROJECT=your-project
# Optional multi-provider keys
# ENV DEEPSEEK_API_KEY=your-key
```

## Rollback Plan

If you encounter issues, you can easily rollback:

### Step 1: Remove Multi-Provider Environment Variables
```bash
unset OPENAI_API_KEY
unset ANTHROPIC_API_KEY
unset DEEPSEEK_API_KEY
unset PROVIDER_STRATEGY
```

### Step 2: Ensure Google Environment is Set
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=your-project-id
export GOOGLE_CLOUD_LOCATION=us-central1
export ROOT_AGENT_MODEL=gemini-1.5-pro
```

### Step 3: Force Strict ADK Mode (if needed)
```bash
export COMPATIBILITY_MODE=strict_adk
```

### Step 4: Restart Services
```bash
# Restart your application to pick up environment changes
```

### Step 5: Verify Original Functionality
```python
from machine_learning_engineering.shared_libraries.backward_compatibility import get_backward_compatibility_manager

manager = get_backward_compatibility_manager()
status = manager.get_compatibility_status()
print(f"Compatibility mode: {status['compatibility_mode']}")
print(f"Legacy config valid: {status['legacy_config_valid']}")
```

## Troubleshooting

### Common Issues

**Issue**: "Agent creation failed with multi-provider features"  
**Solution**: The system automatically falls back to Google-only mode. Check provider API keys and network connectivity.

**Issue**: "Vertex AI deployment not working with multi-provider"  
**Solution**: Ensure required Google environment variables are set. Multi-provider features are optional and won't break Vertex AI deployment.

**Issue**: "Cost optimization not working"  
**Solution**: Verify DEEPSEEK_API_KEY is set and valid. The system falls back to Google models if DeepSeek is unavailable.

**Issue**: "Performance degradation after upgrade"  
**Solution**: The system maintains original performance. Use strict ADK mode if needed: `export COMPATIBILITY_MODE=strict_adk`

### Getting Help

1. **Configuration Issues**: Use the migration utilities to diagnose problems
2. **API Key Problems**: Verify keys are valid and have sufficient quota
3. **Deployment Issues**: Check that required Google environment variables are set
4. **Performance Issues**: Use strict ADK mode for original behavior

## Best Practices

### For Production Deployments

1. **Start with ENHANCED_ADK mode**: Maintains ADK compatibility while adding optional features
2. **Test thoroughly**: Use compatibility tester before deploying
3. **Gradual adoption**: Enable multi-provider features incrementally
4. **Monitor costs**: Use cost estimation utilities to track spending
5. **Have a rollback plan**: Keep Google environment variables properly configured

### For Development

1. **Use FULL_MULTI_PROVIDER mode**: Take advantage of all features
2. **Set up multiple providers**: Get the best model for each task type
3. **Use cost optimization**: Save on development costs with DeepSeek
4. **Experiment with routing**: Try different task-specific optimizations

### For Budget-Conscious Deployments

1. **Install DeepSeek provider**: `poetry install --extras deepseek`
2. **Use budget-conscious agents**: `create_budget_conscious_agent()`
3. **Set cost budgets**: Configure monthly spending limits
4. **Monitor usage**: Use cost estimation features

## Migration Checklist

### Pre-Migration
- [ ] Verify current deployment is working
- [ ] Run configuration validator
- [ ] Check environment variables
- [ ] Backup current configuration
- [ ] Plan maintenance window (if needed)

### Migration
- [ ] Install additional dependencies
- [ ] Set provider API keys (optional)
- [ ] Run compatibility tests
- [ ] Update agent creation (optional)
- [ ] Test new features

### Post-Migration
- [ ] Verify all existing functionality works
- [ ] Test new multi-provider features
- [ ] Monitor performance and costs
- [ ] Update documentation
- [ ] Train team on new features

## Support

For migration support:
1. Use the migration utilities for automated assistance
2. Check the troubleshooting section for common issues
3. Refer to the backward compatibility test results
4. Use strict ADK mode if any issues arise

The migration is designed to be risk-free and reversible. Your existing Google-based deployment will continue to work unchanged, with multi-provider features available as opt-in enhancements.