# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is MLE-STAR (Machine Learning Engineering Agent via Search and Targeted Refinement), a multi-agent system for automated machine learning engineering. The agent can train state-of-the-art ML models on various tasks (classification and regression) through web search and targeted code refinement.

## Development Commands

### Environment Setup
```bash
# Install dependencies with Poetry
poetry install

# Install with development dependencies
poetry install --with dev

# Install with deployment dependencies
poetry install --with deployment

# Activate Poetry environment
poetry env activate
# or
source $(poetry env info --path)/bin/activate
```

### Running the Agent
```bash
# Run via ADK CLI
adk run machine_learning_engineering
# or via Poetry
poetry run adk run machine_learning_engineering

# Run web interface
adk web
```

### Testing
```bash
# Run unit tests
python3 -m pytest tests

# Run evaluation tests
python3 -m pytest eval

# Run specific test
python3 -m pytest tests/test_agents.py::test_happy_path
```

### Code Quality
```bash
# Format code with Black
black .

# Run type checking (if available)
python -m mypy machine_learning_engineering/
```

## Architecture Overview

### Multi-Agent Pipeline
The system follows a sequential multi-agent architecture:

1. **Initialization Agent** (`machine_learning_engineering/sub_agents/initialization/`)
   - Task summarization
   - Web search for state-of-the-art models
   - Initial solution generation and ranking
   - Code integration/merging

2. **Refinement Agent** (`machine_learning_engineering/sub_agents/refinement/`)
   - Iterative code improvement through targeted refinement
   - Performance-based code block targeting
   - Inner/outer loop optimization

3. **Ensemble Agent** (`machine_learning_engineering/sub_agents/ensemble/`)
   - Ensemble strategy generation
   - Multiple model combination

4. **Submission Agent** (`machine_learning_engineering/sub_agents/submission/`)
   - Final submission preparation
   - Output formatting

### Key Components

- **Root Agent** (`machine_learning_engineering/agent.py`): Main orchestrator with frontdoor agent and pipeline sequencing
- **Configuration** (`machine_learning_engineering/shared_libraries/config.py`): Centralized configuration management
- **Shared Libraries** (`machine_learning_engineering/shared_libraries/`):
  - `code_util.py`: Python code execution utilities
  - `debug_util.py`: Debug and error correction agents
  - `common_util.py`: General utility functions
  - `check_leakage_util.py`: Data leakage prevention

### Task Structure
Tasks are stored in `machine_learning_engineering/tasks/` with:
- `task_description.txt`: Problem description
- Data files (train.csv, test.csv, etc.)

## Configuration

Required environment variables:
```bash
export GOOGLE_GENAI_USE_VERTEXAI=true
export GOOGLE_CLOUD_PROJECT=<your-project-id>
export GOOGLE_CLOUD_LOCATION=<your-project-location>
export ROOT_AGENT_MODEL=<Google LLM to use>
export GOOGLE_CLOUD_STORAGE_BUCKET=<your-storage-bucket>  # For deployment only
```

Authentication:
```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project $GOOGLE_CLOUD_PROJECT
```

## Key Configuration Parameters

- `num_solutions`: Number of parallel solution generations (default: 2)
- `num_model_candidates`: Model architectures to consider (default: 2)
- `max_retry`: Maximum retries for failed operations (default: 10)
- `max_debug_round`: Debug iteration limit (default: 5)
- `inner_loop_round`: Inner refinement iterations (default: 1)
- `outer_loop_round`: Outer refinement iterations (default: 1)
- `exec_timeout`: Code execution timeout in seconds (default: 600)

## Deployment

Deploy to Vertex AI Agent Engine:
```bash
poetry install --with deployment
python3 deployment/deploy.py --create

# List deployed agents
python3 deployment/deploy.py --list

# Test deployment
export USER_ID=<any string>
python3 deployment/test_deployment.py --resource_id=${AGENT_ENGINE_ID} --user_id=${USER_ID}

# Delete deployment
python3 deployment/deploy.py --delete --resource_id=${AGENT_ENGINE_ID}
```

## Working with Tasks

To add a new ML task:
1. Create folder under `machine_learning_engineering/tasks/`
2. Add `task_description.txt` with problem description
3. Place data files in the task folder
4. Update `task_name` in config if needed

The workspace directory (`machine_learning_engineering/workspace/`) contains execution outputs and intermediate results.

## Dependencies

- Python 3.12+
- Google ADK for agent framework
- Google GenAI/Vertex AI for LLM integration
- PyTorch (CPU version by default)
- scikit-learn, pandas, numpy for ML
- Poetry for dependency management