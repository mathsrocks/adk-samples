"""Demonstration of Machine Learning Engineering Agent using Agent Development Kit"""

import os
import json
from typing import Optional
from google.genai import types
from google.adk.agents import callback_context as callback_context_module

from google.adk import agents
from machine_learning_engineering.sub_agents.initialization import agent as initialization_agent_module
from machine_learning_engineering.sub_agents.refinement import agent as refinement_agent_module
from machine_learning_engineering.sub_agents.ensemble import agent as ensemble_agent_module
from machine_learning_engineering.sub_agents.submission import agent as submission_agent_module

from machine_learning_engineering import prompt
from machine_learning_engineering.shared_libraries.agent_factory import get_agent_factory


def save_state(
    callback_context: callback_context_module.CallbackContext
) -> Optional[types.Content]:
    """Prints the current state of the callback context."""
    workspace_dir = callback_context.state.get("workspace_dir", "")
    task_name = callback_context.state.get("task_name", "")
    run_cwd = os.path.join(workspace_dir, task_name)
    with open(os.path.join(run_cwd, "final_state.json"), "w") as f:
        json.dump(callback_context.state.to_dict(), f, indent=2)
    return None


# Create the MLE pipeline using the agent factory
factory = get_agent_factory()
mle_pipeline_agent = factory.create_sequential_agent(
    name="mle_pipeline_agent",
    sub_agents=[
        initialization_agent_module.initialization_agent,
        refinement_agent_module.refinement_agent,
        ensemble_agent_module.ensemble_agent,
        submission_agent_module.submission_agent,
    ],
    description="Executes a sequence of sub-agents for solving the MLE task.",
    after_agent_callback=save_state,
)

# For ADK tools compatibility, the root agent must be named `root_agent`
# Create root agent using the multi-provider factory
root_agent = factory.create_agent(
    name="mle_frontdoor_agent",
    instruction=prompt.FRONTDOOR_INSTRUCTION,
    global_instruction=prompt.SYSTEM_INSTRUCTION,
    sub_agents=[mle_pipeline_agent],
    description="Main frontdoor agent for MLE-STAR system",
    temperature=0.01,
    task_type="reasoning",  # Root agent handles high-level reasoning
)
