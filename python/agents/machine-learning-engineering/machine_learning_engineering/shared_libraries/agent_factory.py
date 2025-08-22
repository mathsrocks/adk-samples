"""Multi-provider agent factory for creating agents with different LLM providers."""

import os
from typing import Optional, Dict, Any, Union, List, Callable
from enum import Enum

from google.adk import agents
from google.genai import types

from .config import CONFIG, ProviderStrategy
from .llm_providers import ProviderType, LLMProviderFactory, MultiProviderLLM
from .config_bridge import ConfiguredProviderFactory
from .model_performance_router import (
    ModelPerformanceRouter, TaskType, get_model_performance_router,
    get_best_coding_model, get_best_reasoning_model, get_most_cost_effective_model
)
from .cost_optimization_strategy import (
    CostOptimizationStrategy, CostOptimizationMode, QualityThreshold, CostBudget,
    get_cost_optimization_strategy, get_budget_conscious_model, get_maximum_savings_model
)


class AgentType(Enum):
    """Types of agents that can be created."""
    STANDARD = "standard"
    SEQUENTIAL = "sequential" 
    PARALLEL = "parallel"
    LOOP = "loop"


class ProviderAgentAdapter:
    """Adapter to create Google ADK compatible agents with different providers."""
    
    @staticmethod
    def create_model_config_for_provider(
        provider_type: str,
        model_name: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Union[str, Dict[str, Any]]:
        """Create model configuration compatible with Google ADK agents."""
        
        # For Google provider, return model name (current ADK pattern)
        if provider_type == "google":
            return model_name or CONFIG.get_model_for_task()
        
        # For other providers, we'll need to adapt to ADK's model parameter format
        # This is a bridge solution until full provider integration
        model_config = {
            "provider": provider_type,
            "model_name": model_name or CONFIG.get_model_for_task(),
            "temperature": temperature,
            "max_tokens": max_tokens,
            **kwargs
        }
        
        # Return the model name for now (maintains ADK compatibility)
        # Future enhancement: extend ADK to support provider objects
        return model_name or CONFIG.get_model_for_task()
    
    @staticmethod
    def create_generate_content_config(
        provider_type: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> types.GenerateContentConfig:
        """Create GenerateContentConfig for the provider."""
        
        # Get provider-specific defaults
        provider_config = CONFIG.provider_configs.get(provider_type)
        
        final_temperature = temperature
        if final_temperature is None and provider_config:
            final_temperature = provider_config.temperature
        if final_temperature is None:
            final_temperature = 0.01
        
        final_max_tokens = max_tokens
        if final_max_tokens is None and provider_config:
            final_max_tokens = provider_config.max_tokens
        
        return types.GenerateContentConfig(
            temperature=final_temperature,
            max_output_tokens=final_max_tokens,
            **kwargs
        )


class MultiProviderAgentFactory:
    """Factory for creating agents with different LLM providers."""
    
    def __init__(
        self,
        default_provider: Optional[str] = None,
        fallback_enabled: bool = True,
        routing_strategy: Optional[Union[str, ProviderStrategy]] = None
    ):
        """Initialize the agent factory.
        
        Args:
            default_provider: Default provider to use (e.g., "google", "openai", "anthropic")
            fallback_enabled: Whether to enable fallback to other providers
            routing_strategy: Strategy for provider selection
        """
        self.default_provider = default_provider or CONFIG.provider_type or "google"
        self.fallback_enabled = fallback_enabled
        self.routing_strategy = routing_strategy or CONFIG.provider_strategy
        
        # Initialize model performance router and cost optimization
        self.performance_router = get_model_performance_router()
        self.cost_optimizer = get_cost_optimization_strategy()
        
        # Provider-specific optimizations
        self.provider_optimizations = {
            "anthropic": {
                "temperature": 0.7,  # Claude works well with higher temperature
                "max_tokens": 4096,
                "description_prefix": "[Claude-optimized]"
            },
            "openai": {
                "temperature": 0.1,  # GPT-4 is more consistent with lower temperature
                "max_tokens": 4096,
                "description_prefix": "[GPT-4-optimized]"
            },
            "deepseek": {
                "temperature": 0.3,  # DeepSeek balance
                "max_tokens": 2048,
                "description_prefix": "[DeepSeek-optimized]"
            },
            "groq": {
                "temperature": 0.5,  # Fast inference optimized
                "max_tokens": 2048,
                "description_prefix": "[Groq-optimized]"
            },
            "google": {
                "temperature": 0.01,  # Current default
                "max_tokens": None,
                "description_prefix": "[Gemini-optimized]"
            }
        }
    
    def create_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        description: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        global_instruction: Optional[Union[str, Callable]] = None,
        sub_agents: Optional[List[agents.Agent]] = None,
        generate_content_config: Optional[types.GenerateContentConfig] = None,
        before_model_callback: Optional[Callable] = None,
        after_model_callback: Optional[Callable] = None,
        **kwargs
    ) -> agents.Agent:
        """Create a standard Agent with the specified provider.
        
        Args:
            name: Agent name
            instruction: Agent instruction (string or callable)
            description: Agent description
            provider: LLM provider to use
            model: Specific model name
            temperature: Temperature setting
            max_tokens: Maximum tokens
            global_instruction: Global instruction
            sub_agents: Sub-agents list
            generate_content_config: Generation configuration
            before_model_callback: Before model callback
            after_model_callback: After model callback
            **kwargs: Additional parameters
        """
        # Determine provider to use
        effective_provider = provider or self._select_provider_for_task(kwargs.get("task_type", "default"))
        
        # Get provider optimizations
        optimizations = self.provider_optimizations.get(effective_provider, {})
        
        # Apply provider-specific optimizations
        final_temperature = temperature or optimizations.get("temperature")
        final_max_tokens = max_tokens or optimizations.get("max_tokens")
        
        # Enhance description with provider info
        final_description = description or f"Agent powered by {effective_provider}"
        prefix = optimizations.get("description_prefix", "")
        if prefix:
            final_description = f"{prefix} {final_description}"
        
        # Create model configuration
        final_model = model or CONFIG.get_model_for_task(kwargs.get("task_type", "default"))
        if not model and effective_provider in CONFIG.model_mapping:
            task_type = kwargs.get("task_type", "default")
            provider_models = CONFIG.model_mapping[effective_provider]
            if task_type in provider_models:
                final_model = provider_models[task_type]
            elif "default" in provider_models:
                final_model = provider_models["default"]
        
        # Create model config (adapted for current ADK compatibility)
        model_config = ProviderAgentAdapter.create_model_config_for_provider(
            effective_provider,
            final_model,
            final_temperature,
            final_max_tokens,
            **kwargs
        )
        
        # Create generate content config if not provided
        if generate_content_config is None:
            generate_content_config = ProviderAgentAdapter.create_generate_content_config(
                effective_provider,
                final_temperature,
                final_max_tokens
            )
        
        # Create the agent
        try:
            agent = agents.Agent(
                model=model_config,
                name=name,
                instruction=instruction,
                description=final_description,
                global_instruction=global_instruction,
                sub_agents=sub_agents or [],
                generate_content_config=generate_content_config,
                before_model_callback=before_model_callback,
                after_model_callback=after_model_callback,
                **kwargs
            )
            
            # Add provider metadata to agent for tracking
            if hasattr(agent, '__dict__'):
                agent._mle_provider = effective_provider
                agent._mle_model = final_model
                agent._mle_factory_created = True
            
            return agent
            
        except Exception as e:
            if self.fallback_enabled and effective_provider != "google":
                # Fallback to Google provider
                print(f"Warning: Failed to create agent with {effective_provider}, falling back to Google: {e}")
                return self.create_agent(
                    name=name,
                    instruction=instruction,
                    description=description,
                    provider="google",
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    global_instruction=global_instruction,
                    sub_agents=sub_agents,
                    generate_content_config=generate_content_config,
                    before_model_callback=before_model_callback,
                    after_model_callback=after_model_callback,
                    **kwargs
                )
            else:
                raise e
    
    def create_sequential_agent(
        self,
        name: str,
        sub_agents: List[agents.Agent],
        description: Optional[str] = None,
        before_agent_callback: Optional[Callable] = None,
        after_agent_callback: Optional[Callable] = None,
        **kwargs
    ) -> agents.SequentialAgent:
        """Create a SequentialAgent."""
        return agents.SequentialAgent(
            name=name,
            sub_agents=sub_agents,
            description=description or f"Sequential agent with {len(sub_agents)} sub-agents",
            before_agent_callback=before_agent_callback,
            after_agent_callback=after_agent_callback,
            **kwargs
        )
    
    def create_parallel_agent(
        self,
        name: str,
        sub_agents: List[agents.Agent],
        description: Optional[str] = None,
        **kwargs
    ) -> agents.ParallelAgent:
        """Create a ParallelAgent."""
        return agents.ParallelAgent(
            name=name,
            sub_agents=sub_agents,
            description=description or f"Parallel agent with {len(sub_agents)} sub-agents",
            **kwargs
        )
    
    def create_loop_agent(
        self,
        name: str,
        sub_agent: agents.Agent,
        max_iterations: int,
        description: Optional[str] = None,
        **kwargs
    ) -> agents.LoopAgent:
        """Create a LoopAgent."""
        return agents.LoopAgent(
            name=name,
            sub_agent=sub_agent,
            max_iterations=max_iterations,
            description=description or f"Loop agent with max {max_iterations} iterations",
            **kwargs
        )
    
    def _select_provider_for_task(self, task_type: str) -> str:
        """Select the best provider for a specific task type using performance routing."""
        
        # Map task_type strings to TaskType enums for performance router
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "summarization": TaskType.CREATIVE,
            "fast_inference": TaskType.COST_EFFECTIVE,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "research": TaskType.RESEARCH,
        }
        
        # Get available providers from environment
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue  # Skip unknown providers
        
        # Use performance router if we have a mapped task type
        mapped_task_type = task_type_mapping.get(task_type)
        if mapped_task_type and available_providers:
            
            # Get the best model based on performance routing
            cost_priority = (task_type == "cost_effective" or 
                           CONFIG.provider_strategy == ProviderStrategy.COST_OPTIMIZED)
            
            best_selection = self.performance_router.get_best_model_for_task(
                mapped_task_type,
                available_providers,
                cost_priority=cost_priority
            )
            
            if best_selection:
                provider, model = best_selection
                return provider.value
        
        # Fallback to legacy task-specific provider preferences
        task_preferences = {
            "coding": ["anthropic", "deepseek", "openai", "google"],
            "reasoning": ["openai", "anthropic", "google", "deepseek"], 
            "debugging": ["anthropic", "openai", "google", "deepseek"],
            "summarization": ["anthropic", "openai", "google"],
            "fast_inference": ["groq", "deepseek", "google"],
            "cost_effective": ["deepseek", "groq", "ollama"],
            "default": [self.default_provider]
        }
        
        preferences = task_preferences.get(task_type, task_preferences["default"])
        
        # Find first available provider from preferences
        for provider in preferences:
            if CONFIG.is_provider_configured(provider):
                return provider
        
        # Fallback to default provider
        return self.default_provider
    
    def create_coding_optimized_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        description: Optional[str] = None,
        cost_priority: bool = False,
        **kwargs
    ) -> agents.Agent:
        """Create an agent optimized for coding tasks using performance routing."""
        
        # Get performance-based recommendations
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get best model for coding
        best_selection = self.performance_router.get_best_model_for_task(
            TaskType.CODING,
            available_providers,
            cost_priority=cost_priority
        )
        
        provider = None
        model = None
        if best_selection:
            provider_type, model_name = best_selection
            provider = provider_type.value
            model = model_name
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=description or f"Coding-optimized agent (using {provider or 'default'})",
            task_type="coding",
            provider=provider,
            model=model,
            temperature=0.3,  # Lower temperature for more consistent code
            **kwargs
        )
    
    def create_reasoning_optimized_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        description: Optional[str] = None,
        cost_priority: bool = False,
        **kwargs
    ) -> agents.Agent:
        """Create an agent optimized for reasoning tasks using performance routing."""
        
        # Get performance-based recommendations
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get best model for reasoning
        best_selection = self.performance_router.get_best_model_for_task(
            TaskType.REASONING,
            available_providers,
            cost_priority=cost_priority
        )
        
        provider = None
        model = None
        if best_selection:
            provider_type, model_name = best_selection
            provider = provider_type.value
            model = model_name
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=description or f"Reasoning-optimized agent (using {provider or 'default'})",
            task_type="reasoning",
            provider=provider,
            model=model,
            temperature=0.1,  # Very low temperature for logical reasoning
            **kwargs
        )
    
    def create_cost_optimized_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        description: Optional[str] = None,
        task_type: Optional[str] = None,
        **kwargs
    ) -> agents.Agent:
        """Create a cost-optimized agent using performance routing."""
        
        # Get performance-based recommendations
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Map to TaskType for cost optimization
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get most cost-effective model for the task
        best_selection = self.performance_router.get_best_model_for_task(
            mapped_task,
            available_providers,
            cost_priority=True  # Always prioritize cost
        )
        
        provider = None
        model = None
        cost_info = ""
        
        if best_selection:
            provider_type, model_name = best_selection
            provider = provider_type.value
            model = model_name
            
            # Get cost estimate for display
            estimated_cost = self.performance_router.estimate_cost_for_task(
                provider_type, model_name, 1000  # 1K tokens
            )
            cost_info = f" (${estimated_cost:.5f}/1K tokens)"
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=description or f"Cost-optimized agent{cost_info}",
            task_type=task_type or "cost_effective",
            provider=provider,
            model=model,
            **kwargs
        )
    
    def create_budget_conscious_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        task_type: str = "default",
        quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE,
        budget: Optional[CostBudget] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> agents.Agent:
        """Create a budget-conscious agent using cost optimization strategy."""
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "default": TaskType.COST_EFFECTIVE
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get available providers
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get cost-optimized selection
        optimization_result = self.cost_optimizer.optimize_model_selection(
            task_type=mapped_task,
            optimization_mode=CostOptimizationMode.BUDGET_CONSCIOUS,
            quality_threshold=quality_threshold,
            budget=budget,
            available_providers=available_providers
        )
        
        # Create enhanced description with cost info
        cost_info = f"${optimization_result.estimated_cost_per_1k:.5f}/1K tokens"
        if optimization_result.estimated_savings_percent:
            cost_info += f", {optimization_result.estimated_savings_percent:.1f}% savings"
        
        final_description = description or f"Budget-conscious agent ({cost_info})"
        if optimization_result.fallback_reason:
            final_description += f" - {optimization_result.fallback_reason}"
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=final_description,
            provider=optimization_result.selected_provider.value,
            model=optimization_result.selected_model,
            task_type=task_type,
            **kwargs
        )
    
    def create_maximum_savings_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        task_type: str = "default",
        budget: Optional[CostBudget] = None,
        description: Optional[str] = None,
        **kwargs
    ) -> agents.Agent:
        """Create an agent optimized for maximum cost savings."""
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "default": TaskType.COST_EFFECTIVE
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get available providers
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get maximum savings selection
        optimization_result = self.cost_optimizer.optimize_model_selection(
            task_type=mapped_task,
            optimization_mode=CostOptimizationMode.MAXIMUM_SAVINGS,
            quality_threshold=QualityThreshold.MINIMUM,
            budget=budget,
            available_providers=available_providers
        )
        
        # Create enhanced description with savings info
        cost_info = f"${optimization_result.estimated_cost_per_1k:.5f}/1K tokens"
        if optimization_result.estimated_savings_percent:
            cost_info += f", {optimization_result.estimated_savings_percent:.1f}% savings"
        
        final_description = description or f"Maximum savings agent ({cost_info})"
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=final_description,
            provider=optimization_result.selected_provider.value,
            model=optimization_result.selected_model,
            task_type=task_type,
            **kwargs
        )
    
    def create_adaptive_cost_agent(
        self,
        name: str,
        instruction: Union[str, Callable],
        task_type: str = "default",
        monthly_budget_usd: Optional[float] = None,
        quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE,
        description: Optional[str] = None,
        **kwargs
    ) -> agents.Agent:
        """Create an agent with adaptive cost optimization based on task complexity."""
        
        # Create budget if monthly limit specified
        budget = None
        if monthly_budget_usd:
            # Estimate max cost per 1K tokens based on monthly budget
            # Assume 1M tokens per month for estimation
            estimated_monthly_tokens = 1000000
            max_cost_per_1k = (monthly_budget_usd / estimated_monthly_tokens) * 1000
            budget = CostBudget(
                max_cost_per_1k_tokens=max_cost_per_1k,
                monthly_budget_usd=monthly_budget_usd
            )
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "default": TaskType.COST_EFFECTIVE
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get available providers
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get adaptive selection
        optimization_result = self.cost_optimizer.optimize_model_selection(
            task_type=mapped_task,
            optimization_mode=CostOptimizationMode.ADAPTIVE,
            quality_threshold=quality_threshold,
            budget=budget,
            available_providers=available_providers
        )
        
        # Create enhanced description
        budget_info = ""
        if monthly_budget_usd:
            budget_info = f" (Monthly budget: ${monthly_budget_usd})"
        
        final_description = description or f"Adaptive cost agent{budget_info}"
        
        return self.create_agent(
            name=name,
            instruction=instruction,
            description=final_description,
            provider=optimization_result.selected_provider.value,
            model=optimization_result.selected_model,
            task_type=task_type,
            **kwargs
        )
    
    def get_cost_optimization_recommendations(
        self,
        task_type: str,
        current_monthly_cost: float
    ) -> Dict[str, Any]:
        """Get cost optimization recommendations for a task type."""
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "default": TaskType.COST_EFFECTIVE
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get available providers
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        # Get recommendations from cost optimizer
        return self.cost_optimizer.get_cost_optimization_recommendations(
            task_type=mapped_task,
            current_monthly_cost=current_monthly_cost,
            available_providers=available_providers
        )
    
    def get_fallback_chain(
        self,
        task_type: str,
        available_providers: Optional[List[ProviderType]] = None
    ) -> List[str]:
        """Get the fallback chain for a task type (OpenAI → DeepSeek → Claude → Google)."""
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
            "ensemble": TaskType.ENSEMBLE,
            "refinement": TaskType.REFINEMENT,
            "submission": TaskType.SUBMISSION,
            "default": TaskType.COST_EFFECTIVE
        }
        
        mapped_task = task_type_mapping.get(task_type, TaskType.COST_EFFECTIVE)
        
        # Get fallback chain from cost optimizer
        chain_nodes = self.cost_optimizer.get_fallback_chain(mapped_task, available_providers)
        
        # Convert to string representation
        return [f"{node.provider.value}:{node.model_name}" for node in chain_nodes]
    
    def get_performance_recommendations(
        self, 
        task_type: str
    ) -> Dict[str, Any]:
        """Get performance-based recommendations for a task type."""
        
        # Map string to TaskType enum
        task_type_mapping = {
            "coding": TaskType.CODING,
            "reasoning": TaskType.REASONING,
            "debugging": TaskType.DEBUGGING,
            "data_analysis": TaskType.DATA_ANALYSIS,
            "creative": TaskType.CREATIVE,
            "research": TaskType.RESEARCH,
            "cost_effective": TaskType.COST_EFFECTIVE,
        }
        
        mapped_task = task_type_mapping.get(task_type)
        if not mapped_task:
            return {"error": f"Unknown task type: {task_type}"}
        
        # Get available providers
        available_providers = []
        for provider_name in CONFIG.get_enabled_providers():
            try:
                provider_type = ProviderType(provider_name.lower())
                available_providers.append(provider_type)
            except ValueError:
                continue
        
        if not available_providers:
            return {"error": "No providers available"}
        
        # Get recommendations
        recommendations = self.performance_router.get_task_specific_recommendations(
            mapped_task, available_providers
        )
        
        # Convert to displayable format
        result = {}
        for rec_type, (provider_type, model_name) in recommendations.items():
            # Get performance metrics if available
            metrics = None
            for metric in self.performance_router.performance_metrics:
                if (metric.provider == provider_type and 
                    metric.model_name == model_name and 
                    metric.task_type == mapped_task):
                    metrics = {
                        "success_rate": metric.success_rate,
                        "cost_per_1k_tokens": metric.cost_per_1k_tokens,
                        "quality_score": metric.quality_score,
                        "avg_response_time_ms": metric.avg_response_time_ms
                    }
                    break
            
            result[rec_type] = {
                "provider": provider_type.value,
                "model": model_name,
                "metrics": metrics
            }
        
        return result

    def get_factory_stats(self) -> Dict[str, Any]:
        """Get statistics about the factory configuration."""
        
        # Get performance summary
        performance_summary = self.performance_router.get_performance_summary()
        
        return {
            "default_provider": self.default_provider,
            "fallback_enabled": self.fallback_enabled,
            "routing_strategy": self.routing_strategy.value if hasattr(self.routing_strategy, 'value') else self.routing_strategy,
            "available_providers": CONFIG.get_enabled_providers(),
            "configured_providers": [p for p in CONFIG.provider_configs.keys() if CONFIG.is_provider_configured(p)],
            "provider_optimizations": list(self.provider_optimizations.keys()),
            "performance_routing_enabled": True,
            "cost_optimization_enabled": True,
            "supported_task_types": [t.value for t in TaskType],
            "supported_optimization_modes": [m.value for m in CostOptimizationMode],
            "supported_quality_thresholds": [q.value for q in QualityThreshold],
            "performance_summary": performance_summary
        }


# Global factory instance for convenience
_global_factory = None


def get_agent_factory(
    provider: Optional[str] = None,
    routing_strategy: Optional[Union[str, ProviderStrategy]] = None,
    fallback_enabled: bool = True
) -> MultiProviderAgentFactory:
    """Get the global agent factory instance."""
    global _global_factory
    
    if _global_factory is None or provider is not None:
        _global_factory = MultiProviderAgentFactory(
            default_provider=provider,
            fallback_enabled=fallback_enabled,
            routing_strategy=routing_strategy
        )
    
    return _global_factory


# Convenience functions for backward compatibility
def create_agent(
    name: str,
    instruction: Union[str, Callable],
    model: Optional[str] = None,
    provider: Optional[str] = None,
    **kwargs
) -> agents.Agent:
    """Create an agent with the global factory."""
    factory = get_agent_factory()
    return factory.create_agent(
        name=name,
        instruction=instruction,
        model=model,
        provider=provider,
        **kwargs
    )


def create_coding_agent(
    name: str,
    instruction: Union[str, Callable],
    **kwargs
) -> agents.Agent:
    """Create a coding-optimized agent."""
    factory = get_agent_factory()
    return factory.create_coding_optimized_agent(name, instruction, **kwargs)


def create_reasoning_agent(
    name: str,
    instruction: Union[str, Callable], 
    **kwargs
) -> agents.Agent:
    """Create a reasoning-optimized agent."""
    factory = get_agent_factory()
    return factory.create_reasoning_optimized_agent(name, instruction, **kwargs)


def create_budget_conscious_agent(
    name: str,
    instruction: Union[str, Callable],
    task_type: str = "default",
    quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE,
    budget: Optional[CostBudget] = None,
    **kwargs
) -> agents.Agent:
    """Create a budget-conscious agent that prioritizes DeepSeek for cost savings."""
    factory = get_agent_factory()
    return factory.create_budget_conscious_agent(
        name, instruction, task_type, quality_threshold, budget, **kwargs
    )


def create_maximum_savings_agent(
    name: str,
    instruction: Union[str, Callable],
    task_type: str = "default",
    budget: Optional[CostBudget] = None,
    **kwargs
) -> agents.Agent:
    """Create an agent optimized for maximum cost savings."""
    factory = get_agent_factory()
    return factory.create_maximum_savings_agent(name, instruction, task_type, budget, **kwargs)


def create_adaptive_cost_agent(
    name: str,
    instruction: Union[str, Callable],
    task_type: str = "default",
    monthly_budget_usd: Optional[float] = None,
    quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE,
    **kwargs
) -> agents.Agent:
    """Create an agent with adaptive cost optimization."""
    factory = get_agent_factory()
    return factory.create_adaptive_cost_agent(
        name, instruction, task_type, monthly_budget_usd, quality_threshold, **kwargs
    )


def get_fallback_chain_for_task(
    task_type: str,
    available_providers: Optional[List[ProviderType]] = None
) -> List[str]:
    """Get the fallback chain for a task type (OpenAI → DeepSeek → Claude → Google)."""
    factory = get_agent_factory()
    return factory.get_fallback_chain(task_type, available_providers)