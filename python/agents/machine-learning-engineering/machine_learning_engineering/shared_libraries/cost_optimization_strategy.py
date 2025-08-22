#!/usr/bin/env python3
"""Cost Optimization Strategy for intelligent cost-aware model selection with quality thresholds."""

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union
from .llm_providers import ProviderType
from .model_performance_router import TaskType, get_model_performance_router


class CostOptimizationMode(Enum):
    """Cost optimization modes for different deployment scenarios."""
    PERFORMANCE_FIRST = "performance_first"     # Best performance regardless of cost
    BALANCED = "balanced"                       # Balance cost and performance
    BUDGET_CONSCIOUS = "budget_conscious"       # Optimize for cost with quality thresholds
    MAXIMUM_SAVINGS = "maximum_savings"         # Minimize cost, accept lower quality
    ADAPTIVE = "adaptive"                       # Adapt based on task complexity


class QualityThreshold(Enum):
    """Quality threshold levels for cost optimization."""
    MINIMUM = "minimum"         # 40% success rate threshold
    ACCEPTABLE = "acceptable"   # 55% success rate threshold  
    GOOD = "good"              # 70% success rate threshold
    EXCELLENT = "excellent"     # 85% success rate threshold


@dataclass
class CostBudget:
    """Budget constraints for cost optimization."""
    max_cost_per_1k_tokens: Optional[float] = None     # Maximum cost per 1K tokens
    daily_budget_usd: Optional[float] = None           # Daily budget in USD
    monthly_budget_usd: Optional[float] = None         # Monthly budget in USD
    cost_tracking_enabled: bool = True                 # Enable cost tracking
    alert_threshold_percent: float = 80.0              # Alert when X% of budget used
    
    def is_within_budget(self, cost_per_1k: float, estimated_tokens: int = 1000) -> bool:
        """Check if a cost is within budget constraints."""
        if self.max_cost_per_1k_tokens is None:
            return True
        
        estimated_cost = (estimated_tokens / 1000.0) * cost_per_1k
        return cost_per_1k <= self.max_cost_per_1k_tokens


@dataclass 
class FallbackChainNode:
    """Node in the fallback chain with cost and quality information."""
    provider: ProviderType
    model_name: str
    cost_per_1k_tokens: float
    expected_quality: float                    # Quality score 0-10
    expected_success_rate: float              # Success rate percentage
    priority_order: int                       # Lower number = higher priority
    conditions: Dict[str, Any] = field(default_factory=dict)  # Conditions for using this node
    
    def meets_quality_threshold(self, threshold: QualityThreshold) -> bool:
        """Check if this node meets the quality threshold."""
        threshold_values = {
            QualityThreshold.MINIMUM: 40.0,
            QualityThreshold.ACCEPTABLE: 55.0,
            QualityThreshold.GOOD: 70.0,
            QualityThreshold.EXCELLENT: 85.0
        }
        return self.expected_success_rate >= threshold_values[threshold]
    
    def is_within_budget(self, budget: CostBudget, estimated_tokens: int = 1000) -> bool:
        """Check if this node is within budget."""
        return budget.is_within_budget(self.cost_per_1k_tokens, estimated_tokens)


@dataclass
class CostOptimizationResult:
    """Result of cost optimization selection."""
    selected_provider: ProviderType
    selected_model: str
    estimated_cost_per_1k: float
    expected_quality: float
    expected_success_rate: float
    optimization_mode: CostOptimizationMode
    quality_threshold_met: bool
    budget_compliant: bool
    fallback_reason: Optional[str] = None      # Reason for fallback if applicable
    alternatives_considered: List[str] = field(default_factory=list)
    estimated_savings_percent: Optional[float] = None


class CostOptimizationStrategy:
    """Advanced cost optimization strategy with quality thresholds and fallback chains."""
    
    def __init__(self):
        """Initialize the cost optimization strategy."""
        self.performance_router = get_model_performance_router()
        
        # Define the fallback chains as specified: OpenAI → DeepSeek → Claude → Google Gemini
        self.fallback_chains = self._initialize_fallback_chains()
        
        # Quality threshold mappings
        self.quality_thresholds = {
            QualityThreshold.MINIMUM: 40.0,
            QualityThreshold.ACCEPTABLE: 55.0, 
            QualityThreshold.GOOD: 70.0,
            QualityThreshold.EXCELLENT: 85.0
        }
        
        # Default budgets for different deployment scenarios
        self.default_budgets = {
            CostOptimizationMode.PERFORMANCE_FIRST: CostBudget(max_cost_per_1k_tokens=0.10),
            CostOptimizationMode.BALANCED: CostBudget(max_cost_per_1k_tokens=0.005),
            CostOptimizationMode.BUDGET_CONSCIOUS: CostBudget(max_cost_per_1k_tokens=0.001),
            CostOptimizationMode.MAXIMUM_SAVINGS: CostBudget(max_cost_per_1k_tokens=0.0005),
            CostOptimizationMode.ADAPTIVE: CostBudget(max_cost_per_1k_tokens=0.003)
        }
    
    def _initialize_fallback_chains(self) -> Dict[TaskType, List[FallbackChainNode]]:
        """Initialize fallback chains for different task types."""
        
        # Base fallback chain: OpenAI → DeepSeek → Claude → Google Gemini
        base_chain_data = [
            # OpenAI - Premium performance, higher cost
            {
                "provider": ProviderType.OPENAI,
                "models": {
                    TaskType.CODING: ("gpt-4-turbo-2024-04-09", 0.01, 8.7, 55.2),
                    TaskType.REASONING: ("gpt-4-turbo-2024-04-09", 0.01, 9.1, 75.4),
                    TaskType.DATA_ANALYSIS: ("gpt-4-turbo-2024-04-09", 0.01, 8.9, 74.1),
                    "default": ("gpt-4o-mini", 0.00015, 7.5, 48.7)
                },
                "priority": 1
            },
            
            # DeepSeek - Cost-effective with good performance
            {
                "provider": ProviderType.DEEPSEEK,
                "models": {
                    TaskType.CODING: ("deepseek-coder", 0.00014, 8.3, 62.3),
                    TaskType.REASONING: ("deepseek-chat", 0.00014, 7.2, 52.7),
                    TaskType.DATA_ANALYSIS: ("deepseek-chat", 0.00014, 7.6, 61.8),
                    "default": ("deepseek-chat", 0.00014, 7.8, 57.0)
                },
                "priority": 2
            },
            
            # Claude - High quality, moderate cost
            {
                "provider": ProviderType.ANTHROPIC,
                "models": {
                    TaskType.CODING: ("claude-3-5-sonnet-20241022", 0.003, 9.2, 66.0),
                    TaskType.REASONING: ("claude-3-5-sonnet-20241022", 0.003, 9.0, 73.8),
                    TaskType.DATA_ANALYSIS: ("claude-3-5-sonnet-20241022", 0.003, 9.3, 78.4),
                    "default": ("claude-3-5-haiku-20241022", 0.00025, 8.1, 58.5)
                },
                "priority": 3
            },
            
            # Google Gemini - Balanced option with large context
            {
                "provider": ProviderType.GOOGLE,
                "models": {
                    TaskType.REASONING: ("gemini-1.5-pro", 0.00125, 8.6, 71.2),
                    TaskType.DATA_ANALYSIS: ("gemini-1.5-pro", 0.00125, 8.4, 68.9),
                    "default": ("gemini-1.5-flash", 0.00005, 7.0, 55.3)
                },
                "priority": 4
            }
        ]
        
        # Build fallback chains for each task type
        fallback_chains = {}
        
        for task_type in TaskType:
            chain = []
            
            for provider_data in base_chain_data:
                provider = provider_data["provider"]
                models = provider_data["models"]
                priority = provider_data["priority"]
                
                # Select appropriate model for this task type
                if task_type in models:
                    model_name, cost, quality, success_rate = models[task_type]
                else:
                    model_name, cost, quality, success_rate = models["default"]
                
                # Create fallback chain node
                node = FallbackChainNode(
                    provider=provider,
                    model_name=model_name,
                    cost_per_1k_tokens=cost,
                    expected_quality=quality,
                    expected_success_rate=success_rate,
                    priority_order=priority,
                    conditions={
                        "task_type": task_type,
                        "optimization_preference": "balanced"
                    }
                )
                
                chain.append(node)
            
            # Sort by priority order
            chain.sort(key=lambda x: x.priority_order)
            fallback_chains[task_type] = chain
        
        return fallback_chains
    
    def optimize_model_selection(
        self,
        task_type: TaskType,
        optimization_mode: CostOptimizationMode = CostOptimizationMode.BUDGET_CONSCIOUS,
        quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE,
        budget: Optional[CostBudget] = None,
        available_providers: Optional[List[ProviderType]] = None,
        estimated_tokens: int = 1000
    ) -> CostOptimizationResult:
        """
        Optimize model selection based on cost constraints and quality thresholds.
        
        Args:
            task_type: Type of task to optimize for
            optimization_mode: Cost optimization strategy
            quality_threshold: Minimum quality threshold to maintain
            budget: Budget constraints
            available_providers: List of available providers
            estimated_tokens: Estimated token count for cost calculations
            
        Returns:
            CostOptimizationResult with selected model and reasoning
        """
        
        # Use default budget if none provided
        if budget is None:
            budget = self.default_budgets.get(optimization_mode, CostBudget())
        
        # Get fallback chain for this task type
        fallback_chain = self.fallback_chains.get(task_type, [])
        
        # Filter by available providers if specified
        if available_providers:
            fallback_chain = [
                node for node in fallback_chain 
                if node.provider in available_providers
            ]
        
        if not fallback_chain:
            # No suitable providers available
            return CostOptimizationResult(
                selected_provider=ProviderType.GOOGLE,  # Default fallback
                selected_model="gemini-1.5-flash",
                estimated_cost_per_1k=0.00005,
                expected_quality=7.0,
                expected_success_rate=55.0,
                optimization_mode=optimization_mode,
                quality_threshold_met=False,
                budget_compliant=True,
                fallback_reason="No providers available in fallback chain"
            )
        
        # Apply optimization strategy
        if optimization_mode == CostOptimizationMode.PERFORMANCE_FIRST:
            return self._select_performance_first(fallback_chain, quality_threshold, budget, estimated_tokens, optimization_mode)
        elif optimization_mode == CostOptimizationMode.BUDGET_CONSCIOUS:
            return self._select_budget_conscious(fallback_chain, quality_threshold, budget, estimated_tokens, optimization_mode)
        elif optimization_mode == CostOptimizationMode.MAXIMUM_SAVINGS:
            return self._select_maximum_savings(fallback_chain, quality_threshold, budget, estimated_tokens, optimization_mode)
        elif optimization_mode == CostOptimizationMode.ADAPTIVE:
            return self._select_adaptive(fallback_chain, quality_threshold, budget, estimated_tokens, optimization_mode, task_type)
        else:  # BALANCED
            return self._select_balanced(fallback_chain, quality_threshold, budget, estimated_tokens, optimization_mode)
    
    def _select_performance_first(
        self, 
        chain: List[FallbackChainNode], 
        quality_threshold: QualityThreshold, 
        budget: CostBudget,
        estimated_tokens: int,
        optimization_mode: CostOptimizationMode
    ) -> CostOptimizationResult:
        """Select best performance model that meets quality threshold and budget."""
        
        # Sort by quality (highest first)
        sorted_chain = sorted(chain, key=lambda x: x.expected_quality, reverse=True)
        
        for node in sorted_chain:
            if (node.meets_quality_threshold(quality_threshold) and 
                node.is_within_budget(budget, estimated_tokens)):
                
                return CostOptimizationResult(
                    selected_provider=node.provider,
                    selected_model=node.model_name,
                    estimated_cost_per_1k=node.cost_per_1k_tokens,
                    expected_quality=node.expected_quality,
                    expected_success_rate=node.expected_success_rate,
                    optimization_mode=optimization_mode,
                    quality_threshold_met=True,
                    budget_compliant=True,
                    alternatives_considered=[f"{n.provider.value}:{n.model_name}" for n in chain]
                )
        
        # Fallback to first available option
        return self._fallback_selection(chain[0], optimization_mode, "No options meet both quality and budget constraints")
    
    def _select_budget_conscious(
        self, 
        chain: List[FallbackChainNode], 
        quality_threshold: QualityThreshold, 
        budget: CostBudget,
        estimated_tokens: int,
        optimization_mode: CostOptimizationMode
    ) -> CostOptimizationResult:
        """Select most cost-effective model that meets quality threshold - DeepSeek priority."""
        
        # Primary strategy: Find DeepSeek option first (as specified in requirements)
        deepseek_options = [node for node in chain if node.provider == ProviderType.DEEPSEEK]
        
        for node in deepseek_options:
            if (node.meets_quality_threshold(quality_threshold) and 
                node.is_within_budget(budget, estimated_tokens)):
                
                # Calculate savings compared to most expensive option
                most_expensive_cost = max(n.cost_per_1k_tokens for n in chain)
                savings_percent = ((most_expensive_cost - node.cost_per_1k_tokens) / most_expensive_cost) * 100
                
                return CostOptimizationResult(
                    selected_provider=node.provider,
                    selected_model=node.model_name,
                    estimated_cost_per_1k=node.cost_per_1k_tokens,
                    expected_quality=node.expected_quality,
                    expected_success_rate=node.expected_success_rate,
                    optimization_mode=optimization_mode,
                    quality_threshold_met=True,
                    budget_compliant=True,
                    alternatives_considered=[f"{n.provider.value}:{n.model_name}" for n in chain],
                    estimated_savings_percent=savings_percent
                )
        
        # Secondary strategy: Sort by cost (lowest first) and find first acceptable option
        sorted_chain = sorted(chain, key=lambda x: x.cost_per_1k_tokens)
        
        for node in sorted_chain:
            if (node.meets_quality_threshold(quality_threshold) and 
                node.is_within_budget(budget, estimated_tokens)):
                
                # Calculate savings
                most_expensive_cost = max(n.cost_per_1k_tokens for n in chain)
                savings_percent = ((most_expensive_cost - node.cost_per_1k_tokens) / most_expensive_cost) * 100
                
                return CostOptimizationResult(
                    selected_provider=node.provider,
                    selected_model=node.model_name,
                    estimated_cost_per_1k=node.cost_per_1k_tokens,
                    expected_quality=node.expected_quality,
                    expected_success_rate=node.expected_success_rate,
                    optimization_mode=optimization_mode,
                    quality_threshold_met=True,
                    budget_compliant=True,
                    fallback_reason="DeepSeek not available, selected next best cost option",
                    alternatives_considered=[f"{n.provider.value}:{n.model_name}" for n in chain],
                    estimated_savings_percent=savings_percent
                )
        
        # Fallback with relaxed quality threshold
        return self._fallback_selection(sorted_chain[0], optimization_mode, "No options meet quality threshold, using lowest cost")
    
    def _select_maximum_savings(
        self, 
        chain: List[FallbackChainNode], 
        quality_threshold: QualityThreshold, 
        budget: CostBudget,
        estimated_tokens: int,
        optimization_mode: CostOptimizationMode
    ) -> CostOptimizationResult:
        """Select cheapest model regardless of quality (with minimum threshold)."""
        
        # Sort by cost (lowest first)
        sorted_chain = sorted(chain, key=lambda x: x.cost_per_1k_tokens)
        
        # Use minimum threshold for maximum savings
        min_threshold = QualityThreshold.MINIMUM
        
        for node in sorted_chain:
            if (node.meets_quality_threshold(min_threshold) and 
                node.is_within_budget(budget, estimated_tokens)):
                
                most_expensive_cost = max(n.cost_per_1k_tokens for n in chain)
                savings_percent = ((most_expensive_cost - node.cost_per_1k_tokens) / most_expensive_cost) * 100
                
                return CostOptimizationResult(
                    selected_provider=node.provider,
                    selected_model=node.model_name,
                    estimated_cost_per_1k=node.cost_per_1k_tokens,
                    expected_quality=node.expected_quality,
                    expected_success_rate=node.expected_success_rate,
                    optimization_mode=optimization_mode,
                    quality_threshold_met=node.meets_quality_threshold(quality_threshold),
                    budget_compliant=True,
                    alternatives_considered=[f"{n.provider.value}:{n.model_name}" for n in chain],
                    estimated_savings_percent=savings_percent
                )
        
        # Use cheapest option even if it doesn't meet threshold
        cheapest = sorted_chain[0]
        return self._fallback_selection(cheapest, optimization_mode, "Maximum savings mode: using cheapest option")
    
    def _select_adaptive(
        self, 
        chain: List[FallbackChainNode], 
        quality_threshold: QualityThreshold, 
        budget: CostBudget,
        estimated_tokens: int,
        optimization_mode: CostOptimizationMode,
        task_type: TaskType
    ) -> CostOptimizationResult:
        """Adaptive selection based on task complexity and estimated workload."""
        
        # Determine complexity and adjust strategy
        high_complexity_tasks = [TaskType.REASONING, TaskType.DATA_ANALYSIS, TaskType.ENSEMBLE]
        medium_complexity_tasks = [TaskType.CODING, TaskType.DEBUGGING, TaskType.REFINEMENT] 
        low_complexity_tasks = [TaskType.CREATIVE, TaskType.RESEARCH, TaskType.SUBMISSION]
        
        if task_type in high_complexity_tasks:
            # Use performance-first for complex tasks
            return self._select_performance_first(chain, quality_threshold, budget, estimated_tokens, optimization_mode)
        elif task_type in medium_complexity_tasks:
            # Use balanced approach for medium tasks
            return self._select_balanced(chain, quality_threshold, budget, estimated_tokens, optimization_mode)
        else:
            # Use budget-conscious for simple tasks
            return self._select_budget_conscious(chain, quality_threshold, budget, estimated_tokens, optimization_mode)
    
    def _select_balanced(
        self, 
        chain: List[FallbackChainNode], 
        quality_threshold: QualityThreshold, 
        budget: CostBudget,
        estimated_tokens: int,
        optimization_mode: CostOptimizationMode
    ) -> CostOptimizationResult:
        """Select model with best cost/quality ratio."""
        
        # Calculate cost-quality score for each option
        scored_options = []
        for node in chain:
            if (node.meets_quality_threshold(quality_threshold) and 
                node.is_within_budget(budget, estimated_tokens)):
                
                # Higher quality and lower cost = better score
                # Normalize cost (0-1, lower is better) and quality (0-1, higher is better)
                max_cost = max(n.cost_per_1k_tokens for n in chain)
                min_cost = min(n.cost_per_1k_tokens for n in chain)
                max_quality = max(n.expected_quality for n in chain)
                min_quality = min(n.expected_quality for n in chain)
                
                if max_cost > min_cost:
                    normalized_cost = 1.0 - ((node.cost_per_1k_tokens - min_cost) / (max_cost - min_cost))
                else:
                    normalized_cost = 1.0
                
                if max_quality > min_quality:
                    normalized_quality = (node.expected_quality - min_quality) / (max_quality - min_quality)
                else:
                    normalized_quality = 1.0
                
                # Balanced score: 60% quality, 40% cost-effectiveness
                balance_score = (0.6 * normalized_quality) + (0.4 * normalized_cost)
                
                scored_options.append((balance_score, node))
        
        if scored_options:
            # Select best balanced option
            best_score, best_node = max(scored_options, key=lambda x: x[0])
            
            most_expensive_cost = max(n.cost_per_1k_tokens for n in chain)
            savings_percent = ((most_expensive_cost - best_node.cost_per_1k_tokens) / most_expensive_cost) * 100
            
            return CostOptimizationResult(
                selected_provider=best_node.provider,
                selected_model=best_node.model_name,
                estimated_cost_per_1k=best_node.cost_per_1k_tokens,
                expected_quality=best_node.expected_quality,
                expected_success_rate=best_node.expected_success_rate,
                optimization_mode=optimization_mode,
                quality_threshold_met=True,
                budget_compliant=True,
                alternatives_considered=[f"{n.provider.value}:{n.model_name}" for n in chain],
                estimated_savings_percent=savings_percent
            )
        
        # Fallback to first option in chain (OpenAI)
        return self._fallback_selection(chain[0], optimization_mode, "No balanced options found")
    
    def _fallback_selection(
        self, 
        node: FallbackChainNode, 
        optimization_mode: CostOptimizationMode,
        reason: str
    ) -> CostOptimizationResult:
        """Create a fallback selection result."""
        return CostOptimizationResult(
            selected_provider=node.provider,
            selected_model=node.model_name,
            estimated_cost_per_1k=node.cost_per_1k_tokens,
            expected_quality=node.expected_quality,
            expected_success_rate=node.expected_success_rate,
            optimization_mode=optimization_mode,
            quality_threshold_met=False,
            budget_compliant=True,
            fallback_reason=reason
        )
    
    def get_fallback_chain(
        self, 
        task_type: TaskType,
        available_providers: Optional[List[ProviderType]] = None
    ) -> List[FallbackChainNode]:
        """Get the fallback chain for a specific task type."""
        chain = self.fallback_chains.get(task_type, [])
        
        if available_providers:
            chain = [node for node in chain if node.provider in available_providers]
        
        return chain
    
    def estimate_monthly_cost(
        self,
        task_type: TaskType,
        tokens_per_day: int,
        optimization_mode: CostOptimizationMode = CostOptimizationMode.BUDGET_CONSCIOUS,
        quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE
    ) -> Dict[str, Any]:
        """Estimate monthly cost for a given usage pattern."""
        
        # Get optimized selection
        result = self.optimize_model_selection(
            task_type=task_type,
            optimization_mode=optimization_mode,
            quality_threshold=quality_threshold,
            estimated_tokens=tokens_per_day
        )
        
        # Calculate monthly costs
        daily_cost = (tokens_per_day / 1000.0) * result.estimated_cost_per_1k
        monthly_cost = daily_cost * 30
        
        # Compare with other modes
        comparisons = {}
        for mode in CostOptimizationMode:
            if mode != optimization_mode:
                comp_result = self.optimize_model_selection(
                    task_type=task_type,
                    optimization_mode=mode,
                    quality_threshold=quality_threshold,
                    estimated_tokens=tokens_per_day
                )
                comp_daily_cost = (tokens_per_day / 1000.0) * comp_result.estimated_cost_per_1k
                comp_monthly_cost = comp_daily_cost * 30
                comparisons[mode.value] = {
                    "monthly_cost": comp_monthly_cost,
                    "provider": comp_result.selected_provider.value,
                    "model": comp_result.selected_model
                }
        
        return {
            "selected_option": {
                "provider": result.selected_provider.value,
                "model": result.selected_model,
                "optimization_mode": optimization_mode.value,
                "daily_cost": daily_cost,
                "monthly_cost": monthly_cost,
                "quality_threshold_met": result.quality_threshold_met,
                "expected_success_rate": result.expected_success_rate
            },
            "usage_pattern": {
                "tokens_per_day": tokens_per_day,
                "tokens_per_month": tokens_per_day * 30,
                "task_type": task_type.value
            },
            "comparisons": comparisons,
            "potential_savings": result.estimated_savings_percent
        }
    
    def get_cost_optimization_recommendations(
        self,
        task_type: TaskType,
        current_monthly_cost: float,
        available_providers: Optional[List[ProviderType]] = None
    ) -> Dict[str, Any]:
        """Get cost optimization recommendations for current usage."""
        
        recommendations = []
        
        # Test different optimization modes
        modes_to_test = [
            CostOptimizationMode.BUDGET_CONSCIOUS,
            CostOptimizationMode.MAXIMUM_SAVINGS,
            CostOptimizationMode.BALANCED
        ]
        
        for mode in modes_to_test:
            result = self.optimize_model_selection(
                task_type=task_type,
                optimization_mode=mode,
                available_providers=available_providers,
                estimated_tokens=1000
            )
            
            # Estimate monthly savings
            estimated_monthly_cost = current_monthly_cost * (result.estimated_cost_per_1k / 0.01)  # Assume current is GPT-4 cost
            potential_savings = current_monthly_cost - estimated_monthly_cost
            savings_percent = (potential_savings / current_monthly_cost) * 100 if current_monthly_cost > 0 else 0
            
            recommendations.append({
                "optimization_mode": mode.value,
                "recommended_provider": result.selected_provider.value,
                "recommended_model": result.selected_model,
                "estimated_monthly_cost": estimated_monthly_cost,
                "potential_monthly_savings": potential_savings,
                "savings_percent": savings_percent,
                "quality_impact": result.expected_success_rate,
                "quality_threshold_met": result.quality_threshold_met
            })
        
        # Sort by savings potential
        recommendations.sort(key=lambda x: x["savings_percent"], reverse=True)
        
        return {
            "current_monthly_cost": current_monthly_cost,
            "task_type": task_type.value,
            "recommendations": recommendations,
            "top_recommendation": recommendations[0] if recommendations else None
        }


def get_cost_optimization_strategy() -> CostOptimizationStrategy:
    """Get a singleton instance of the CostOptimizationStrategy."""
    if not hasattr(get_cost_optimization_strategy, '_instance'):
        get_cost_optimization_strategy._instance = CostOptimizationStrategy()
    return get_cost_optimization_strategy._instance


# Convenience functions for common cost optimization scenarios
def get_budget_conscious_model(
    task_type: TaskType,
    available_providers: Optional[List[ProviderType]] = None,
    quality_threshold: QualityThreshold = QualityThreshold.ACCEPTABLE
) -> CostOptimizationResult:
    """Get budget-conscious model recommendation with DeepSeek priority."""
    strategy = get_cost_optimization_strategy()
    return strategy.optimize_model_selection(
        task_type=task_type,
        optimization_mode=CostOptimizationMode.BUDGET_CONSCIOUS,
        quality_threshold=quality_threshold,
        available_providers=available_providers
    )


def get_maximum_savings_model(
    task_type: TaskType,
    available_providers: Optional[List[ProviderType]] = None
) -> CostOptimizationResult:
    """Get maximum savings model recommendation."""
    strategy = get_cost_optimization_strategy()
    return strategy.optimize_model_selection(
        task_type=task_type,
        optimization_mode=CostOptimizationMode.MAXIMUM_SAVINGS,
        quality_threshold=QualityThreshold.MINIMUM,
        available_providers=available_providers
    )


def get_fallback_chain_for_task(
    task_type: TaskType,
    available_providers: Optional[List[ProviderType]] = None
) -> List[FallbackChainNode]:
    """Get the fallback chain for a task type following OpenAI → DeepSeek → Claude → Google."""
    strategy = get_cost_optimization_strategy()
    return strategy.get_fallback_chain(task_type, available_providers)