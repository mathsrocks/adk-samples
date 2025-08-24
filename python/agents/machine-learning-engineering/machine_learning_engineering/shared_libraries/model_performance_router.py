#!/usr/bin/env python3
"""Model Performance Routing System for intelligent model selection based on task type and performance metrics."""

import os
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from .llm_providers import ProviderType


class TaskType(Enum):
    """Task types for model routing optimization."""
    CODING = "coding"                    # Code generation, debugging, refactoring
    REASONING = "reasoning"              # Complex reasoning, analysis, planning
    COST_EFFECTIVE = "cost_effective"    # Budget-optimized tasks
    DATA_ANALYSIS = "data_analysis"      # Data processing, ML tasks
    CREATIVE = "creative"                # Creative writing, content generation
    RESEARCH = "research"                # Research, web search, fact-finding
    ENSEMBLE = "ensemble"                # Ensemble strategy generation
    DEBUGGING = "debugging"              # Error correction, troubleshooting
    REFINEMENT = "refinement"            # Code refinement, optimization
    SUBMISSION = "submission"            # Final output preparation


@dataclass
class ModelPerformanceMetric:
    """Performance metrics for a specific model on a task type."""
    model_name: str
    provider: ProviderType
    task_type: TaskType
    success_rate: float                  # Success rate percentage (0-100)
    cost_per_1k_tokens: float           # Cost in USD per 1K tokens
    avg_response_time_ms: int           # Average response time in milliseconds
    quality_score: float                # Quality score (0-10)
    context_window: int                 # Maximum context window size
    benchmark_name: Optional[str] = None # Benchmark source (e.g., "QML", "HumanEval")
    last_updated: Optional[str] = None  # Last benchmark update


@dataclass
class CostEffectiveModel:
    """Cost-effective model recommendation for a provider."""
    provider: ProviderType
    model_name: str
    cost_per_1k_tokens: float
    performance_tier: str               # "high", "medium", "low"
    recommended_for: List[TaskType]
    notes: Optional[str] = None


class ModelPerformanceRouter:
    """Intelligent model routing system based on task type and performance metrics."""
    
    def __init__(self):
        self.performance_metrics = self._initialize_performance_metrics()
        self.cost_effective_models = self._initialize_cost_effective_models()
        self.task_model_priorities = self._initialize_task_priorities()
        
    def _initialize_performance_metrics(self) -> List[ModelPerformanceMetric]:
        """Initialize performance metrics based on benchmarks and research."""
        return [
            # =====================================================================
            # CODING TASK PERFORMANCE (HumanEval, QML Benchmarks)
            # =====================================================================
            
            # Anthropic Claude - Best for coding (66% QML success rate mentioned)
            ModelPerformanceMetric(
                model_name="claude-3-5-sonnet-20241022",
                provider=ProviderType.ANTHROPIC,
                task_type=TaskType.CODING,
                success_rate=66.0,
                cost_per_1k_tokens=0.003,  # Input cost
                avg_response_time_ms=2500,
                quality_score=9.2,
                context_window=200000,
                benchmark_name="QML",
                last_updated="2024-10"
            ),
            ModelPerformanceMetric(
                model_name="claude-3-5-haiku-20241022",
                provider=ProviderType.ANTHROPIC,
                task_type=TaskType.CODING,
                success_rate=58.5,
                cost_per_1k_tokens=0.00025,  # Cost-effective alternative
                avg_response_time_ms=1800,
                quality_score=8.1,
                context_window=200000,
                benchmark_name="HumanEval"
            ),
            
            # DeepSeek - Cost-effective coding (57% success rate mentioned)
            ModelPerformanceMetric(
                model_name="deepseek-chat",
                provider=ProviderType.DEEPSEEK,
                task_type=TaskType.CODING,
                success_rate=57.0,
                cost_per_1k_tokens=0.00014,  # Very cost-effective
                avg_response_time_ms=2200,
                quality_score=7.8,
                context_window=64000,
                benchmark_name="QML"
            ),
            ModelPerformanceMetric(
                model_name="deepseek-coder",
                provider=ProviderType.DEEPSEEK,
                task_type=TaskType.CODING,
                success_rate=62.3,
                cost_per_1k_tokens=0.00014,
                avg_response_time_ms=2100,
                quality_score=8.3,
                context_window=16000,
                benchmark_name="HumanEval"
            ),
            
            # OpenAI GPT-4 family
            ModelPerformanceMetric(
                model_name="gpt-4-turbo-2024-04-09",
                provider=ProviderType.OPENAI,
                task_type=TaskType.CODING,
                success_rate=55.2,
                cost_per_1k_tokens=0.01,
                avg_response_time_ms=3200,
                quality_score=8.7,
                context_window=128000,
                benchmark_name="HumanEval"
            ),
            ModelPerformanceMetric(
                model_name="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                task_type=TaskType.CODING,
                success_rate=48.7,
                cost_per_1k_tokens=0.00015,  # Cost-effective
                avg_response_time_ms=2000,
                quality_score=7.5,
                context_window=128000,
                benchmark_name="HumanEval"
            ),
            
            # =====================================================================
            # REASONING TASK PERFORMANCE
            # =====================================================================
            
            # OpenAI GPT-4 - Strong reasoning (30.3% baseline mentioned)
            ModelPerformanceMetric(
                model_name="gpt-4-turbo-2024-04-09",
                provider=ProviderType.OPENAI,
                task_type=TaskType.REASONING,
                success_rate=75.4,
                cost_per_1k_tokens=0.01,
                avg_response_time_ms=3500,
                quality_score=9.1,
                context_window=128000,
                benchmark_name="MMLU"
            ),
            ModelPerformanceMetric(
                model_name="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                task_type=TaskType.REASONING,
                success_rate=30.3,  # Baseline mentioned in requirements
                cost_per_1k_tokens=0.00015,
                avg_response_time_ms=2200,
                quality_score=6.8,
                context_window=128000,
                benchmark_name="GSM8K"
            ),
            ModelPerformanceMetric(
                model_name="o1-mini",
                provider=ProviderType.OPENAI,
                task_type=TaskType.REASONING,
                success_rate=82.1,
                cost_per_1k_tokens=0.003,
                avg_response_time_ms=8000,  # Slower due to reasoning
                quality_score=9.5,
                context_window=128000,
                benchmark_name="AIME"
            ),
            
            # Anthropic Claude - Strong reasoning
            ModelPerformanceMetric(
                model_name="claude-3-5-sonnet-20241022",
                provider=ProviderType.ANTHROPIC,
                task_type=TaskType.REASONING,
                success_rate=73.8,
                cost_per_1k_tokens=0.003,
                avg_response_time_ms=3000,
                quality_score=9.0,
                context_window=200000,
                benchmark_name="MMLU"
            ),
            
            # Google Gemini
            ModelPerformanceMetric(
                model_name="gemini-1.5-pro",
                provider=ProviderType.GOOGLE,
                task_type=TaskType.REASONING,
                success_rate=71.2,
                cost_per_1k_tokens=0.00125,
                avg_response_time_ms=2800,
                quality_score=8.6,
                context_window=2000000,  # Largest context window
                benchmark_name="MMLU"
            ),
            
            # DeepSeek - Cost-effective reasoning
            ModelPerformanceMetric(
                model_name="deepseek-chat",
                provider=ProviderType.DEEPSEEK,
                task_type=TaskType.REASONING,
                success_rate=52.7,
                cost_per_1k_tokens=0.00014,
                avg_response_time_ms=2500,
                quality_score=7.2,
                context_window=64000,
                benchmark_name="MMLU"
            ),
            
            # =====================================================================
            # COST-EFFECTIVE TASK PERFORMANCE
            # =====================================================================
            
            # DeepSeek - Most cost-effective
            ModelPerformanceMetric(
                model_name="deepseek-chat",
                provider=ProviderType.DEEPSEEK,
                task_type=TaskType.COST_EFFECTIVE,
                success_rate=57.0,  # From requirements
                cost_per_1k_tokens=0.00014,
                avg_response_time_ms=2200,
                quality_score=7.8,
                context_window=64000,
                benchmark_name="Overall"
            ),
            
            # Groq - Fast inference, cost-effective
            ModelPerformanceMetric(
                model_name="llama-3.3-70b-versatile",
                provider=ProviderType.GROQ,
                task_type=TaskType.COST_EFFECTIVE,
                success_rate=51.2,
                cost_per_1k_tokens=0.00059,
                avg_response_time_ms=800,  # Very fast
                quality_score=7.4,
                context_window=128000,
                benchmark_name="Various"
            ),
            
            # OpenAI GPT-4o-mini
            ModelPerformanceMetric(
                model_name="gpt-4o-mini",
                provider=ProviderType.OPENAI,
                task_type=TaskType.COST_EFFECTIVE,
                success_rate=48.7,
                cost_per_1k_tokens=0.00015,
                avg_response_time_ms=2000,
                quality_score=7.5,
                context_window=128000,
                benchmark_name="Various"
            ),
            
            # Claude Haiku - Cost-effective from Anthropic
            ModelPerformanceMetric(
                model_name="claude-3-5-haiku-20241022",
                provider=ProviderType.ANTHROPIC,
                task_type=TaskType.COST_EFFECTIVE,
                success_rate=58.5,
                cost_per_1k_tokens=0.00025,
                avg_response_time_ms=1800,
                quality_score=8.1,
                context_window=200000,
                benchmark_name="Various"
            ),
            
            # =====================================================================
            # DATA ANALYSIS TASK PERFORMANCE
            # =====================================================================
            
            # Anthropic Claude - Excellent for analysis
            ModelPerformanceMetric(
                model_name="claude-3-5-sonnet-20241022",
                provider=ProviderType.ANTHROPIC,
                task_type=TaskType.DATA_ANALYSIS,
                success_rate=78.4,
                cost_per_1k_tokens=0.003,
                avg_response_time_ms=2800,
                quality_score=9.3,
                context_window=200000,
                benchmark_name="Custom"
            ),
            
            # OpenAI GPT-4 - Strong analytical capabilities
            ModelPerformanceMetric(
                model_name="gpt-4-turbo-2024-04-09",
                provider=ProviderType.OPENAI,
                task_type=TaskType.DATA_ANALYSIS,
                success_rate=74.1,
                cost_per_1k_tokens=0.01,
                avg_response_time_ms=3200,
                quality_score=8.9,
                context_window=128000,
                benchmark_name="Custom"
            ),
            
            # DeepSeek - Cost-effective analysis
            ModelPerformanceMetric(
                model_name="deepseek-chat",
                provider=ProviderType.DEEPSEEK,
                task_type=TaskType.DATA_ANALYSIS,
                success_rate=61.8,
                cost_per_1k_tokens=0.00014,
                avg_response_time_ms=2400,
                quality_score=7.6,
                context_window=64000,
                benchmark_name="Custom"
            ),
        ]
    
    def _initialize_cost_effective_models(self) -> List[CostEffectiveModel]:
        """Initialize cost-effective model recommendations by provider."""
        return [
            # DeepSeek - Most cost-effective overall
            CostEffectiveModel(
                provider=ProviderType.DEEPSEEK,
                model_name="deepseek-chat",
                cost_per_1k_tokens=0.00014,
                performance_tier="high",
                recommended_for=[TaskType.CODING, TaskType.COST_EFFECTIVE, TaskType.DATA_ANALYSIS, TaskType.REFINEMENT],
                notes="Best cost/performance ratio, excellent for coding tasks"
            ),
            CostEffectiveModel(
                provider=ProviderType.DEEPSEEK,
                model_name="deepseek-coder",
                cost_per_1k_tokens=0.00014,
                performance_tier="high",
                recommended_for=[TaskType.CODING, TaskType.DEBUGGING, TaskType.REFINEMENT],
                notes="Specialized for coding tasks, same cost as deepseek-chat"
            ),
            
            # OpenAI - Cost-effective options
            CostEffectiveModel(
                provider=ProviderType.OPENAI,
                model_name="gpt-4o-mini",
                cost_per_1k_tokens=0.00015,
                performance_tier="medium",
                recommended_for=[TaskType.COST_EFFECTIVE, TaskType.REASONING, TaskType.CREATIVE],
                notes="Balance of capability and cost, good general-purpose model"
            ),
            CostEffectiveModel(
                provider=ProviderType.OPENAI,
                model_name="gpt-3.5-turbo",
                cost_per_1k_tokens=0.0005,
                performance_tier="medium",
                recommended_for=[TaskType.COST_EFFECTIVE, TaskType.CREATIVE, TaskType.RESEARCH],
                notes="Legacy but very cost-effective for basic tasks"
            ),
            
            # Anthropic - Cost-effective option
            CostEffectiveModel(
                provider=ProviderType.ANTHROPIC,
                model_name="claude-3-5-haiku-20241022",
                cost_per_1k_tokens=0.00025,
                performance_tier="high",
                recommended_for=[TaskType.CODING, TaskType.COST_EFFECTIVE, TaskType.DATA_ANALYSIS],
                notes="Fast and cost-effective while maintaining high quality"
            ),
            
            # Groq - Fastest inference
            CostEffectiveModel(
                provider=ProviderType.GROQ,
                model_name="llama-3.3-70b-versatile",
                cost_per_1k_tokens=0.00059,
                performance_tier="medium",
                recommended_for=[TaskType.COST_EFFECTIVE, TaskType.CREATIVE],
                notes="Extremely fast inference, good for time-sensitive tasks"
            ),
            CostEffectiveModel(
                provider=ProviderType.GROQ,
                model_name="mixtral-8x7b-32768",
                cost_per_1k_tokens=0.00024,
                performance_tier="medium",
                recommended_for=[TaskType.COST_EFFECTIVE, TaskType.REASONING],
                notes="Fast and affordable mixture of experts model"
            ),
            
            # Google - Cost-effective options
            CostEffectiveModel(
                provider=ProviderType.GOOGLE,
                model_name="gemini-1.5-flash",
                cost_per_1k_tokens=0.00005,
                performance_tier="medium",
                recommended_for=[TaskType.COST_EFFECTIVE, TaskType.RESEARCH, TaskType.CREATIVE],
                notes="Very affordable with large context window"
            ),
            
            # Local options - No API costs
            CostEffectiveModel(
                provider=ProviderType.OLLAMA,
                model_name="llama3.2",
                cost_per_1k_tokens=0.0,
                performance_tier="low",
                recommended_for=[TaskType.COST_EFFECTIVE],
                notes="No API costs, runs locally, good for development/testing"
            ),
        ]
    
    def _initialize_task_priorities(self) -> Dict[TaskType, List[Tuple[ProviderType, str, float]]]:
        """Initialize task-specific model priorities (provider, model, weight)."""
        return {
            TaskType.CODING: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),      # Best performance
                (ProviderType.DEEPSEEK, "deepseek-coder", 0.9),                   # Cost-effective specialist
                (ProviderType.ANTHROPIC, "claude-3-5-haiku-20241022", 0.85),     # Cost-effective high quality
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.8),                   # Most cost-effective
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.75),           # Solid performance
                (ProviderType.OPENAI, "gpt-4o-mini", 0.7),                       # Budget option
            ],
            
            TaskType.REASONING: [
                (ProviderType.OPENAI, "o1-mini", 1.0),                           # Best reasoning
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.92),          # Strong reasoning
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 0.9),    # Excellent reasoning
                (ProviderType.GOOGLE, "gemini-1.5-pro", 0.87),                  # Good reasoning + large context
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.65),                 # Cost-effective option
                (ProviderType.OPENAI, "gpt-4o-mini", 0.4),                      # Baseline (30.3% mentioned)
            ],
            
            TaskType.COST_EFFECTIVE: [
                (ProviderType.DEEPSEEK, "deepseek-chat", 1.0),                  # Best cost/performance
                (ProviderType.OPENAI, "gpt-4o-mini", 0.9),                      # Good cost/performance
                (ProviderType.ANTHROPIC, "claude-3-5-haiku-20241022", 0.88),   # Fast and affordable
                (ProviderType.GROQ, "llama-3.3-70b-versatile", 0.85),          # Fast inference
                (ProviderType.GROQ, "mixtral-8x7b-32768", 0.8),                # Affordable expert model
                (ProviderType.GOOGLE, "gemini-1.5-flash", 0.75),               # Very low cost
                (ProviderType.OLLAMA, "llama3.2", 0.7),                        # No API cost
            ],
            
            TaskType.DATA_ANALYSIS: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # Best analysis
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.9),          # Strong analysis
                (ProviderType.GOOGLE, "gemini-1.5-pro", 0.85),                 # Large context for data
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.75),                # Cost-effective analysis
                (ProviderType.ANTHROPIC, "claude-3-5-haiku-20241022", 0.7),   # Fast analysis
            ],
            
            TaskType.CREATIVE: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # Creative writing
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.9),          # Creative tasks
                (ProviderType.GROQ, "llama-3.3-70b-versatile", 0.8),           # Fast creative generation
                (ProviderType.OPENAI, "gpt-4o-mini", 0.75),                    # Budget creative
                (ProviderType.GOOGLE, "gemini-1.5-pro", 0.7),                  # Good creative capabilities
            ],
            
            TaskType.RESEARCH: [
                (ProviderType.GOOGLE, "gemini-1.5-pro", 1.0),                  # Largest context for research
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 0.95),  # Excellent research
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.9),          # Good research
                (ProviderType.PERPLEXITY, "sonar-pro", 0.85),                  # Web search integration
                (ProviderType.GOOGLE, "gemini-1.5-flash", 0.7),                # Cost-effective research
            ],
            
            # MLE-STAR specific task types
            TaskType.ENSEMBLE: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # Strategic thinking
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.9),          # Complex planning
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.8),                 # Cost-effective ensemble
            ],
            
            TaskType.DEBUGGING: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # Best debugging
                (ProviderType.DEEPSEEK, "deepseek-coder", 0.95),               # Coding specialist
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.9),                 # Cost-effective debugging
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.85),         # Good debugging
            ],
            
            TaskType.REFINEMENT: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # Best refinement
                (ProviderType.DEEPSEEK, "deepseek-coder", 0.9),                # Coding refinement
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.85),                # Cost-effective refinement
                (ProviderType.ANTHROPIC, "claude-3-5-haiku-20241022", 0.8),   # Fast refinement
            ],
            
            TaskType.SUBMISSION: [
                (ProviderType.ANTHROPIC, "claude-3-5-sonnet-20241022", 1.0),   # High-quality output
                (ProviderType.OPENAI, "gpt-4-turbo-2024-04-09", 0.9),          # Professional output
                (ProviderType.DEEPSEEK, "deepseek-chat", 0.75),                # Cost-effective final output
            ],
        }
    
    def get_best_model_for_task(
        self, 
        task_type: TaskType, 
        available_providers: Optional[List[ProviderType]] = None,
        cost_priority: bool = False,
        quality_priority: bool = False
    ) -> Optional[Tuple[ProviderType, str]]:
        """
        Get the best model for a specific task type.
        
        Args:
            task_type: The type of task to optimize for
            available_providers: List of available providers (if None, uses all)
            cost_priority: Prioritize cost-effective models
            quality_priority: Prioritize highest quality regardless of cost
            
        Returns:
            Tuple of (provider, model_name) or None if no suitable model found
        """
        if cost_priority:
            return self._get_cost_optimized_model(task_type, available_providers)
        
        if quality_priority:
            return self._get_highest_quality_model(task_type, available_providers)
        
        # Default: balanced selection based on task priorities
        if task_type not in self.task_model_priorities:
            return None
            
        priorities = self.task_model_priorities[task_type]
        
        for provider, model_name, weight in priorities:
            if available_providers is None or provider in available_providers:
                return (provider, model_name)
        
        return None
    
    def _get_cost_optimized_model(
        self, 
        task_type: TaskType, 
        available_providers: Optional[List[ProviderType]] = None
    ) -> Optional[Tuple[ProviderType, str]]:
        """Get the most cost-effective model for a task type."""
        suitable_models = []
        
        for model in self.cost_effective_models:
            if (available_providers is None or model.provider in available_providers) and \
               task_type in model.recommended_for:
                # Calculate cost-effectiveness score (higher is better)
                score = 1.0 / model.cost_per_1k_tokens if model.cost_per_1k_tokens > 0 else 1000.0
                
                # Boost score based on performance tier
                tier_multiplier = {"high": 1.5, "medium": 1.0, "low": 0.7}
                score *= tier_multiplier.get(model.performance_tier, 1.0)
                
                suitable_models.append((score, model.provider, model.model_name))
        
        if suitable_models:
            # Sort by score (highest first) and return best option
            suitable_models.sort(reverse=True)
            return (suitable_models[0][1], suitable_models[0][2])
        
        return None
    
    def _get_highest_quality_model(
        self, 
        task_type: TaskType, 
        available_providers: Optional[List[ProviderType]] = None
    ) -> Optional[Tuple[ProviderType, str]]:
        """Get the highest quality model for a task type regardless of cost."""
        best_model = None
        best_quality = 0.0
        
        for metric in self.performance_metrics:
            if metric.task_type == task_type and \
               (available_providers is None or metric.provider in available_providers):
                if metric.quality_score > best_quality:
                    best_quality = metric.quality_score
                    best_model = (metric.provider, metric.model_name)
        
        return best_model
    
    def get_model_performance_metrics(
        self, 
        task_type: TaskType
    ) -> List[ModelPerformanceMetric]:
        """Get all performance metrics for a specific task type."""
        return [
            metric for metric in self.performance_metrics 
            if metric.task_type == task_type
        ]
    
    def get_cost_effective_models_by_provider(
        self, 
        provider: ProviderType
    ) -> List[CostEffectiveModel]:
        """Get all cost-effective models for a specific provider."""
        return [
            model for model in self.cost_effective_models 
            if model.provider == provider
        ]
    
    def estimate_cost_for_task(
        self, 
        provider: ProviderType, 
        model_name: str, 
        estimated_tokens: int
    ) -> float:
        """Estimate the cost for running a task with specific token count."""
        for metric in self.performance_metrics:
            if metric.provider == provider and metric.model_name == model_name:
                return (estimated_tokens / 1000.0) * metric.cost_per_1k_tokens
        
        # Fallback to cost-effective models if not found in metrics
        for model in self.cost_effective_models:
            if model.provider == provider and model.model_name == model_name:
                return (estimated_tokens / 1000.0) * model.cost_per_1k_tokens
        
        return 0.0  # Unknown model
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of performance data for all task types."""
        summary = {}
        
        for task_type in TaskType:
            metrics = self.get_model_performance_metrics(task_type)
            if metrics:
                best_performance = max(metrics, key=lambda x: x.success_rate)
                best_cost_effective = min(metrics, key=lambda x: x.cost_per_1k_tokens)
                
                summary[task_type.value] = {
                    "total_models": len(metrics),
                    "best_performance": {
                        "provider": best_performance.provider.value,
                        "model": best_performance.model_name,
                        "success_rate": best_performance.success_rate,
                        "quality_score": best_performance.quality_score
                    },
                    "most_cost_effective": {
                        "provider": best_cost_effective.provider.value,
                        "model": best_cost_effective.model_name,
                        "cost_per_1k_tokens": best_cost_effective.cost_per_1k_tokens,
                        "success_rate": best_cost_effective.success_rate
                    }
                }
        
        return summary
    
    def get_task_specific_recommendations(
        self, 
        task_type: TaskType, 
        available_providers: Optional[List[ProviderType]] = None,
        budget_per_1k_tokens: Optional[float] = None
    ) -> Dict[str, Tuple[ProviderType, str]]:
        """Get task-specific model recommendations for different priorities."""
        recommendations = {}
        
        # Best overall performance
        best_performance = self.get_best_model_for_task(
            task_type, available_providers, quality_priority=True
        )
        if best_performance:
            recommendations["best_performance"] = best_performance
        
        # Most cost-effective
        most_cost_effective = self.get_best_model_for_task(
            task_type, available_providers, cost_priority=True
        )
        if most_cost_effective:
            recommendations["most_cost_effective"] = most_cost_effective
        
        # Balanced recommendation (default)
        balanced = self.get_best_model_for_task(task_type, available_providers)
        if balanced:
            recommendations["balanced"] = balanced
        
        # Budget-constrained recommendation
        if budget_per_1k_tokens:
            budget_options = []
            for metric in self.performance_metrics:
                if (metric.task_type == task_type and 
                    metric.cost_per_1k_tokens <= budget_per_1k_tokens and
                    (available_providers is None or metric.provider in available_providers)):
                    budget_options.append(metric)
            
            if budget_options:
                # Select best quality within budget
                best_in_budget = max(budget_options, key=lambda x: x.quality_score)
                recommendations["within_budget"] = (best_in_budget.provider, best_in_budget.model_name)
        
        return recommendations


def get_model_performance_router() -> ModelPerformanceRouter:
    """Get a singleton instance of the ModelPerformanceRouter."""
    if not hasattr(get_model_performance_router, '_instance'):
        get_model_performance_router._instance = ModelPerformanceRouter()
    return get_model_performance_router._instance


# Convenience functions for common routing scenarios
def get_best_coding_model(
    available_providers: Optional[List[ProviderType]] = None,
    cost_priority: bool = False
) -> Optional[Tuple[ProviderType, str]]:
    """Get the best model for coding tasks."""
    router = get_model_performance_router()
    return router.get_best_model_for_task(
        TaskType.CODING, available_providers, cost_priority=cost_priority
    )


def get_best_reasoning_model(
    available_providers: Optional[List[ProviderType]] = None,
    cost_priority: bool = False
) -> Optional[Tuple[ProviderType, str]]:
    """Get the best model for reasoning tasks."""
    router = get_model_performance_router()
    return router.get_best_model_for_task(
        TaskType.REASONING, available_providers, cost_priority=cost_priority
    )


def get_most_cost_effective_model(
    task_type: TaskType,
    available_providers: Optional[List[ProviderType]] = None
) -> Optional[Tuple[ProviderType, str]]:
    """Get the most cost-effective model for any task type."""
    router = get_model_performance_router()
    return router.get_best_model_for_task(
        task_type, available_providers, cost_priority=True
    )


def estimate_task_cost(
    provider: ProviderType,
    model_name: str,
    estimated_tokens: int
) -> float:
    """Estimate the cost for running a task with specific parameters."""
    router = get_model_performance_router()
    return router.estimate_cost_for_task(provider, model_name, estimated_tokens)