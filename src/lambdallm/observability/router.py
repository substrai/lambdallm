"""Cost-aware model router for LambdaLLM.

Automatically selects the optimal model based on:
- Input complexity (token count)
- Remaining budget
- Quality requirements
- Custom routing rules
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.core.models import Model, ModelConfig

logger = logging.getLogger("lambdallm")

# Cost per 1K tokens (input) for routing decisions
MODEL_INPUT_COSTS = {
    Model.CLAUDE_3_HAIKU.value: 0.00025,
    Model.CLAUDE_3_SONNET.value: 0.003,
    Model.CLAUDE_3_5_SONNET.value: 0.003,
    Model.CLAUDE_3_OPUS.value: 0.015,
    Model.TITAN_TEXT_EXPRESS.value: 0.0002,
    Model.TITAN_TEXT_LITE.value: 0.00015,
    Model.LLAMA3_8B.value: 0.0003,
    Model.LLAMA3_70B.value: 0.00265,
}


@dataclass
class RoutingRule:
    """A single routing rule."""

    condition: str  # e.g., "input_tokens < 100", "budget_consumed > 0.8"
    model: str  # Model ID or alias
    priority: int = 0


@dataclass
class RouterConfig:
    """Configuration for the model router."""

    strategy: str = "cost-optimized"  # cost-optimized | quality-first | balanced
    models: dict = field(default_factory=dict)
    rules: list[RoutingRule] = field(default_factory=list)
    quality_threshold: float = 0.85
    default_model: str = Model.CLAUDE_3_HAIKU.value


class CostAwareRouter:
    """Selects the optimal model based on cost and quality constraints.

    Strategies:
    - cost-optimized: Always pick cheapest model that meets quality threshold
    - quality-first: Pick best model unless budget is critical
    - balanced: Weighted selection based on cost/quality tradeoff

    Example:
        router = CostAwareRouter(config=RouterConfig(
            strategy="cost-optimized",
            rules=[
                RoutingRule("input_tokens < 100", "fast"),
                RoutingRule("budget_consumed > 0.8", "fast"),
            ],
        ))

        model = router.select(prompt, context)
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()

    def select(self, prompt: str, context: Any) -> ModelConfig:
        """Select the best model for this request.

        Args:
            prompt: The formatted prompt (used to estimate tokens).
            context: LambdaLLMContext with budget info.

        Returns:
            ModelConfig for the selected model.
        """
        # Estimate input tokens
        input_tokens = len(prompt) // 4  # Rough estimate

        # Build evaluation context
        eval_context = {
            "input_tokens": input_tokens,
            "budget_consumed": self._get_budget_utilization(context),
            "remaining_time_ms": getattr(context, "remaining_time_ms", 900000),
        }

        # Check rules first (highest priority)
        for rule in sorted(self.config.rules, key=lambda r: r.priority, reverse=True):
            if self._evaluate_condition(rule.condition, eval_context):
                model_id = self._resolve_model_alias(rule.model)
                logger.debug(f"Router: rule matched '{rule.condition}' -> {model_id}")
                return ModelConfig(model_id=model_id)

        # Apply strategy
        if self.config.strategy == "cost-optimized":
            return self._select_cheapest(input_tokens, eval_context)
        elif self.config.strategy == "quality-first":
            return self._select_best_quality(eval_context)
        else:  # balanced
            return self._select_balanced(input_tokens, eval_context)

    def _select_cheapest(self, input_tokens: int, eval_context: dict) -> ModelConfig:
        """Select the cheapest model."""
        # If budget is critical, force cheapest
        if eval_context.get("budget_consumed", 0) > 0.9:
            return ModelConfig(model_id=Model.TITAN_TEXT_LITE.value)

        # For short prompts, use cheapest fast model
        if input_tokens < 200:
            return ModelConfig(model_id=Model.CLAUDE_3_HAIKU.value)

        # Default to Haiku (best cost/quality ratio)
        return ModelConfig(model_id=Model.CLAUDE_3_HAIKU.value)

    def _select_best_quality(self, eval_context: dict) -> ModelConfig:
        """Select the highest quality model (unless budget critical)."""
        if eval_context.get("budget_consumed", 0) > 0.9:
            return ModelConfig(model_id=Model.CLAUDE_3_HAIKU.value)

        return ModelConfig(model_id=Model.CLAUDE_3_5_SONNET.value)

    def _select_balanced(self, input_tokens: int, eval_context: dict) -> ModelConfig:
        """Balance cost and quality."""
        budget_used = eval_context.get("budget_consumed", 0)

        if budget_used > 0.8:
            return ModelConfig(model_id=Model.CLAUDE_3_HAIKU.value)
        elif input_tokens > 1000:
            return ModelConfig(model_id=Model.CLAUDE_3_SONNET.value)
        else:
            return ModelConfig(model_id=Model.CLAUDE_3_HAIKU.value)

    def _evaluate_condition(self, condition: str, context: dict) -> bool:
        """Safely evaluate a routing condition."""
        try:
            # Simple condition parser (no eval for security)
            parts = condition.split()
            if len(parts) == 3:
                var, op, value = parts
                var_value = context.get(var, 0)
                threshold = float(value)

                if op == "<":
                    return var_value < threshold
                elif op == ">":
                    return var_value > threshold
                elif op == ">=":
                    return var_value >= threshold
                elif op == "<=":
                    return var_value <= threshold
                elif op == "==":
                    return var_value == threshold
        except (ValueError, KeyError):
            pass

        return False

    def _resolve_model_alias(self, alias: str) -> str:
        """Resolve a model alias (fast/smart/powerful) to model ID."""
        aliases = {
            "fast": Model.CLAUDE_3_HAIKU.value,
            "smart": Model.CLAUDE_3_SONNET.value,
            "powerful": Model.CLAUDE_3_5_SONNET.value,
            "cheapest": Model.TITAN_TEXT_LITE.value,
        }
        return aliases.get(alias, alias)

    def _get_budget_utilization(self, context: Any) -> float:
        """Get current budget utilization from context."""
        if hasattr(context, "_budget_exceeded") and context._budget_exceeded:
            return 1.0
        if hasattr(context, "total_cost") and hasattr(context, "model"):
            # Rough estimate based on current cost
            return min(context.total_cost / 0.50, 1.0)  # Assume $0.50 per-request budget
        return 0.0
