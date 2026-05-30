"""Cost-aware model routing with automatic fallback.

Tracks cumulative cost per session/tenant and automatically downgrades
to cheaper models as budget limits are approached. Supports configurable
model tiers (premium/standard/economy) with threshold-based routing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class ModelTier(str, Enum):
    """Model pricing tiers from most to least expensive."""

    PREMIUM = "premium"
    STANDARD = "standard"
    ECONOMY = "economy"


@dataclass
class ModelConfig:
    """Configuration for a model within a tier."""

    name: str
    tier: ModelTier
    cost_per_1k_input_tokens: float
    cost_per_1k_output_tokens: float
    max_context_length: int = 128000
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def avg_cost_per_1k_tokens(self) -> float:
        """Average cost across input and output tokens."""
        return (self.cost_per_1k_input_tokens + self.cost_per_1k_output_tokens) / 2


@dataclass
class BudgetConfig:
    """Budget configuration with tier thresholds."""

    max_budget: float
    premium_threshold: float = 0.7
    standard_threshold: float = 0.9
    reset_interval_seconds: Optional[float] = None
    on_budget_exceeded: str = "use_economy"

    def __post_init__(self) -> None:
        if self.premium_threshold >= self.standard_threshold:
            raise ValueError(
                "premium_threshold must be less than standard_threshold"
            )
        if self.max_budget <= 0:
            raise ValueError("max_budget must be positive")


@dataclass
class CostRecord:
    """Record of a single cost event."""

    model_name: str
    tier: ModelTier
    input_tokens: int
    output_tokens: int
    cost: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""

    selected_model: ModelConfig
    original_tier: ModelTier
    actual_tier: ModelTier
    was_downgraded: bool
    budget_remaining: float
    budget_utilization: float
    reason: str


class BudgetExceededError(Exception):
    """Raised when the budget is exceeded and policy is reject."""
    pass


class CostRouter:
    """Cost-aware model router with automatic tier fallback.

    Tracks cumulative spending and automatically routes requests to
    cheaper model tiers as budget thresholds are approached.

    Example:
        >>> router = CostRouter(budget=BudgetConfig(max_budget=10.0))
        >>> router.register_model(ModelConfig(
        ...     name="gpt-4", tier=ModelTier.PREMIUM,
        ...     cost_per_1k_input_tokens=0.03,
        ...     cost_per_1k_output_tokens=0.06
        ... ))
        >>> decision = router.route(preferred_tier=ModelTier.PREMIUM)
    """

    def __init__(
        self,
        budget: BudgetConfig,
        on_downgrade: Optional[Callable[[RoutingDecision], None]] = None,
        on_budget_warning: Optional[Callable[[float, float], None]] = None,
    ) -> None:
        self._budget = budget
        self._models: dict[ModelTier, list[ModelConfig]] = {
            ModelTier.PREMIUM: [],
            ModelTier.STANDARD: [],
            ModelTier.ECONOMY: [],
        }
        self._cost_history: list[CostRecord] = []
        self._cumulative_cost: float = 0.0
        self._last_reset: float = time.time()
        self._on_downgrade = on_downgrade
        self._on_budget_warning = on_budget_warning

    @property
    def cumulative_cost(self) -> float:
        """Current cumulative cost after applying any resets."""
        self._maybe_reset()
        return self._cumulative_cost

    @property
    def budget_remaining(self) -> float:
        """Remaining budget."""
        return max(0.0, self._budget.max_budget - self.cumulative_cost)

    @property
    def budget_utilization(self) -> float:
        """Budget utilization as a fraction (0.0 to 1.0+)."""
        if self._budget.max_budget == 0:
            return 1.0
        return self.cumulative_cost / self._budget.max_budget

    @property
    def cost_history(self) -> list[CostRecord]:
        """Read-only access to cost history."""
        return list(self._cost_history)

    def register_model(self, model: ModelConfig) -> None:
        """Register a model for routing."""
        self._models[model.tier].append(model)

    def register_models(self, models: list[ModelConfig]) -> None:
        """Register multiple models for routing."""
        for model in models:
            self.register_model(model)

    def get_available_tier(self) -> ModelTier:
        """Determine the highest tier available given current budget usage."""
        self._maybe_reset()
        utilization = self.budget_utilization

        if utilization < self._budget.premium_threshold:
            return ModelTier.PREMIUM
        elif utilization < self._budget.standard_threshold:
            return ModelTier.STANDARD
        else:
            return ModelTier.ECONOMY

    def route(
        self,
        preferred_tier: ModelTier = ModelTier.PREMIUM,
        model_name: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a request to the appropriate model based on budget.

        Args:
            preferred_tier: The tier the caller would prefer to use.
            model_name: Optional specific model name to prefer within tier.

        Returns:
            RoutingDecision with the selected model and routing metadata.

        Raises:
            ValueError: If no models are registered for the resolved tier.
            BudgetExceededError: If budget is exceeded and policy is reject.
        """
        self._maybe_reset()
        available_tier = self.get_available_tier()

        if self.budget_utilization >= 1.0:
            if self._budget.on_budget_exceeded == "reject":
                raise BudgetExceededError(
                    f"Budget of ${self._budget.max_budget:.2f} exceeded. "
                    f"Current spend: ${self._cumulative_cost:.2f}"
                )

        tier_priority = [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.ECONOMY]
        preferred_idx = tier_priority.index(preferred_tier)
        available_idx = tier_priority.index(available_tier)
        actual_idx = max(preferred_idx, available_idx)
        actual_tier = tier_priority[actual_idx]

        selected_model = self._find_model(actual_tier, model_name)
        was_downgraded = actual_tier != preferred_tier

        decision = RoutingDecision(
            selected_model=selected_model,
            original_tier=preferred_tier,
            actual_tier=actual_tier,
            was_downgraded=was_downgraded,
            budget_remaining=self.budget_remaining,
            budget_utilization=self.budget_utilization,
            reason=self._build_reason(preferred_tier, actual_tier),
        )

        if was_downgraded and self._on_downgrade:
            self._on_downgrade(decision)

        return decision

    def record_cost(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Record the cost of a completed request.

        Args:
            model_name: Name of the model used.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens generated.

        Returns:
            The CostRecord created.
        """
        model = self._find_model_by_name(model_name)
        cost = (
            (input_tokens / 1000) * model.cost_per_1k_input_tokens
            + (output_tokens / 1000) * model.cost_per_1k_output_tokens
        )

        record = CostRecord(
            model_name=model_name,
            tier=model.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=cost,
        )

        self._cost_history.append(record)
        self._cumulative_cost += cost

        if self._on_budget_warning and self.budget_utilization >= 0.8:
            self._on_budget_warning(self._cumulative_cost, self._budget.max_budget)

        return record

    def reset_budget(self) -> None:
        """Manually reset the cumulative cost tracker."""
        self._cumulative_cost = 0.0
        self._cost_history.clear()
        self._last_reset = time.time()

    def get_cost_summary(self) -> dict[str, Any]:
        """Get a summary of cost usage by tier."""
        summary: dict[str, Any] = {
            "total_cost": self._cumulative_cost,
            "budget_max": self._budget.max_budget,
            "budget_remaining": self.budget_remaining,
            "budget_utilization": self.budget_utilization,
            "by_tier": {},
            "request_count": len(self._cost_history),
        }

        for tier in ModelTier:
            tier_records = [r for r in self._cost_history if r.tier == tier]
            summary["by_tier"][tier.value] = {
                "cost": sum(r.cost for r in tier_records),
                "requests": len(tier_records),
                "total_tokens": sum(
                    r.input_tokens + r.output_tokens for r in tier_records
                ),
            }

        return summary

    def _maybe_reset(self) -> None:
        """Reset budget if the reset interval has elapsed."""
        if self._budget.reset_interval_seconds is None:
            return
        elapsed = time.time() - self._last_reset
        if elapsed >= self._budget.reset_interval_seconds:
            self.reset_budget()

    def _find_model(
        self, tier: ModelTier, model_name: Optional[str] = None
    ) -> ModelConfig:
        """Find a model in the given tier, falling back to lower tiers."""
        tier_priority = [ModelTier.PREMIUM, ModelTier.STANDARD, ModelTier.ECONOMY]
        start_idx = tier_priority.index(tier)

        for idx in range(start_idx, len(tier_priority)):
            current_tier = tier_priority[idx]
            models = self._models[current_tier]
            if not models:
                continue
            if model_name:
                for m in models:
                    if m.name == model_name:
                        return m
            return models[0]

        raise ValueError("No models registered. Cannot route request.")

    def _find_model_by_name(self, model_name: str) -> ModelConfig:
        """Find a model by name across all tiers."""
        for tier_models in self._models.values():
            for model in tier_models:
                if model.name == model_name:
                    return model
        raise ValueError(f"Model '{model_name}' not registered")

    def _build_reason(self, preferred: ModelTier, actual: ModelTier) -> str:
        """Build a human-readable reason for the routing decision."""
        if preferred == actual:
            return f"Within budget for {actual.value} tier"
        return (
            f"Downgraded from {preferred.value} to {actual.value} "
            f"(budget utilization: {self.budget_utilization:.1%})"
        )
