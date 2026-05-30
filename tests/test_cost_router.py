"""Tests for cost-aware model routing with automatic fallback."""

import time
from unittest.mock import MagicMock

import pytest

from lambdallm.core.cost_router import (
    BudgetConfig,
    BudgetExceededError,
    CostRecord,
    CostRouter,
    ModelConfig,
    ModelTier,
    RoutingDecision,
)


@pytest.fixture
def sample_models() -> list[ModelConfig]:
    """Create sample models across all tiers."""
    return [
        ModelConfig(
            name="gpt-4o",
            tier=ModelTier.PREMIUM,
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
        ),
        ModelConfig(
            name="gpt-4o-mini",
            tier=ModelTier.STANDARD,
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
        ),
        ModelConfig(
            name="gpt-3.5-turbo",
            tier=ModelTier.ECONOMY,
            cost_per_1k_input_tokens=0.0001,
            cost_per_1k_output_tokens=0.0002,
        ),
    ]


@pytest.fixture
def router(sample_models: list[ModelConfig]) -> CostRouter:
    """Create a router with sample models and $10 budget."""
    budget = BudgetConfig(max_budget=10.0)
    r = CostRouter(budget=budget)
    r.register_models(sample_models)
    return r


class TestCostRouterBasicRouting:
    """Test basic routing behavior."""

    def test_routes_to_premium_when_under_budget(self, router: CostRouter) -> None:
        """Should route to premium tier when budget utilization is low."""
        decision = router.route(preferred_tier=ModelTier.PREMIUM)
        assert decision.selected_model.name == "gpt-4o"
        assert decision.actual_tier == ModelTier.PREMIUM
        assert decision.was_downgraded is False

    def test_routes_to_standard_when_requested(self, router: CostRouter) -> None:
        """Should route to standard tier when explicitly requested."""
        decision = router.route(preferred_tier=ModelTier.STANDARD)
        assert decision.selected_model.name == "gpt-4o-mini"
        assert decision.actual_tier == ModelTier.STANDARD
        assert decision.was_downgraded is False

    def test_downgrades_to_standard_near_premium_threshold(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should downgrade to standard when premium threshold is exceeded."""
        budget = BudgetConfig(max_budget=10.0, premium_threshold=0.7)
        router = CostRouter(budget=budget)
        router.register_models(sample_models)

        # Manually set cost to 75% of budget
        router._cumulative_cost = 7.5

        decision = router.route(preferred_tier=ModelTier.PREMIUM)
        assert decision.actual_tier == ModelTier.STANDARD
        assert decision.was_downgraded is True
        assert "Downgraded" in decision.reason

    def test_downgrades_to_economy_near_standard_threshold(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should downgrade to economy when standard threshold is exceeded."""
        budget = BudgetConfig(max_budget=10.0, standard_threshold=0.9)
        router = CostRouter(budget=budget)
        router.register_models(sample_models)

        # Set cost to 95% of budget
        router._cumulative_cost = 9.5

        decision = router.route(preferred_tier=ModelTier.PREMIUM)
        assert decision.actual_tier == ModelTier.ECONOMY
        assert decision.was_downgraded is True


class TestCostRecording:
    """Test cost recording and tracking."""

    def test_record_cost_updates_cumulative(self, router: CostRouter) -> None:
        """Should update cumulative cost when recording."""
        record = router.record_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        assert record.cost > 0
        assert router.cumulative_cost == record.cost

    def test_record_cost_multiple_requests(self, router: CostRouter) -> None:
        """Should accumulate costs across multiple requests."""
        router.record_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        router.record_cost("gpt-4o", input_tokens=2000, output_tokens=1000)
        assert len(router.cost_history) == 2
        assert router.cumulative_cost > 0

    def test_record_cost_unknown_model_raises(self, router: CostRouter) -> None:
        """Should raise ValueError for unregistered model."""
        with pytest.raises(ValueError, match="not registered"):
            router.record_cost("unknown-model", input_tokens=100, output_tokens=50)


class TestBudgetManagement:
    """Test budget management features."""

    def test_budget_exceeded_reject_policy(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should raise BudgetExceededError when policy is reject."""
        budget = BudgetConfig(max_budget=1.0, on_budget_exceeded="reject")
        router = CostRouter(budget=budget)
        router.register_models(sample_models)

        # Exceed budget
        router._cumulative_cost = 1.5

        with pytest.raises(BudgetExceededError):
            router.route(preferred_tier=ModelTier.PREMIUM)

    def test_budget_exceeded_economy_policy(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should fall back to economy when policy is use_economy."""
        budget = BudgetConfig(max_budget=1.0, on_budget_exceeded="use_economy")
        router = CostRouter(budget=budget)
        router.register_models(sample_models)

        router._cumulative_cost = 1.5

        decision = router.route(preferred_tier=ModelTier.PREMIUM)
        assert decision.actual_tier == ModelTier.ECONOMY

    def test_reset_budget_clears_state(self, router: CostRouter) -> None:
        """Should clear all cost state on reset."""
        router.record_cost("gpt-4o", input_tokens=5000, output_tokens=2000)
        assert router.cumulative_cost > 0

        router.reset_budget()
        assert router.cumulative_cost == 0.0
        assert len(router.cost_history) == 0

    def test_auto_reset_after_interval(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should auto-reset when reset interval elapses."""
        budget = BudgetConfig(max_budget=10.0, reset_interval_seconds=0.1)
        router = CostRouter(budget=budget)
        router.register_models(sample_models)

        router._cumulative_cost = 8.0
        router._last_reset = time.time() - 1.0  # 1 second ago

        # Accessing cumulative_cost triggers reset check
        assert router.cumulative_cost == 0.0


class TestCallbacks:
    """Test callback functionality."""

    def test_on_downgrade_callback_fires(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should fire downgrade callback when tier changes."""
        callback = MagicMock()
        budget = BudgetConfig(max_budget=10.0)
        router = CostRouter(budget=budget, on_downgrade=callback)
        router.register_models(sample_models)

        router._cumulative_cost = 7.5  # Past premium threshold

        router.route(preferred_tier=ModelTier.PREMIUM)
        callback.assert_called_once()
        decision = callback.call_args[0][0]
        assert isinstance(decision, RoutingDecision)
        assert decision.was_downgraded is True

    def test_budget_warning_callback(
        self, sample_models: list[ModelConfig]
    ) -> None:
        """Should fire warning callback when utilization >= 80%."""
        warning_cb = MagicMock()
        budget = BudgetConfig(max_budget=10.0)
        router = CostRouter(budget=budget, on_budget_warning=warning_cb)
        router.register_models(sample_models)

        # Set cost near 80%
        router._cumulative_cost = 7.9
        router.record_cost("gpt-4o", input_tokens=1000, output_tokens=500)

        warning_cb.assert_called_once()


class TestCostSummary:
    """Test cost summary reporting."""

    def test_get_cost_summary_structure(self, router: CostRouter) -> None:
        """Should return properly structured summary."""
        router.record_cost("gpt-4o", input_tokens=1000, output_tokens=500)
        router.record_cost("gpt-3.5-turbo", input_tokens=2000, output_tokens=1000)

        summary = router.get_cost_summary()
        assert "total_cost" in summary
        assert "budget_max" in summary
        assert "by_tier" in summary
        assert summary["request_count"] == 2
        assert summary["by_tier"]["premium"]["requests"] == 1
        assert summary["by_tier"]["economy"]["requests"] == 1


class TestBudgetConfigValidation:
    """Test budget configuration validation."""

    def test_invalid_threshold_order_raises(self) -> None:
        """Should raise if premium_threshold >= standard_threshold."""
        with pytest.raises(ValueError, match="premium_threshold"):
            BudgetConfig(
                max_budget=10.0,
                premium_threshold=0.9,
                standard_threshold=0.7,
            )

    def test_zero_budget_raises(self) -> None:
        """Should raise if max_budget is not positive."""
        with pytest.raises(ValueError, match="max_budget must be positive"):
            BudgetConfig(max_budget=0.0)
