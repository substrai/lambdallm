"""Tests for multi-model chain with conditional routing."""

import pytest

from lambdallm.chains.multi_model import (
    ChainBranch,
    ComplexityLevel,
    ModelProfile,
    ModelRouter,
    ModelSelectionResult,
    MultiModelChain,
    MultiModelChainResult,
    MultiModelStep,
    RoutingCondition,
    SelectionStrategy,
    StepResult,
)


# --- Fixtures ---

@pytest.fixture
def cheap_model():
    return ModelProfile(
        model_id="gpt-3.5-turbo",
        cost_per_1k_tokens=0.002,
        avg_latency_ms=200.0,
        quality_score=0.7,
        max_tokens=4096,
        complexity_ceiling=ComplexityLevel.MEDIUM,
    )


@pytest.fixture
def expensive_model():
    return ModelProfile(
        model_id="gpt-4",
        cost_per_1k_tokens=0.03,
        avg_latency_ms=800.0,
        quality_score=0.95,
        max_tokens=8192,
        complexity_ceiling=ComplexityLevel.CRITICAL,
    )


@pytest.fixture
def fast_model():
    return ModelProfile(
        model_id="claude-3-haiku",
        cost_per_1k_tokens=0.001,
        avg_latency_ms=100.0,
        quality_score=0.65,
        max_tokens=4096,
        complexity_ceiling=ComplexityLevel.LOW,
    )


@pytest.fixture
def router(cheap_model, expensive_model, fast_model):
    return ModelRouter(models=[cheap_model, expensive_model, fast_model])


@pytest.fixture
def basic_chain(router):
    steps = [
        MultiModelStep(
            name="classify",
            prompt="Classify this: {input}",
            complexity=ComplexityLevel.LOW,
            strategy=SelectionStrategy.LOWEST_COST,
        ),
        MultiModelStep(
            name="analyze",
            prompt="Analyze: {classify.output}",
            complexity=ComplexityLevel.HIGH,
            strategy=SelectionStrategy.HIGHEST_QUALITY,
        ),
    ]
    return MultiModelChain(name="test-chain", steps=steps, router=router)


# --- ModelRouter Tests ---

class TestModelRouter:
    def test_select_lowest_cost(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.LOWEST_COST,
            complexity=ComplexityLevel.LOW,
        )
        result = router.select_model(step)
        assert result.selected_model.model_id == "claude-3-haiku"

    def test_select_lowest_latency(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.LOWEST_LATENCY,
            complexity=ComplexityLevel.LOW,
        )
        result = router.select_model(step)
        assert result.selected_model.model_id == "claude-3-haiku"

    def test_select_highest_quality(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.HIGHEST_QUALITY,
            complexity=ComplexityLevel.LOW,
        )
        result = router.select_model(step)
        assert result.selected_model.model_id == "gpt-4"

    def test_filter_by_complexity(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.LOWEST_COST,
            complexity=ComplexityLevel.CRITICAL,
        )
        result = router.select_model(step)
        # Only gpt-4 handles CRITICAL complexity
        assert result.selected_model.model_id == "gpt-4"

    def test_filter_by_cost_constraint(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.HIGHEST_QUALITY,
            complexity=ComplexityLevel.LOW,
            max_cost=0.005,
        )
        result = router.select_model(step)
        # gpt-4 is too expensive, gpt-3.5 is too expensive, only haiku fits
        assert result.selected_model.model_id in ["claude-3-haiku", "gpt-3.5-turbo"]

    def test_filter_by_latency_constraint(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.HIGHEST_QUALITY,
            complexity=ComplexityLevel.LOW,
            max_latency_ms=150.0,
        )
        result = router.select_model(step)
        assert result.selected_model.model_id == "claude-3-haiku"

    def test_no_suitable_model_raises(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.LOWEST_COST,
            complexity=ComplexityLevel.CRITICAL,
            max_cost=0.001,
        )
        with pytest.raises(ValueError, match="No suitable model"):
            router.select_model(step)

    def test_fallback_model(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.LOWEST_COST,
            complexity=ComplexityLevel.CRITICAL,
            max_cost=0.001,
            fallback_model="gpt-3.5-turbo",
        )
        result = router.select_model(step)
        assert result.selected_model.model_id == "gpt-3.5-turbo"
        assert "Fallback" in result.reason

    def test_round_robin_strategy(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.ROUND_ROBIN,
            complexity=ComplexityLevel.LOW,
        )
        first = router.select_model(step)
        second = router.select_model(step)
        # Should cycle through models
        assert first.selected_model.model_id != second.selected_model.model_id

    def test_threshold_strategy(self, router):
        step = MultiModelStep(
            name="test",
            prompt="Hello",
            strategy=SelectionStrategy.THRESHOLD,
            complexity=ComplexityLevel.MEDIUM,
            min_quality=0.8,
        )
        result = router.select_model(step)
        # Should pick cheapest model with quality >= 0.8
        assert result.selected_model.quality_score >= 0.8

    def test_add_and_remove_model(self, router):
        new_model = ModelProfile(model_id="new-model", cost_per_1k_tokens=0.005)
        router.add_model(new_model)
        assert router.get_model("new-model") is not None

        removed = router.remove_model("new-model")
        assert removed is True
        assert router.get_model("new-model") is None

    def test_remove_nonexistent_model(self, router):
        removed = router.remove_model("nonexistent")
        assert removed is False


# --- MultiModelChain Tests ---

class TestMultiModelChain:
    def test_basic_execution(self, basic_chain):
        result = basic_chain.execute(input="test data")
        assert result.success is True
        assert result.step_count == 2
        assert len(result.models_used) == 2

    def test_chain_with_model_invoker(self, router):
        def mock_invoker(model_id, prompt):
            return f"Response from {model_id}"

        steps = [
            MultiModelStep(
                name="step1",
                prompt="Process: {input}",
                complexity=ComplexityLevel.LOW,
                strategy=SelectionStrategy.LOWEST_COST,
            ),
        ]
        chain = MultiModelChain(name="test", steps=steps, router=router)
        result = chain.execute(input="data", model_invoker=mock_invoker)
        assert result.success is True
        assert "Response from" in result.final_output

    def test_chain_cost_budget_exceeded(self, router):
        steps = [
            MultiModelStep(
                name="step1",
                prompt="First: {input}",
                complexity=ComplexityLevel.HIGH,
                strategy=SelectionStrategy.HIGHEST_QUALITY,
            ),
            MultiModelStep(
                name="step2",
                prompt="Second: {step1.output}",
                complexity=ComplexityLevel.HIGH,
                strategy=SelectionStrategy.HIGHEST_QUALITY,
            ),
        ]
        chain = MultiModelChain(
            name="test", steps=steps, router=router, max_total_cost=0.001
        )
        result = chain.execute(input="data")
        # First step costs 0.03 which exceeds budget for second step
        # But first step runs, second might be blocked
        assert result.step_count >= 1

    def test_chain_with_transform_step(self, router):
        steps = [
            MultiModelStep(
                name="step1",
                prompt="Process: {input}",
                complexity=ComplexityLevel.LOW,
                strategy=SelectionStrategy.LOWEST_COST,
            ),
            MultiModelStep(
                name="transform",
                func=lambda ctx: ctx.get("step1", "").upper(),
                complexity=ComplexityLevel.LOW,
            ),
        ]
        chain = MultiModelChain(name="test", steps=steps, router=router)
        result = chain.execute(input="data")
        assert result.success is True
        assert result.step_count == 2

    def test_chain_stop_on_failure(self, router):
        def failing_func(ctx):
            raise RuntimeError("Step failed")

        steps = [
            MultiModelStep(name="fail", func=failing_func, complexity=ComplexityLevel.LOW),
            MultiModelStep(
                name="never_reached",
                prompt="Process: {fail.output}",
                complexity=ComplexityLevel.LOW,
            ),
        ]
        chain = MultiModelChain(name="test", steps=steps, router=router, stop_on_failure=True)
        result = chain.execute()
        assert result.success is False
        assert result.step_count == 1

    def test_chain_continue_on_failure(self, router):
        def failing_func(ctx):
            raise RuntimeError("Step failed")

        steps = [
            MultiModelStep(name="fail", func=failing_func, complexity=ComplexityLevel.LOW),
            MultiModelStep(
                name="continues",
                prompt="Process: {input}",
                complexity=ComplexityLevel.LOW,
            ),
        ]
        chain = MultiModelChain(name="test", steps=steps, router=router, stop_on_failure=False)
        result = chain.execute(input="data")
        assert result.step_count == 2

    def test_chain_branching(self, router):
        condition = RoutingCondition(
            name="is_complex",
            predicate=lambda ctx: ctx.get("input", "") == "complex",
            priority=1,
        )
        steps = [
            MultiModelStep(
                name="branch_step",
                prompt="Default path: {input}",
                complexity=ComplexityLevel.LOW,
                strategy=SelectionStrategy.LOWEST_COST,
                branches=[
                    ChainBranch(name="complex_branch", condition=condition),
                    ChainBranch(name="default_branch", condition=None),
                ],
            ),
        ]
        chain = MultiModelChain(name="test", steps=steps, router=router)

        # With complex input
        result = chain.execute(input="complex")
        assert result.step_results[0].branch_taken == "complex_branch"

        # With simple input
        result = chain.execute(input="simple")
        assert result.step_results[0].branch_taken == "default_branch"

    def test_chain_validation_empty_steps(self, router):
        with pytest.raises(ValueError, match="must have at least one step"):
            MultiModelChain(name="test", steps=[], router=router)

    def test_chain_validation_duplicate_names(self, router):
        steps = [
            MultiModelStep(name="dup", prompt="A", complexity=ComplexityLevel.LOW),
            MultiModelStep(name="dup", prompt="B", complexity=ComplexityLevel.LOW),
        ]
        with pytest.raises(ValueError, match="duplicate step names"):
            MultiModelChain(name="test", steps=steps, router=router)

    def test_chain_validation_no_models(self):
        router = ModelRouter(models=[])
        steps = [
            MultiModelStep(name="s1", prompt="Hello", complexity=ComplexityLevel.LOW),
        ]
        with pytest.raises(ValueError, match="requires at least one model"):
            MultiModelChain(name="test", steps=steps, router=router)

    def test_estimate_cost(self, basic_chain):
        cost = basic_chain.estimate_cost()
        assert cost > 0.0

    def test_estimate_latency(self, basic_chain):
        latency = basic_chain.estimate_latency()
        assert latency > 0.0

    def test_get_step(self, basic_chain):
        step = basic_chain.get_step("classify")
        assert step is not None
        assert step.name == "classify"

        missing = basic_chain.get_step("nonexistent")
        assert missing is None


# --- MultiModelStep Tests ---

class TestMultiModelStep:
    def test_step_requires_prompt_or_func(self):
        with pytest.raises(ValueError, match="must have either"):
            MultiModelStep(name="invalid")

    def test_step_cannot_have_both(self):
        with pytest.raises(ValueError, match="cannot have both"):
            MultiModelStep(name="invalid", prompt="Hello", func=lambda x: x)

    def test_is_llm_step(self):
        step = MultiModelStep(name="llm", prompt="Hello", complexity=ComplexityLevel.LOW)
        assert step.is_llm_step is True

        func_step = MultiModelStep(name="func", func=lambda x: x, complexity=ComplexityLevel.LOW)
        assert func_step.is_llm_step is False

    def test_is_branching_step(self):
        step = MultiModelStep(name="branch", prompt="Hello", complexity=ComplexityLevel.LOW)
        assert step.is_branching_step is False

        step.branches = [ChainBranch(name="b1")]
        assert step.is_branching_step is True

    def test_has_constraints(self):
        step = MultiModelStep(name="s", prompt="Hello", complexity=ComplexityLevel.LOW)
        assert step.has_constraints is False

        step_with_cost = MultiModelStep(
            name="s", prompt="Hello", complexity=ComplexityLevel.LOW, max_cost=0.01
        )
        assert step_with_cost.has_constraints is True


# --- ModelProfile Tests ---

class TestModelProfile:
    def test_cost_efficiency(self):
        model = ModelProfile(model_id="test", cost_per_1k_tokens=0.01, quality_score=0.8)
        assert model.cost_efficiency == 80.0

    def test_cost_efficiency_zero_cost(self):
        model = ModelProfile(model_id="test", cost_per_1k_tokens=0.0, quality_score=0.8)
        assert model.cost_efficiency == float("inf")

    def test_speed_quality_ratio(self):
        model = ModelProfile(model_id="test", avg_latency_ms=1000.0, quality_score=0.8)
        assert model.speed_quality_ratio == 0.8
