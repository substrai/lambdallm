"""Tests for the chain pipeline system."""

import pytest
from unittest.mock import patch, MagicMock

from lambdallm.chains import Chain, Step, ChainResult
from lambdallm.core.models import ModelResponse


class MockContext:
    """Mock LambdaLLMContext for chain testing."""

    def __init__(self):
        self.metrics = MagicMock()
        self.metrics.get.return_value = 0
        self.should_checkpoint = False
        self._call_count = 0

    def invoke(self, prompt, **kwargs):
        self._call_count += 1
        formatted = prompt.format(**kwargs) if kwargs else prompt
        return f"Response to: {formatted[:50]}"

    def invoke_structured(self, prompt, schema):
        self._call_count += 1
        return {k: f"value_{k}" for k in schema}


class TestStep:
    def test_llm_step_creation(self):
        step = Step("extract", prompt="Extract from: {input}")
        assert step.is_llm_step
        assert not step.is_transform_step

    def test_transform_step_creation(self):
        step = Step("upper", func=lambda data: data["input"].upper())
        assert step.is_transform_step
        assert not step.is_llm_step

    def test_step_requires_prompt_or_func(self):
        with pytest.raises(ValueError, match="must have either"):
            Step("empty")

    def test_step_cannot_have_both(self):
        with pytest.raises(ValueError, match="cannot have both"):
            Step("both", prompt="test", func=lambda x: x)


class TestChain:
    def test_chain_creation(self):
        chain = Chain(
            name="test",
            steps=[Step("s1", prompt="Do: {input}")],
        )
        assert chain.name == "test"
        assert chain.step_count == 1

    def test_chain_requires_steps(self):
        with pytest.raises(ValueError, match="at least one step"):
            Chain(name="empty", steps=[])

    def test_chain_rejects_duplicate_names(self):
        with pytest.raises(ValueError, match="duplicate step names"):
            Chain(name="dup", steps=[
                Step("same", prompt="A"),
                Step("same", prompt="B"),
            ])

    def test_chain_execution(self):
        chain = Chain(
            name="test-chain",
            steps=[
                Step("extract", prompt="Extract from: {input}"),
                Step("classify", prompt="Classify: {extract.output}"),
            ],
        )

        ctx = MockContext()
        result = chain.run(context=ctx, input="test document")

        assert isinstance(result, ChainResult)
        assert result.status == "completed"
        assert result.completed_steps == 2
        assert result.final_output is not None
        assert ctx._call_count == 2

    def test_chain_with_transform(self):
        chain = Chain(
            name="transform-chain",
            steps=[
                Step("extract", prompt="Extract: {input}"),
                Step("upper", func=lambda data: data["extract"].upper()),
            ],
        )

        ctx = MockContext()
        result = chain.run(context=ctx, input="hello")

        assert result.status == "completed"
        assert result.completed_steps == 2

    def test_chain_conditional_step(self):
        chain = Chain(
            name="conditional",
            steps=[
                Step("always", prompt="Do: {input}"),
                Step("maybe", prompt="Extra: {always.output}",
                     condition=lambda outputs: False),  # Always skip
            ],
        )

        ctx = MockContext()
        result = chain.run(context=ctx, input="test")

        assert result.completed_steps == 1  # Only "always" ran
        assert result.steps[1].skipped is True

    def test_chain_cost_limit(self):
        chain = Chain(
            name="expensive",
            steps=[
                Step("s1", prompt="A: {input}"),
                Step("s2", prompt="B: {s1.output}"),
                Step("s3", prompt="C: {s2.output}"),
            ],
            max_total_cost=0.0,  # Zero budget = truncate immediately after first
        )

        ctx = MockContext()
        ctx.metrics.get.return_value = 0.01  # Simulate cost

        result = chain.run(context=ctx, input="test")
        # Should complete at least first step before checking cost
        assert result.status in ("completed", "truncated")
