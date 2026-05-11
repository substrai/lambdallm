"""Tests for the observability system."""

import pytest
from lambdallm.observability.tracer import Tracer, Span
from lambdallm.observability.metrics import MetricsEmitter, MetricPoint
from lambdallm.observability.cost_tracker import CostTracker, CostEntry, CostReport
from lambdallm.observability.router import CostAwareRouter, RouterConfig, RoutingRule
from lambdallm.observability.ab_testing import Experiment, Variant, ABTestManager
from lambdallm.observability.prompt_analytics import PromptAnalytics


class TestTracer:
    def test_span_creation(self):
        tracer = Tracer(enabled=True)
        with tracer.span("test.operation") as span:
            span.set_attribute("key", "value")

        summary = tracer.get_trace_summary()
        assert summary["span_count"] == 1
        assert summary["errors"] == 0

    def test_span_error_tracking(self):
        tracer = Tracer(enabled=True)
        with pytest.raises(ValueError):
            with tracer.span("failing.op") as span:
                raise ValueError("test error")

        summary = tracer.get_trace_summary()
        assert summary["errors"] == 1

    def test_nested_spans(self):
        tracer = Tracer(enabled=True)
        with tracer.span("parent") as parent:
            with tracer.span("child") as child:
                pass

        summary = tracer.get_trace_summary()
        assert summary["span_count"] == 2

    def test_disabled_tracer(self):
        tracer = Tracer(enabled=False)
        with tracer.span("noop") as span:
            span.set_attribute("key", "value")  # Should not raise

        assert tracer.get_trace_summary()["span_count"] == 0


class TestMetricsEmitter:
    def test_record_metric(self):
        emitter = MetricsEmitter(enabled=True)
        emitter.record("test.metric", 42.0, unit="Count")
        assert emitter.pending_count == 1

    def test_record_model_invocation(self):
        emitter = MetricsEmitter(enabled=True)
        emitter.record_model_invocation(
            model_id="claude-3-haiku",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200,
            cost_usd=0.001,
        )
        assert emitter.pending_count == 5  # 5 metrics recorded

    def test_disabled_emitter(self):
        emitter = MetricsEmitter(enabled=False)
        emitter.record("test", 1.0)
        assert emitter.pending_count == 0


class TestCostAwareRouter:
    def test_rule_based_routing(self):
        config = RouterConfig(
            strategy="cost-optimized",
            rules=[
                RoutingRule(condition="input_tokens < 100", model="fast", priority=1),
            ],
        )
        router = CostAwareRouter(config)

        class MockContext:
            total_cost = 0
            remaining_time_ms = 30000

        # Short prompt should match rule
        result = router.select("short prompt", MockContext())
        assert "haiku" in result.model_id

    def test_cost_optimized_strategy(self):
        router = CostAwareRouter(RouterConfig(strategy="cost-optimized"))

        class MockContext:
            total_cost = 0
            remaining_time_ms = 30000

        result = router.select("a" * 100, MockContext())
        assert "haiku" in result.model_id  # Should pick cheapest


class TestABTesting:
    def test_experiment_creation(self):
        exp = Experiment(
            name="test-exp",
            variants=[
                Variant("control", weight=0.7),
                Variant("treatment", weight=0.3),
            ],
        )
        assert len(exp.variants) == 2

    def test_variant_selection(self):
        exp = Experiment(
            name="test",
            variants=[
                Variant("a", weight=0.5),
                Variant("b", weight=0.5),
            ],
        )

        # Run many selections, both should appear
        selections = [exp.select_variant().name for _ in range(100)]
        assert "a" in selections
        assert "b" in selections

    def test_sticky_sessions(self):
        exp = Experiment(
            name="test",
            variants=[
                Variant("a", weight=0.5),
                Variant("b", weight=0.5),
            ],
        )

        # Same user_id should always get same variant
        v1 = exp.select_variant(user_id="user-123")
        v2 = exp.select_variant(user_id="user-123")
        assert v1.name == v2.name

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            Experiment(
                name="bad",
                variants=[
                    Variant("a", weight=0.3),
                    Variant("b", weight=0.3),
                ],
            )


class TestPromptAnalytics:
    def test_record_and_report(self):
        analytics = PromptAnalytics()
        analytics.record("summarize", tokens_in=100, tokens_out=50, latency_ms=200, cost_usd=0.001)
        analytics.record("summarize", tokens_in=150, tokens_out=60, latency_ms=250, cost_usd=0.0015)

        report = analytics.get_report("summarize")
        assert report is not None
        assert report.invocation_count == 2
        assert report.avg_cost == pytest.approx(0.00125, rel=0.01)

    def test_version_comparison(self):
        analytics = PromptAnalytics()
        analytics.record("summarize", version="1.0", cost_usd=0.002)
        analytics.record("summarize", version="2.0", cost_usd=0.001)

        comparison = analytics.compare_versions("summarize")
        assert len(comparison) == 2

    def test_optimization_suggestions(self):
        analytics = PromptAnalytics()
        # Record expensive prompt
        for _ in range(10):
            analytics.record("expensive", tokens_in=1000, tokens_out=100, cost_usd=0.02, latency_ms=6000)

        suggestions = analytics.get_optimization_suggestions("expensive")
        assert len(suggestions) > 0
        assert any("cost" in s.lower() for s in suggestions)
