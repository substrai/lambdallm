"""A/B testing for prompts and models in LambdaLLM.

Route a percentage of traffic to different prompt versions or models,
compare metrics, and automatically promote winners.
"""

import random
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("lambdallm")


@dataclass
class Variant:
    """A single variant in an A/B test."""

    name: str
    weight: float  # 0.0 to 1.0 (percentage of traffic)
    prompt_template: Optional[str] = None
    model_id: Optional[str] = None
    config_overrides: dict = field(default_factory=dict)


@dataclass
class ExperimentResult:
    """Metrics for a single variant."""

    variant_name: str
    request_count: int = 0
    avg_latency_ms: float = 0.0
    avg_cost_usd: float = 0.0
    avg_tokens_out: float = 0.0
    error_rate: float = 0.0
    quality_score: float = 0.0  # User-defined quality metric


@dataclass
class Experiment:
    """An A/B test experiment.

    Example:
        experiment = Experiment(
            name="summarize-prompt-v2",
            variants=[
                Variant("control", weight=0.7, prompt_template="Summarize: {text}"),
                Variant("treatment", weight=0.3, prompt_template="Concisely summarize: {text}"),
            ],
        )

        variant = experiment.select_variant()
        # Use variant.prompt_template for this request
        # Record metrics after execution
        experiment.record_result(variant.name, latency_ms=150, cost_usd=0.001)
    """

    name: str
    variants: list[Variant]
    description: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "active"  # active | paused | completed
    min_sample_size: int = 100  # Minimum requests before declaring winner

    def __post_init__(self):
        # Validate weights sum to ~1.0
        total_weight = sum(v.weight for v in self.variants)
        if abs(total_weight - 1.0) > 0.01:
            raise ValueError(f"Variant weights must sum to 1.0, got {total_weight}")

    def select_variant(self, user_id: Optional[str] = None) -> Variant:
        """Select a variant for this request.

        Uses weighted random selection. If user_id is provided,
        ensures the same user always gets the same variant (sticky sessions).

        Args:
            user_id: Optional user ID for consistent assignment.

        Returns:
            Selected Variant.
        """
        if self.status != "active":
            return self.variants[0]  # Return control if experiment is not active

        if user_id:
            # Deterministic assignment based on user_id hash
            hash_value = hash(f"{self.name}:{user_id}") % 1000 / 1000.0
        else:
            hash_value = random.random()

        cumulative = 0.0
        for variant in self.variants:
            cumulative += variant.weight
            if hash_value <= cumulative:
                return variant

        return self.variants[-1]  # Fallback

    def record_result(
        self,
        variant_name: str,
        latency_ms: float = 0,
        cost_usd: float = 0,
        tokens_out: int = 0,
        error: bool = False,
        quality_score: float = 0,
    ) -> None:
        """Record metrics for a variant execution.

        In production, this would write to DynamoDB or CloudWatch.
        """
        logger.info(json.dumps({
            "event": "ab_test.result",
            "experiment": self.name,
            "variant": variant_name,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "tokens_out": tokens_out,
            "error": error,
            "quality_score": quality_score,
        }))

    def get_results(self) -> list[ExperimentResult]:
        """Get aggregated results for all variants.

        In production, this would query DynamoDB/CloudWatch.
        Returns placeholder structure for now.
        """
        return [
            ExperimentResult(variant_name=v.name)
            for v in self.variants
        ]


class ABTestManager:
    """Manages multiple A/B test experiments.

    Example:
        manager = ABTestManager()
        manager.register(experiment)

        # In handler:
        variant = manager.get_variant("summarize-prompt-v2")
        # Use variant config...
        manager.record("summarize-prompt-v2", variant.name, latency_ms=100)
    """

    def __init__(self):
        self._experiments: dict[str, Experiment] = {}

    def register(self, experiment: Experiment) -> None:
        """Register an experiment."""
        self._experiments[experiment.name] = experiment
        logger.info(f"Registered A/B experiment: {experiment.name} ({len(experiment.variants)} variants)")

    def get_variant(self, experiment_name: str, user_id: Optional[str] = None) -> Optional[Variant]:
        """Get the selected variant for an experiment."""
        experiment = self._experiments.get(experiment_name)
        if not experiment or experiment.status != "active":
            return None
        return experiment.select_variant(user_id)

    def record(self, experiment_name: str, variant_name: str, **metrics) -> None:
        """Record metrics for a variant."""
        experiment = self._experiments.get(experiment_name)
        if experiment:
            experiment.record_result(variant_name, **metrics)

    def list_experiments(self) -> list[dict]:
        """List all registered experiments."""
        return [
            {"name": e.name, "status": e.status, "variants": len(e.variants)}
            for e in self._experiments.values()
        ]
