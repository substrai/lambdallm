"""Prompt analytics for LambdaLLM.

Tracks prompt performance metrics over time:
- Latency per prompt template
- Cost per prompt template
- Success/failure rates
- Token efficiency (output quality vs tokens used)
- Version comparison
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("lambdallm")


@dataclass
class PromptMetrics:
    """Aggregated metrics for a prompt template."""

    prompt_name: str
    version: str = "1.0.0"
    invocation_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    total_latency_ms: float = 0.0
    error_count: int = 0
    parse_failure_count: int = 0  # Structured output parse failures

    @property
    def avg_cost(self) -> float:
        return self.total_cost_usd / max(self.invocation_count, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.invocation_count, 1)

    @property
    def avg_tokens_out(self) -> float:
        return self.total_tokens_out / max(self.invocation_count, 1)

    @property
    def success_rate(self) -> float:
        if self.invocation_count == 0:
            return 0.0
        return (self.invocation_count - self.error_count) / self.invocation_count

    @property
    def token_efficiency(self) -> float:
        """Ratio of output tokens to input tokens (higher = more efficient)."""
        if self.total_tokens_in == 0:
            return 0.0
        return self.total_tokens_out / self.total_tokens_in


class PromptAnalytics:
    """Tracks and reports on prompt template performance.

    Collects metrics per prompt invocation and provides
    aggregated views for optimization decisions.

    Example:
        analytics = PromptAnalytics()

        # Record after each prompt invocation
        analytics.record(
            prompt_name="summarize",
            version="1.0.0",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200,
            cost_usd=0.0003,
        )

        # Get performance report
        report = analytics.get_report("summarize")
        print(f"Avg cost: ${report.avg_cost:.6f}")
        print(f"Success rate: {report.success_rate:.1%}")
    """

    def __init__(self):
        self._metrics: dict[str, PromptMetrics] = {}

    def record(
        self,
        prompt_name: str,
        version: str = "1.0.0",
        tokens_in: int = 0,
        tokens_out: int = 0,
        latency_ms: float = 0,
        cost_usd: float = 0,
        error: bool = False,
        parse_failure: bool = False,
    ) -> None:
        """Record metrics for a prompt invocation."""
        key = f"{prompt_name}:{version}"

        if key not in self._metrics:
            self._metrics[key] = PromptMetrics(prompt_name=prompt_name, version=version)

        m = self._metrics[key]
        m.invocation_count += 1
        m.total_tokens_in += tokens_in
        m.total_tokens_out += tokens_out
        m.total_latency_ms += latency_ms
        m.total_cost_usd += cost_usd
        if error:
            m.error_count += 1
        if parse_failure:
            m.parse_failure_count += 1

        # Emit as structured log for CloudWatch Insights
        logger.info(json.dumps({
            "event": "prompt.invocation",
            "prompt_name": prompt_name,
            "version": version,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "cost_usd": cost_usd,
            "error": error,
        }))

    def get_report(self, prompt_name: str, version: Optional[str] = None) -> Optional[PromptMetrics]:
        """Get aggregated metrics for a prompt."""
        if version:
            return self._metrics.get(f"{prompt_name}:{version}")

        # Return latest version metrics
        matching = [m for k, m in self._metrics.items() if k.startswith(f"{prompt_name}:")]
        return matching[-1] if matching else None

    def compare_versions(self, prompt_name: str) -> list[dict]:
        """Compare metrics across prompt versions."""
        versions = [
            m for k, m in self._metrics.items() if k.startswith(f"{prompt_name}:")
        ]

        return [
            {
                "version": m.version,
                "invocations": m.invocation_count,
                "avg_cost": round(m.avg_cost, 6),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "success_rate": round(m.success_rate, 3),
                "token_efficiency": round(m.token_efficiency, 2),
            }
            for m in versions
        ]

    def get_all_reports(self) -> list[dict]:
        """Get summary of all tracked prompts."""
        return [
            {
                "prompt_name": m.prompt_name,
                "version": m.version,
                "invocations": m.invocation_count,
                "avg_cost": round(m.avg_cost, 6),
                "avg_latency_ms": round(m.avg_latency_ms, 1),
                "success_rate": round(m.success_rate, 3),
            }
            for m in self._metrics.values()
        ]

    def get_optimization_suggestions(self, prompt_name: str) -> list[str]:
        """Suggest optimizations based on metrics."""
        report = self.get_report(prompt_name)
        if not report:
            return []

        suggestions = []

        if report.avg_cost > 0.01:
            suggestions.append(
                f"High cost (${report.avg_cost:.4f}/call). Consider using a cheaper model "
                f"or reducing max_tokens."
            )

        if report.token_efficiency < 0.3:
            suggestions.append(
                f"Low token efficiency ({report.token_efficiency:.2f}). "
                f"Your prompts may be too verbose. Try shorter instructions."
            )

        if report.success_rate < 0.95:
            suggestions.append(
                f"Success rate below 95% ({report.success_rate:.1%}). "
                f"Check error logs and consider adding retry logic."
            )

        if report.parse_failure_count > report.invocation_count * 0.1:
            suggestions.append(
                "High parse failure rate. Add explicit JSON formatting instructions "
                "to your prompt or use output_schema with retry."
            )

        if report.avg_latency_ms > 5000:
            suggestions.append(
                f"High latency ({report.avg_latency_ms:.0f}ms). "
                f"Consider using a faster model (Haiku) or reducing max_tokens."
            )

        return suggestions
