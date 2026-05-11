"""Metrics emission for LambdaLLM.

Publishes custom metrics to CloudWatch for monitoring:
- Request latency, error rates, throughput
- Token usage and cost per model
- Agent iterations and tool call counts
- Budget utilization percentage
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("lambdallm")


@dataclass
class MetricPoint:
    """A single metric data point."""

    name: str
    value: float
    unit: str = "None"  # None | Milliseconds | Count | Bytes
    dimensions: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class MetricsEmitter:
    """Emits custom metrics to CloudWatch.

    Batches metrics and flushes at the end of each Lambda invocation
    to minimize API calls (cost optimization).

    Metrics emitted automatically by the framework:
    - lambdallm.handler.latency (Milliseconds)
    - lambdallm.handler.errors (Count)
    - lambdallm.model.invocations (Count)
    - lambdallm.model.tokens_in (Count)
    - lambdallm.model.tokens_out (Count)
    - lambdallm.model.cost_usd (None - treated as gauge)
    - lambdallm.model.latency (Milliseconds)
    - lambdallm.agent.iterations (Count)
    - lambdallm.agent.tool_calls (Count)
    - lambdallm.budget.utilization_percent (None)

    Example:
        emitter = MetricsEmitter(namespace="MyApp/LambdaLLM")
        emitter.record("custom.metric", 42.0, unit="Count")
        emitter.flush()  # Sends all batched metrics to CloudWatch
    """

    def __init__(
        self,
        namespace: str = "LambdaLLM",
        enabled: bool = True,
        region: str = "us-east-1",
        batch_size: int = 20,
    ):
        self.namespace = namespace
        self.enabled = enabled
        self.region = region
        self.batch_size = batch_size
        self._buffer: list[MetricPoint] = []
        self._client = None

    def record(
        self,
        name: str,
        value: float,
        unit: str = "None",
        dimensions: Optional[dict] = None,
    ) -> None:
        """Record a metric data point.

        Args:
            name: Metric name (e.g., "model.latency").
            value: Metric value.
            unit: CloudWatch unit (Milliseconds, Count, Bytes, None).
            dimensions: Optional dimensions for filtering.
        """
        if not self.enabled:
            return

        point = MetricPoint(
            name=name,
            value=value,
            unit=unit,
            dimensions=dimensions or {},
        )
        self._buffer.append(point)

        # Auto-flush if buffer is full
        if len(self._buffer) >= self.batch_size:
            self.flush()

    def record_model_invocation(
        self,
        model_id: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        cost_usd: float,
    ) -> None:
        """Record metrics for a model invocation (convenience method)."""
        dims = {"ModelId": model_id}

        self.record("model.invocations", 1, "Count", dims)
        self.record("model.tokens_in", tokens_in, "Count", dims)
        self.record("model.tokens_out", tokens_out, "Count", dims)
        self.record("model.latency", latency_ms, "Milliseconds", dims)
        self.record("model.cost_usd", cost_usd, "None", dims)

    def record_handler_execution(
        self,
        handler_name: str,
        latency_ms: float,
        status_code: int,
        error: bool = False,
    ) -> None:
        """Record metrics for a handler execution."""
        dims = {"Handler": handler_name}

        self.record("handler.latency", latency_ms, "Milliseconds", dims)
        self.record("handler.invocations", 1, "Count", dims)
        if error:
            self.record("handler.errors", 1, "Count", dims)
        self.record("handler.status_code", status_code, "Count", dims)

    def record_agent_execution(
        self,
        agent_name: str,
        iterations: int,
        tool_calls: int,
        cost_usd: float,
        status: str,
    ) -> None:
        """Record metrics for an agent execution."""
        dims = {"Agent": agent_name}

        self.record("agent.iterations", iterations, "Count", dims)
        self.record("agent.tool_calls", tool_calls, "Count", dims)
        self.record("agent.cost_usd", cost_usd, "None", dims)
        self.record("agent.executions", 1, "Count", dims)
        if status != "completed":
            self.record("agent.incomplete", 1, "Count", dims)

    def flush(self) -> None:
        """Flush all buffered metrics to CloudWatch."""
        if not self._buffer or not self.enabled:
            return

        try:
            self._publish_to_cloudwatch(self._buffer)
            logger.debug(f"Flushed {len(self._buffer)} metrics to CloudWatch")
        except Exception as e:
            # Log but don't fail the request for metrics issues
            logger.warning(f"Failed to flush metrics: {e}")
            # Fallback: emit as structured log
            self._emit_as_log(self._buffer)
        finally:
            self._buffer = []

    def _publish_to_cloudwatch(self, metrics: list[MetricPoint]) -> None:
        """Publish metrics to CloudWatch."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("cloudwatch", region_name=self.region)
            except ImportError:
                self._emit_as_log(metrics)
                return

        # Convert to CloudWatch format (batch up to 20 per API call)
        metric_data = []
        for point in metrics:
            datum = {
                "MetricName": point.name,
                "Value": point.value,
                "Unit": point.unit,
                "Timestamp": point.timestamp,
            }
            if point.dimensions:
                datum["Dimensions"] = [
                    {"Name": k, "Value": str(v)} for k, v in point.dimensions.items()
                ]
            metric_data.append(datum)

        # CloudWatch allows max 20 metrics per PutMetricData call
        for i in range(0, len(metric_data), 20):
            batch = metric_data[i : i + 20]
            self._client.put_metric_data(
                Namespace=self.namespace,
                MetricData=batch,
            )

    def _emit_as_log(self, metrics: list[MetricPoint]) -> None:
        """Fallback: emit metrics as structured logs (EMF format)."""
        for point in metrics:
            logger.info(json.dumps({
                "_aws": {
                    "Timestamp": int(point.timestamp * 1000),
                    "CloudWatchMetrics": [{
                        "Namespace": self.namespace,
                        "Dimensions": [list(point.dimensions.keys())] if point.dimensions else [[]],
                        "Metrics": [{"Name": point.name, "Unit": point.unit}],
                    }],
                },
                point.name: point.value,
                **point.dimensions,
            }))

    @property
    def pending_count(self) -> int:
        return len(self._buffer)
