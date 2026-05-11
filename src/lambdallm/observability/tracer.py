"""Distributed tracing for LambdaLLM.

Integrates with AWS X-Ray to provide end-to-end request tracing:
request → model call → tool execution → response.

Traces are auto-created by the @handler decorator (Observable by Default).
"""

import time
import uuid
import json
import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger("lambdallm")


@dataclass
class Span:
    """A single span in a distributed trace."""

    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    status: str = "ok"  # ok | error
    attributes: dict = field(default_factory=dict)
    events: list[dict] = field(default_factory=list)

    @property
    def duration_ms(self) -> float:
        if self.end_time:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def add_event(self, name: str, attributes: Optional[dict] = None) -> None:
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })

    def end(self, status: str = "ok") -> None:
        self.end_time = time.time()
        self.status = status

    def to_xray_segment(self) -> dict:
        """Convert to AWS X-Ray subsegment format."""
        segment = {
            "name": self.name,
            "id": self.span_id,
            "trace_id": self.trace_id,
            "start_time": self.start_time,
            "end_time": self.end_time or time.time(),
            "annotations": {k: v for k, v in self.attributes.items() if isinstance(v, (str, int, float, bool))},
            "metadata": {"lambdallm": self.attributes},
        }
        if self.parent_id:
            segment["parent_id"] = self.parent_id
        if self.status == "error":
            segment["fault"] = True
        return segment


class Tracer:
    """Distributed tracer for LambdaLLM operations.

    Creates hierarchical spans for:
    - Handler execution (root span)
    - Model invocations
    - Tool executions
    - Chain steps
    - State operations

    Integrates with X-Ray when running in Lambda, falls back to
    structured logging for local development.

    Example:
        tracer = Tracer()

        with tracer.span("model.invoke") as span:
            span.set_attribute("model_id", "claude-3-haiku")
            result = call_model(prompt)
            span.set_attribute("tokens_out", result.tokens_out)
    """

    def __init__(self, service_name: str = "lambdallm", enabled: bool = True):
        self.service_name = service_name
        self.enabled = enabled
        self._trace_id = self._generate_trace_id()
        self._spans: list[Span] = []
        self._active_span: Optional[Span] = None
        self._xray_available = self._check_xray()

    @contextmanager
    def span(self, name: str, attributes: Optional[dict] = None):
        """Create a new span as a context manager.

        Args:
            name: Span name (e.g., "model.invoke", "tool.search")
            attributes: Initial span attributes.

        Yields:
            Span object for adding attributes and events.
        """
        if not self.enabled:
            yield _NoOpSpan()
            return

        parent_id = self._active_span.span_id if self._active_span else None

        span = Span(
            name=f"{self.service_name}.{name}",
            trace_id=self._trace_id,
            parent_id=parent_id,
        )

        if attributes:
            span.attributes.update(attributes)

        previous_span = self._active_span
        self._active_span = span

        try:
            yield span
            span.end("ok")
        except Exception as e:
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            span.end("error")
            raise
        finally:
            self._spans.append(span)
            self._active_span = previous_span

            # Emit span
            self._emit_span(span)

    def get_trace_summary(self) -> dict:
        """Get a summary of all spans in this trace."""
        return {
            "trace_id": self._trace_id,
            "service": self.service_name,
            "span_count": len(self._spans),
            "total_duration_ms": sum(s.duration_ms for s in self._spans),
            "errors": sum(1 for s in self._spans if s.status == "error"),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "status": s.status,
                    "attributes": s.attributes,
                }
                for s in self._spans
            ],
        }

    def _emit_span(self, span: Span) -> None:
        """Emit span to X-Ray or structured log."""
        if self._xray_available:
            self._emit_to_xray(span)
        else:
            # Structured log fallback
            logger.debug(json.dumps({
                "trace": "span_complete",
                "name": span.name,
                "duration_ms": round(span.duration_ms, 2),
                "status": span.status,
                "attributes": span.attributes,
            }))

    def _emit_to_xray(self, span: Span) -> None:
        """Send span to X-Ray as a subsegment."""
        try:
            from aws_xray_sdk.core import xray_recorder
            subsegment = xray_recorder.begin_subsegment(span.name)
            for key, value in span.attributes.items():
                if isinstance(value, (str, int, float, bool)):
                    subsegment.put_annotation(key, value)
            subsegment.put_metadata("lambdallm", span.attributes)
            if span.status == "error":
                subsegment.add_fault_flag()
            xray_recorder.end_subsegment()
        except Exception as e:
            logger.debug(f"X-Ray emission failed (non-critical): {e}")

    def _check_xray(self) -> bool:
        """Check if X-Ray SDK is available."""
        try:
            import aws_xray_sdk
            return True
        except ImportError:
            return False

    def _generate_trace_id(self) -> str:
        """Generate an X-Ray compatible trace ID."""
        import time as time_mod
        hex_time = hex(int(time_mod.time()))[2:]
        unique = uuid.uuid4().hex[:24]
        return f"1-{hex_time}-{unique}"


class _NoOpSpan:
    """No-op span when tracing is disabled."""

    def set_attribute(self, key, value):
        pass

    def add_event(self, name, attributes=None):
        pass

    def end(self, status="ok"):
        pass
