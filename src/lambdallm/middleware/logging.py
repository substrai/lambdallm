"""Logging middleware for LambdaLLM.

Provides structured JSON logging for every request/response cycle.
Enabled by default (Observable by Default principle).
"""

import json
import logging
import time
from typing import Any

from lambdallm.middleware.base import Middleware

logger = logging.getLogger("lambdallm")


class LoggingMiddleware(Middleware):
    """Structured logging middleware.

    Logs request metadata, response status, latency, and cost
    in JSON format for easy CloudWatch Insights querying.

    Config:
        log_prompts: Whether to log prompt content (default: False for PII safety)
        log_level: Logging level (default: INFO)
    """

    def __init__(self, log_prompts: bool = False, log_level: int = logging.INFO):
        self.log_prompts = log_prompts
        self.log_level = log_level
        self._start_time: float = 0

    def before_invoke(self, event: dict, context: Any) -> dict:
        self._start_time = time.time()

        log_data = {
            "event": "request.start",
            "path": event.get("path", event.get("rawPath", "/")),
            "method": event.get("httpMethod", event.get("requestContext", {}).get("http", {}).get("method", "INVOKE")),
            "request_id": event.get("requestContext", {}).get("requestId", "local"),
        }

        if self.log_prompts:
            body = event.get("body", "")
            if isinstance(body, str) and body:
                try:
                    log_data["body_preview"] = json.loads(body)
                except json.JSONDecodeError:
                    log_data["body_preview"] = body[:200]

        logger.log(self.log_level, json.dumps(log_data))
        return event

    def after_invoke(self, event: dict, result: Any, context: Any) -> Any:
        elapsed_ms = (time.time() - self._start_time) * 1000

        log_data = {
            "event": "request.complete",
            "latency_ms": round(elapsed_ms, 2),
            "status_code": result.get("statusCode", 200) if isinstance(result, dict) else 200,
            "model_invocations": context.metrics.get("model.invocations") or 0,
            "total_cost_usd": round(context.metrics.get("model.total_cost_usd") or 0, 6),
            "tokens_in": context.metrics.get("model.tokens_in") or 0,
            "tokens_out": context.metrics.get("model.tokens_out") or 0,
        }

        logger.log(self.log_level, json.dumps(log_data))
        return result

    def on_error(self, event: dict, error: Exception, context: Any) -> None:
        elapsed_ms = (time.time() - self._start_time) * 1000

        log_data = {
            "event": "request.error",
            "latency_ms": round(elapsed_ms, 2),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }

        logger.error(json.dumps(log_data))
