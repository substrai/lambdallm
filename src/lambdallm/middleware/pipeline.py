"""Request/Response middleware pipeline for LambdaLLM.

Provides a pluggable pre/post processing pipeline with ordered execution,
error handling, and context passing between middleware stages.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MiddlewareStage(str, Enum):
    """Stages in the middleware pipeline."""

    PRE_REQUEST = "pre_request"
    POST_RESPONSE = "post_response"


@dataclass
class PipelineContext:
    """Shared context passed through the middleware chain.

    Allows middleware to share state, metadata, and timing information.
    """

    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, float] = field(default_factory=dict)
    errors: List[Dict[str, Any]] = field(default_factory=list)
    _aborted: bool = False
    abort_reason: Optional[str] = None

    def set(self, key: str, value: Any) -> None:
        """Set a context value."""
        self.metadata[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Get a context value."""
        return self.metadata.get(key, default)

    def abort(self, reason: str) -> None:
        """Abort the pipeline execution."""
        self._aborted = True
        self.abort_reason = reason

    @property
    def is_aborted(self) -> bool:
        """Check if pipeline has been aborted."""
        return self._aborted

    def record_error(self, middleware_name: str, error: Exception) -> None:
        """Record an error from a middleware."""
        self.errors.append(
            {
                "middleware": middleware_name,
                "error_type": type(error).__name__,
                "message": str(error),
                "timestamp": time.time(),
            }
        )


@dataclass
class Request:
    """Represents an LLM request flowing through the pipeline."""

    prompt: str
    model: str = "default"
    parameters: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Response:
    """Represents an LLM response flowing through the pipeline."""

    content: str
    model: str = "default"
    usage: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


class Middleware(ABC):
    """Base class for middleware components.

    Each middleware can process requests before they are sent and
    responses after they are received.
    """

    def __init__(self, name: Optional[str] = None, priority: int = 100):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.enabled = True

    @abstractmethod
    def process_request(self, request: Request, context: PipelineContext) -> Request:
        """Process a request before it is sent to the LLM."""
        ...

    @abstractmethod
    def process_response(
        self, response: Response, context: PipelineContext
    ) -> Response:
        """Process a response after it is received from the LLM."""
        ...

    def on_error(self, error: Exception, context: PipelineContext) -> None:
        """Handle errors during pipeline execution."""
        context.record_error(self.name, error)


class LoggingMiddleware(Middleware):
    """Middleware that logs requests and responses."""

    def __init__(self, log_level: int = logging.INFO, priority: int = 10):
        super().__init__(name="LoggingMiddleware", priority=priority)
        self.log_level = log_level
        self.logs: List[Dict[str, Any]] = []

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        entry = {
            "stage": "request",
            "prompt_length": len(request.prompt),
            "model": request.model,
            "timestamp": time.time(),
        }
        self.logs.append(entry)
        logger.log(self.log_level, f"Request to {request.model}: {len(request.prompt)} chars")
        context.set("request_logged", True)
        return request

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        entry = {
            "stage": "response",
            "content_length": len(response.content),
            "model": response.model,
            "latency_ms": response.latency_ms,
            "timestamp": time.time(),
        }
        self.logs.append(entry)
        logger.log(self.log_level, f"Response from {response.model}: {len(response.content)} chars")
        context.set("response_logged", True)
        return response


class ValidationMiddleware(Middleware):
    """Middleware that validates requests and responses."""

    def __init__(
        self,
        max_prompt_length: int = 100000,
        max_response_length: int = 50000,
        required_fields: Optional[List[str]] = None,
        priority: int = 20,
    ):
        super().__init__(name="ValidationMiddleware", priority=priority)
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.required_fields = required_fields or []

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        if not request.prompt.strip():
            raise ValueError("Request prompt cannot be empty")
        if len(request.prompt) > self.max_prompt_length:
            raise ValueError(
                f"Prompt exceeds maximum length: {len(request.prompt)} > {self.max_prompt_length}"
            )
        for field_name in self.required_fields:
            if field_name not in request.parameters:
                raise ValueError(f"Required field missing: {field_name}")
        context.set("request_validated", True)
        return request

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        if len(response.content) > self.max_response_length:
            raise ValueError(
                f"Response exceeds maximum length: {len(response.content)} > {self.max_response_length}"
            )
        context.set("response_validated", True)
        return response


class TransformationMiddleware(Middleware):
    """Middleware that transforms requests and responses."""

    def __init__(
        self,
        request_transforms: Optional[List[Callable[[Request], Request]]] = None,
        response_transforms: Optional[List[Callable[[Response], Response]]] = None,
        priority: int = 50,
    ):
        super().__init__(name="TransformationMiddleware", priority=priority)
        self.request_transforms = request_transforms or []
        self.response_transforms = response_transforms or []

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        for transform in self.request_transforms:
            request = transform(request)
        context.set("request_transformed", True)
        return request

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        for transform in self.response_transforms:
            response = transform(response)
        context.set("response_transformed", True)
        return response


class MiddlewarePipeline:
    """Manages an ordered chain of middleware with error handling.

    Middleware are executed in priority order (lowest first) for requests,
    and in reverse priority order for responses.
    """

    def __init__(self, error_strategy: str = "continue"):
        """Initialize the pipeline.

        Args:
            error_strategy: How to handle errors - 'continue', 'abort', or 'raise'.
        """
        self._middlewares: List[Middleware] = []
        self.error_strategy = error_strategy
        self._sorted = False

    def add(self, middleware: Middleware) -> "MiddlewarePipeline":
        """Add a middleware to the pipeline."""
        self._middlewares.append(middleware)
        self._sorted = False
        return self

    def remove(self, name: str) -> bool:
        """Remove a middleware by name."""
        initial_count = len(self._middlewares)
        self._middlewares = [m for m in self._middlewares if m.name != name]
        return len(self._middlewares) < initial_count

    def get(self, name: str) -> Optional[Middleware]:
        """Get a middleware by name."""
        for m in self._middlewares:
            if m.name == name:
                return m
        return None

    @property
    def middlewares(self) -> List[Middleware]:
        """Get sorted list of active middlewares."""
        if not self._sorted:
            self._middlewares.sort(key=lambda m: m.priority)
            self._sorted = True
        return [m for m in self._middlewares if m.enabled]

    def process_request(
        self, request: Request, context: Optional[PipelineContext] = None
    ) -> Tuple[Request, PipelineContext]:
        """Process a request through all middleware in priority order."""
        if context is None:
            context = PipelineContext()

        context.timestamps["request_start"] = time.time()

        for middleware in self.middlewares:
            if context.is_aborted:
                break
            try:
                request = middleware.process_request(request, context)
            except Exception as e:
                middleware.on_error(e, context)
                if self.error_strategy == "raise":
                    raise
                elif self.error_strategy == "abort":
                    context.abort(f"Error in {middleware.name}: {e}")
                    break
                # 'continue' - just log and move on

        context.timestamps["request_end"] = time.time()
        return request, context

    def process_response(
        self, response: Response, context: Optional[PipelineContext] = None
    ) -> Tuple[Response, PipelineContext]:
        """Process a response through all middleware in reverse priority order."""
        if context is None:
            context = PipelineContext()

        context.timestamps["response_start"] = time.time()

        # Process in reverse order for responses
        for middleware in reversed(self.middlewares):
            if context.is_aborted:
                break
            try:
                response = middleware.process_response(response, context)
            except Exception as e:
                middleware.on_error(e, context)
                if self.error_strategy == "raise":
                    raise
                elif self.error_strategy == "abort":
                    context.abort(f"Error in {middleware.name}: {e}")
                    break

        context.timestamps["response_end"] = time.time()
        return response, context

    def execute(
        self,
        request: Request,
        handler: Callable[[Request], Response],
        context: Optional[PipelineContext] = None,
    ) -> Tuple[Response, PipelineContext]:
        """Execute the full pipeline: pre-process request, call handler, post-process response."""
        request, context = self.process_request(request, context)

        if context.is_aborted:
            return Response(content="", metadata={"aborted": True, "reason": context.abort_reason}), context

        context.timestamps["handler_start"] = time.time()
        response = handler(request)
        context.timestamps["handler_end"] = time.time()

        response, context = self.process_response(response, context)
        return response, context

    def clear(self) -> None:
        """Remove all middleware from the pipeline."""
        self._middlewares.clear()
        self._sorted = False

    def __len__(self) -> int:
        return len(self._middlewares)
