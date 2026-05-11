"""The @handler decorator - the heart of LambdaLLM.

This decorator wraps AWS Lambda handler functions with LLM orchestration,
providing automatic model selection, error handling, cost tracking,
timeout management, and response formatting.

The framework controls the lifecycle (Inversion of Control principle):
- User writes business logic
- Framework handles everything else
"""

import functools
import json
import logging
import time
from typing import Any, Callable, Optional, Union

from lambdallm.core.models import Model, ModelConfig, ModelResponse
from lambdallm.core.context import LambdaLLMContext
from lambdallm.core.exceptions import (
    LambdaLLMError,
    ModelInvocationError,
    TimeoutError,
    BudgetExceededError,
)

logger = logging.getLogger("lambdallm")


def handler(
    model: Union[Model, str, None] = None,
    timeout_strategy: str = "fail-fast",
    timeout_buffer: int = 5,
    max_retries: int = 3,
    fallback_model: Optional[Union[Model, str]] = None,
    middleware: Optional[list] = None,
    session: Optional[Any] = None,
    router: Optional[Any] = None,
):
    """Decorator that wraps a Lambda handler with LLM orchestration.

    Args:
        model: Default model to use. Can be a Model enum or model ID string.
        timeout_strategy: How to handle Lambda timeout. One of:
            - "fail-fast": Return error before timeout
            - "truncate": Return partial result
            - "checkpoint": Save progress, resume on next invocation
        timeout_buffer: Seconds to reserve before Lambda timeout for cleanup.
        max_retries: Number of retries on transient model errors.
        fallback_model: Model to use if primary model fails.
        middleware: List of middleware instances to apply.
        session: Session configuration for state management.
        router: Model router for cost-aware model selection.

    Example:
        @handler(model=Model.CLAUDE_3_HAIKU)
        def lambda_handler(event, context):
            result = context.invoke("Summarize: {text}", text=event["body"]["text"])
            return {"statusCode": 200, "body": result}
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(event: dict, lambda_context: Any) -> dict:
            start_time = time.time()

            # Build the LambdaLLM context that gets passed to the user function
            llm_context = LambdaLLMContext(
                model=_resolve_model(model),
                timeout_strategy=timeout_strategy,
                timeout_buffer=timeout_buffer,
                max_retries=max_retries,
                fallback_model=_resolve_model(fallback_model) if fallback_model else None,
                lambda_context=lambda_context,
                middleware=middleware or [],
                session_config=session,
                router=router,
            )

            try:
                # Execute middleware: before
                processed_event = _run_before_middleware(event, llm_context)

                # Call the user's handler function
                result = func(processed_event, llm_context)

                # Execute middleware: after
                result = _run_after_middleware(processed_event, result, llm_context)

                # Track metrics
                elapsed_ms = (time.time() - start_time) * 1000
                llm_context.metrics.record("handler.latency_ms", elapsed_ms)
                llm_context.metrics.record("handler.success", 1)

                return _format_response(result)

            except BudgetExceededError as e:
                logger.warning(f"Budget exceeded: {e}")
                return _error_response(429, str(e))

            except TimeoutError as e:
                logger.warning(f"Timeout approaching: {e}")
                if timeout_strategy == "checkpoint" and llm_context.checkpoint:
                    return _checkpoint_response(llm_context.checkpoint)
                return _error_response(408, str(e))

            except ModelInvocationError as e:
                logger.error(f"Model invocation failed: {e}")
                return _error_response(502, str(e))

            except LambdaLLMError as e:
                logger.error(f"LambdaLLM error: {e}")
                return _error_response(500, str(e))

            except Exception as e:
                logger.exception(f"Unexpected error: {e}")
                llm_context.metrics.record("handler.error", 1)
                return _error_response(500, "Internal server error")

        # Attach metadata for introspection
        wrapper._lambdallm_handler = True
        wrapper._lambdallm_config = {
            "model": model,
            "timeout_strategy": timeout_strategy,
            "max_retries": max_retries,
        }

        return wrapper

    return decorator


def _resolve_model(model: Union[Model, str, None]) -> Optional[ModelConfig]:
    """Resolve a model identifier to a ModelConfig."""
    if model is None:
        return ModelConfig.from_model(Model.CLAUDE_3_HAIKU)  # sensible default
    if isinstance(model, Model):
        return ModelConfig.from_model(model)
    if isinstance(model, str):
        return ModelConfig(model_id=model)
    return model


def _run_before_middleware(event: dict, context: LambdaLLMContext) -> dict:
    """Execute before_invoke on all middleware."""
    processed = event
    for mw in context.middleware:
        processed = mw.before_invoke(processed, context)
    return processed


def _run_after_middleware(event: dict, result: Any, context: LambdaLLMContext) -> Any:
    """Execute after_invoke on all middleware."""
    processed = result
    for mw in context.middleware:
        processed = mw.after_invoke(event, processed, context)
    return processed


def _format_response(result: Any) -> dict:
    """Format the handler result as a proper Lambda response."""
    if isinstance(result, dict) and "statusCode" in result:
        # Already formatted as Lambda response
        if isinstance(result.get("body"), (dict, list)):
            result["body"] = json.dumps(result["body"])
        return result

    # Auto-wrap in Lambda response format
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(result) if not isinstance(result, str) else result,
    }


def _error_response(status_code: int, message: str) -> dict:
    """Create a standardized error response."""
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"error": message}),
    }


def _checkpoint_response(checkpoint: dict) -> dict:
    """Create a checkpoint response for resumable operations."""
    return {
        "statusCode": 202,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({
            "status": "in_progress",
            "checkpoint": checkpoint,
            "message": "Operation checkpointed. Invoke again to resume.",
        }),
    }
