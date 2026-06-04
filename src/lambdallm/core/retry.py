"""Retry strategy with exponential backoff and jitter.

Provides configurable retry logic for LLM API calls, handling throttling
and transient errors with per-model retry budgets.

Features:
- Exponential backoff with full jitter (decorrelated jitter option)
- Per-model retry budgets to prevent cascade failures
- Configurable exception lists for retryable errors
- Async and sync support
- Callback hooks for retry events
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)


class RetryBudgetExhausted(Exception):
    """Raised when a model's retry budget is exhausted."""

    def __init__(self, model: str, budget: int, window_seconds: float):
        self.model = model
        self.budget = budget
        self.window_seconds = window_seconds
        super().__init__(
            f"Retry budget exhausted for model '{model}': "
            f"{budget} retries in {window_seconds}s window"
        )


class MaxRetriesExceeded(Exception):
    """Raised when maximum retries are exceeded."""

    def __init__(self, attempts: int, last_exception: Exception):
        self.attempts = attempts
        self.last_exception = last_exception
        super().__init__(
            f"Max retries ({attempts}) exceeded. Last error: {last_exception}"
        )


@dataclass
class RetryConfig:
    """Configuration for retry behavior.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds before first retry.
        max_delay: Maximum delay cap in seconds.
        exponential_base: Multiplier for exponential backoff.
        jitter: Jitter strategy — "full", "equal", "decorrelated", or "none".
        retryable_exceptions: Tuple of exception types that trigger retries.
        non_retryable_exceptions: Exceptions that should never be retried.
        retry_on: Optional callable that receives the exception and returns bool.
        on_retry: Optional callback invoked on each retry (attempt, delay, exc).
        budget_max_retries: Max retries allowed per model in the budget window.
        budget_window_seconds: Time window for budget tracking.
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    jitter: str = "full"
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,)
    non_retryable_exceptions: Tuple[Type[Exception], ...] = ()
    retry_on: Optional[Callable[[Exception], bool]] = None
    on_retry: Optional[Callable[[int, float, Exception], None]] = None
    budget_max_retries: Optional[int] = None
    budget_window_seconds: float = 60.0


@dataclass
class RetryBudget:
    """Tracks retry budget per model to prevent cascade failures.

    Maintains a sliding window of retry timestamps and enforces
    a maximum number of retries within that window.
    """

    max_retries: int = 50
    window_seconds: float = 60.0
    _model_retries: Dict[str, list] = field(default_factory=dict)

    def record_retry(self, model: str) -> None:
        """Record a retry attempt for a model."""
        if model not in self._model_retries:
            self._model_retries[model] = []
        self._model_retries[model].append(time.time())

    def is_budget_available(self, model: str) -> bool:
        """Check if the model has remaining retry budget."""
        self._cleanup_expired(model)
        retries = self._model_retries.get(model, [])
        return len(retries) < self.max_retries

    def get_usage(self, model: str) -> Tuple[int, int]:
        """Get current usage (used, max) for a model."""
        self._cleanup_expired(model)
        used = len(self._model_retries.get(model, []))
        return used, self.max_retries

    def reset(self, model: Optional[str] = None) -> None:
        """Reset budget for a model or all models."""
        if model:
            self._model_retries.pop(model, None)
        else:
            self._model_retries.clear()

    def _cleanup_expired(self, model: str) -> None:
        """Remove retry records outside the current window."""
        if model not in self._model_retries:
            return
        cutoff = time.time() - self.window_seconds
        self._model_retries[model] = [
            t for t in self._model_retries[model] if t > cutoff
        ]


class RetryStrategy:
    """Retry strategy with exponential backoff and jitter.

    Provides configurable retry logic for handling throttling and transient
    errors when calling LLM APIs. Supports per-model retry budgets.

    Example:
        strategy = RetryStrategy(RetryConfig(max_retries=5, base_delay=1.0))

        @strategy.wrap(model="gpt-4")
        def call_llm(prompt):
            return api.complete(prompt)

        # Or use directly:
        result = strategy.execute(call_llm, args=(prompt,), model="gpt-4")
    """

    def __init__(
        self,
        config: Optional[RetryConfig] = None,
        budget: Optional[RetryBudget] = None,
    ):
        self.config = config or RetryConfig()
        self.budget = budget
        self._attempt_history: list = []

    def calculate_delay(self, attempt: int, last_delay: float = 0.0) -> float:
        """Calculate the delay for a given retry attempt.

        Args:
            attempt: The current attempt number (0-indexed).
            last_delay: The delay from the previous attempt (for decorrelated).

        Returns:
            Delay in seconds with jitter applied.
        """
        # Calculate base exponential delay
        exp_delay = self.config.base_delay * (
            self.config.exponential_base ** attempt
        )
        exp_delay = min(exp_delay, self.config.max_delay)

        # Apply jitter strategy
        if self.config.jitter == "full":
            # Full jitter: uniform random between 0 and exp_delay
            return random.uniform(0, exp_delay)
        elif self.config.jitter == "equal":
            # Equal jitter: half fixed, half random
            half = exp_delay / 2
            return half + random.uniform(0, half)
        elif self.config.jitter == "decorrelated":
            # Decorrelated jitter: based on last delay
            base = self.config.base_delay
            return random.uniform(base, max(base, last_delay * 3))
        else:
            # No jitter
            return exp_delay

    def should_retry(self, exception: Exception) -> bool:
        """Determine if an exception should trigger a retry.

        Args:
            exception: The exception that was raised.

        Returns:
            True if the operation should be retried.
        """
        # Check non-retryable first
        if isinstance(exception, self.config.non_retryable_exceptions):
            return False

        # Check custom retry_on callable
        if self.config.retry_on is not None:
            return self.config.retry_on(exception)

        # Check retryable exceptions
        return isinstance(exception, self.config.retryable_exceptions)

    def execute(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Execute a function with retry logic.

        Args:
            func: The callable to execute.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.
            model: Optional model name for budget tracking.

        Returns:
            The result of the successful function call.

        Raises:
            MaxRetriesExceeded: If all retry attempts fail.
            RetryBudgetExhausted: If the model's retry budget is exhausted.
        """
        kwargs = kwargs or {}
        last_delay = 0.0
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as exc:
                last_exception = exc

                # Check if we should retry this exception
                if not self.should_retry(exc):
                    raise

                # Check if we've exhausted retries
                if attempt >= self.config.max_retries:
                    raise MaxRetriesExceeded(
                        attempts=attempt + 1, last_exception=exc
                    )

                # Check budget
                if model and self.budget:
                    if not self.budget.is_budget_available(model):
                        raise RetryBudgetExhausted(
                            model=model,
                            budget=self.budget.max_retries,
                            window_seconds=self.budget.window_seconds,
                        )
                    self.budget.record_retry(model)

                # Calculate delay
                delay = self.calculate_delay(attempt, last_delay)
                last_delay = delay

                # Record attempt
                self._attempt_history.append(
                    {
                        "attempt": attempt + 1,
                        "delay": delay,
                        "exception": str(exc),
                        "model": model,
                    }
                )

                # Invoke on_retry callback
                if self.config.on_retry:
                    self.config.on_retry(attempt + 1, delay, exc)

                # Sleep before retry
                time.sleep(delay)

        # Should not reach here, but safety fallback
        raise MaxRetriesExceeded(
            attempts=self.config.max_retries + 1,
            last_exception=last_exception or RuntimeError("Unknown error"),
        )

    async def execute_async(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
    ) -> Any:
        """Execute an async function with retry logic.

        Args:
            func: The async callable to execute.
            args: Positional arguments for the callable.
            kwargs: Keyword arguments for the callable.
            model: Optional model name for budget tracking.

        Returns:
            The result of the successful function call.

        Raises:
            MaxRetriesExceeded: If all retry attempts fail.
            RetryBudgetExhausted: If the model's retry budget is exhausted.
        """
        kwargs = kwargs or {}
        last_delay = 0.0
        last_exception: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as exc:
                last_exception = exc

                if not self.should_retry(exc):
                    raise

                if attempt >= self.config.max_retries:
                    raise MaxRetriesExceeded(
                        attempts=attempt + 1, last_exception=exc
                    )

                if model and self.budget:
                    if not self.budget.is_budget_available(model):
                        raise RetryBudgetExhausted(
                            model=model,
                            budget=self.budget.max_retries,
                            window_seconds=self.budget.window_seconds,
                        )
                    self.budget.record_retry(model)

                delay = self.calculate_delay(attempt, last_delay)
                last_delay = delay

                self._attempt_history.append(
                    {
                        "attempt": attempt + 1,
                        "delay": delay,
                        "exception": str(exc),
                        "model": model,
                    }
                )

                if self.config.on_retry:
                    self.config.on_retry(attempt + 1, delay, exc)

                await asyncio.sleep(delay)

        raise MaxRetriesExceeded(
            attempts=self.config.max_retries + 1,
            last_exception=last_exception or RuntimeError("Unknown error"),
        )

    def wrap(
        self, model: Optional[str] = None
    ) -> Callable:
        """Decorator to wrap a function with retry logic.

        Args:
            model: Optional model name for budget tracking.

        Returns:
            Decorator that adds retry behavior.

        Example:
            @strategy.wrap(model="gpt-4")
            def call_api(prompt):
                return client.complete(prompt)
        """
        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return self.execute(func, args=args, kwargs=kwargs, model=model)
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        return decorator

    def wrap_async(
        self, model: Optional[str] = None
    ) -> Callable:
        """Decorator to wrap an async function with retry logic.

        Args:
            model: Optional model name for budget tracking.

        Returns:
            Decorator that adds retry behavior to async functions.
        """
        def decorator(func: Callable) -> Callable:
            async def wrapper(*args, **kwargs):
                return await self.execute_async(
                    func, args=args, kwargs=kwargs, model=model
                )
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
            return wrapper
        return decorator

    @property
    def attempt_history(self) -> list:
        """Get the history of retry attempts."""
        return list(self._attempt_history)

    def reset_history(self) -> None:
        """Clear the attempt history."""
        self._attempt_history.clear()


def create_retry_strategy(
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: str = "full",
    retryable_exceptions: Optional[Tuple[Type[Exception], ...]] = None,
    budget_max_retries: Optional[int] = None,
    budget_window_seconds: float = 60.0,
) -> RetryStrategy:
    """Factory to create a configured RetryStrategy.

    Args:
        max_retries: Maximum number of retry attempts.
        base_delay: Initial delay in seconds.
        max_delay: Maximum delay cap in seconds.
        jitter: Jitter strategy — "full", "equal", "decorrelated", or "none".
        retryable_exceptions: Exceptions that trigger retries.
        budget_max_retries: Max retries per model in budget window.
        budget_window_seconds: Budget tracking window.

    Returns:
        Configured RetryStrategy instance.
    """
    config = RetryConfig(
        max_retries=max_retries,
        base_delay=base_delay,
        max_delay=max_delay,
        jitter=jitter,
        retryable_exceptions=retryable_exceptions or (Exception,),
        budget_max_retries=budget_max_retries,
        budget_window_seconds=budget_window_seconds,
    )

    budget = None
    if budget_max_retries is not None:
        budget = RetryBudget(
            max_retries=budget_max_retries,
            window_seconds=budget_window_seconds,
        )

    return RetryStrategy(config=config, budget=budget)
