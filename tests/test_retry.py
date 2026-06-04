"""Tests for retry strategy with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock, patch

import pytest

from lambdallm.core.retry import (
    MaxRetriesExceeded,
    RetryBudget,
    RetryBudgetExhausted,
    RetryConfig,
    RetryStrategy,
    create_retry_strategy,
)


class TestRetryConfig:
    """Tests for RetryConfig defaults and customization."""

    def test_default_config(self):
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 60.0
        assert config.exponential_base == 2.0
        assert config.jitter == "full"

    def test_custom_config(self):
        config = RetryConfig(
            max_retries=5,
            base_delay=0.5,
            max_delay=30.0,
            jitter="equal",
        )
        assert config.max_retries == 5
        assert config.base_delay == 0.5
        assert config.max_delay == 30.0
        assert config.jitter == "equal"


class TestRetryStrategy:
    """Tests for RetryStrategy execution and backoff."""

    def test_successful_execution_no_retry(self):
        """Test that successful calls don't trigger retries."""
        strategy = RetryStrategy(RetryConfig(max_retries=3))
        func = MagicMock(return_value="success")

        result = strategy.execute(func)

        assert result == "success"
        func.assert_called_once()

    @patch("lambdallm.core.retry.time.sleep")
    def test_retries_on_failure_then_succeeds(self, mock_sleep):
        """Test retry on transient failures with eventual success."""
        call_count = 0

        def flaky_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Connection refused")
            return "recovered"

        strategy = RetryStrategy(RetryConfig(max_retries=5, base_delay=0.1))
        result = strategy.execute(flaky_func)

        assert result == "recovered"
        assert call_count == 3

    @patch("lambdallm.core.retry.time.sleep")
    def test_max_retries_exceeded(self, mock_sleep):
        """Test that MaxRetriesExceeded is raised after exhausting retries."""
        func = MagicMock(side_effect=RuntimeError("always fails"))
        strategy = RetryStrategy(RetryConfig(max_retries=3, base_delay=0.01))

        with pytest.raises(MaxRetriesExceeded) as exc_info:
            strategy.execute(func)

        assert exc_info.value.attempts == 4  # initial + 3 retries
        assert func.call_count == 4

    @patch("lambdallm.core.retry.time.sleep")
    def test_non_retryable_exception_raises_immediately(self, mock_sleep):
        """Test that non-retryable exceptions are not retried."""
        func = MagicMock(side_effect=ValueError("bad input"))
        config = RetryConfig(
            max_retries=5,
            retryable_exceptions=(ConnectionError,),
            non_retryable_exceptions=(ValueError,),
        )
        strategy = RetryStrategy(config)

        with pytest.raises(ValueError, match="bad input"):
            strategy.execute(func)

        func.assert_called_once()

    def test_calculate_delay_full_jitter(self):
        """Test full jitter produces delays within expected range."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter="full")
        strategy = RetryStrategy(config)

        delays = [strategy.calculate_delay(2) for _ in range(100)]
        # attempt 2: base * 2^2 = 4.0, full jitter: [0, 4.0]
        assert all(0 <= d <= 4.0 for d in delays)

    def test_calculate_delay_equal_jitter(self):
        """Test equal jitter produces delays in upper half range."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter="equal")
        strategy = RetryStrategy(config)

        delays = [strategy.calculate_delay(1) for _ in range(100)]
        # attempt 1: base * 2^1 = 2.0, equal jitter: [1.0, 2.0]
        assert all(1.0 <= d <= 2.0 for d in delays)

    def test_calculate_delay_no_jitter(self):
        """Test no jitter produces deterministic delays."""
        config = RetryConfig(base_delay=1.0, exponential_base=2.0, jitter="none")
        strategy = RetryStrategy(config)

        delay = strategy.calculate_delay(3)
        # base * 2^3 = 8.0
        assert delay == 8.0

    def test_calculate_delay_respects_max_delay(self):
        """Test that delay is capped at max_delay."""
        config = RetryConfig(
            base_delay=1.0, exponential_base=2.0, max_delay=5.0, jitter="none"
        )
        strategy = RetryStrategy(config)

        delay = strategy.calculate_delay(10)
        assert delay == 5.0

    @patch("lambdallm.core.retry.time.sleep")
    def test_on_retry_callback(self, mock_sleep):
        """Test that on_retry callback is invoked on each retry."""
        callback = MagicMock()
        call_count = 0

        def failing_then_ok():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise IOError("io error")
            return "ok"

        config = RetryConfig(max_retries=5, base_delay=0.01, on_retry=callback)
        strategy = RetryStrategy(config)
        strategy.execute(failing_then_ok)

        assert callback.call_count == 2
        # Verify callback args: (attempt_number, delay, exception)
        first_call = callback.call_args_list[0]
        assert first_call[0][0] == 1  # first retry attempt

    @patch("lambdallm.core.retry.time.sleep")
    def test_custom_retry_on_callable(self, mock_sleep):
        """Test custom retry_on function for selective retries."""
        config = RetryConfig(
            max_retries=5,
            base_delay=0.01,
            retry_on=lambda exc: "throttled" in str(exc),
        )
        strategy = RetryStrategy(config)

        # Should not retry non-matching exception
        func = MagicMock(side_effect=RuntimeError("bad request"))
        with pytest.raises(RuntimeError, match="bad request"):
            strategy.execute(func)
        func.assert_called_once()

        # Should retry matching exception
        call_count = 0

        def throttled_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("throttled by API")
            return "ok"

        result = strategy.execute(throttled_func)
        assert result == "ok"


class TestRetryBudget:
    """Tests for per-model retry budget tracking."""

    def test_budget_allows_retries_within_limit(self):
        """Test budget permits retries under the limit."""
        budget = RetryBudget(max_retries=5, window_seconds=60.0)

        for _ in range(4):
            assert budget.is_budget_available("gpt-4")
            budget.record_retry("gpt-4")

        assert budget.is_budget_available("gpt-4")

    def test_budget_exhausted(self):
        """Test budget blocks retries when exhausted."""
        budget = RetryBudget(max_retries=3, window_seconds=60.0)

        for _ in range(3):
            budget.record_retry("gpt-4")

        assert not budget.is_budget_available("gpt-4")

    def test_budget_per_model_isolation(self):
        """Test that budgets are tracked independently per model."""
        budget = RetryBudget(max_retries=2, window_seconds=60.0)

        budget.record_retry("gpt-4")
        budget.record_retry("gpt-4")

        assert not budget.is_budget_available("gpt-4")
        assert budget.is_budget_available("claude-3")

    @patch("lambdallm.core.retry.time.time")
    def test_budget_window_expiry(self, mock_time):
        """Test that old retries expire outside the window."""
        budget = RetryBudget(max_retries=2, window_seconds=10.0)

        # Record retries at time 100
        mock_time.return_value = 100.0
        budget.record_retry("gpt-4")
        budget.record_retry("gpt-4")
        assert not budget.is_budget_available("gpt-4")

        # Advance time past the window
        mock_time.return_value = 111.0
        assert budget.is_budget_available("gpt-4")

    def test_budget_reset(self):
        """Test budget reset clears retry history."""
        budget = RetryBudget(max_retries=2, window_seconds=60.0)
        budget.record_retry("gpt-4")
        budget.record_retry("gpt-4")

        budget.reset("gpt-4")
        assert budget.is_budget_available("gpt-4")

    @patch("lambdallm.core.retry.time.sleep")
    def test_strategy_with_budget_exhaustion(self, mock_sleep):
        """Test that strategy raises RetryBudgetExhausted when budget is gone."""
        budget = RetryBudget(max_retries=2, window_seconds=60.0)
        config = RetryConfig(max_retries=10, base_delay=0.01)
        strategy = RetryStrategy(config=config, budget=budget)

        func = MagicMock(side_effect=RuntimeError("fail"))

        with pytest.raises(RetryBudgetExhausted) as exc_info:
            strategy.execute(func, model="gpt-4")

        assert exc_info.value.model == "gpt-4"

    def test_get_usage(self):
        """Test getting current budget usage for a model."""
        budget = RetryBudget(max_retries=10, window_seconds=60.0)
        budget.record_retry("gpt-4")
        budget.record_retry("gpt-4")
        budget.record_retry("gpt-4")

        used, max_r = budget.get_usage("gpt-4")
        assert used == 3
        assert max_r == 10


class TestCreateRetryStrategy:
    """Tests for the factory function."""

    def test_create_with_defaults(self):
        """Test factory with default arguments."""
        strategy = create_retry_strategy()
        assert strategy.config.max_retries == 3
        assert strategy.budget is None

    def test_create_with_budget(self):
        """Test factory creates budget when budget_max_retries is set."""
        strategy = create_retry_strategy(
            max_retries=5,
            budget_max_retries=20,
            budget_window_seconds=120.0,
        )
        assert strategy.budget is not None
        assert strategy.budget.max_retries == 20
        assert strategy.budget.window_seconds == 120.0

    @patch("lambdallm.core.retry.time.sleep")
    def test_wrap_decorator(self, mock_sleep):
        """Test the wrap decorator applies retry logic."""
        strategy = create_retry_strategy(max_retries=3, base_delay=0.01)
        call_count = 0

        @strategy.wrap(model="gpt-4")
        def flaky_api():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("timeout")
            return "result"

        result = flaky_api()
        assert result == "result"
        assert call_count == 2


class TestAsyncRetry:
    """Tests for async retry execution."""

    @patch("lambdallm.core.retry.asyncio.sleep")
    def test_async_execute_with_retries(self, mock_sleep):
        """Test async retry with eventual success."""
        mock_sleep.return_value = asyncio.Future()
        mock_sleep.return_value.set_result(None)

        call_count = 0

        async def flaky_async():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("timeout")
            return "async_result"

        strategy = RetryStrategy(RetryConfig(max_retries=5, base_delay=0.01))
        result = asyncio.run(strategy.execute_async(flaky_async))

        assert result == "async_result"
        assert call_count == 3
