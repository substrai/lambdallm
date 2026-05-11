"""Mock utilities for testing LambdaLLM handlers.

Provides mock providers and contexts so tests run without
AWS credentials or real LLM API calls.
"""

import functools
from typing import Any, Optional
from unittest.mock import patch, MagicMock

from lambdallm.core.models import ModelResponse
from lambdallm.providers.base import BaseProvider


class MockLambdaContext:
    """Mock AWS Lambda context for testing."""

    function_name = "test-function"
    function_version = "$LATEST"
    memory_limit_in_mb = 256
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"
    log_group_name = "/aws/lambda/test"
    log_stream_name = "test-stream"

    def __init__(self, timeout_ms: int = 30000):
        self._timeout_ms = timeout_ms

    def get_remaining_time_in_millis(self):
        return self._timeout_ms


class MockProvider(BaseProvider):
    """Mock model provider for testing.

    Returns predefined responses without making API calls.

    Example:
        provider = MockProvider(responses=["Hello!", "World!"])
        response = provider.invoke("test prompt", config)
        assert response.content == "Hello!"
        response = provider.invoke("another prompt", config)
        assert response.content == "World!"
    """

    def __init__(self, responses: Optional[list[str]] = None, raises: Optional[Exception] = None):
        self.responses = responses or ["Mock response"]
        self.raises = raises
        self._call_count = 0
        self._calls: list[dict] = []

    def invoke(self, prompt: str, config: Any) -> ModelResponse:
        if self.raises:
            raise self.raises

        self._calls.append({"prompt": prompt, "config": config})
        response_text = self.responses[self._call_count % len(self.responses)]
        self._call_count += 1

        return ModelResponse(
            content=response_text,
            model_id="mock-model",
            tokens_in=len(prompt) // 4,
            tokens_out=len(response_text) // 4,
            latency_ms=50.0,
            cost_usd=0.0001,
        )

    def supports_streaming(self) -> bool:
        return False

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def last_prompt(self) -> Optional[str]:
        return self._calls[-1]["prompt"] if self._calls else None


def mock_model(responses: Optional[list[str]] = None, raises: Optional[Exception] = None):
    """Decorator to mock the model provider in tests.

    Example:
        @mock_model(responses=["Test response"])
        def test_my_handler(self):
            result = my_handler(event, context)
            assert result["statusCode"] == 200
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            provider = MockProvider(responses=responses, raises=raises)
            with patch("lambdallm.providers.get_provider", return_value=provider):
                return func(*args, **kwargs)
        return wrapper

    return decorator
