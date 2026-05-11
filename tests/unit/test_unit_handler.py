"""Unit Tests: Handler Decorator

Proves the @handler decorator works in isolation.
Shows how to mock the provider and test without AWS credentials.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from lambdallm import handler, Model
from lambdallm.core.models import ModelResponse
from lambdallm.testing import MockProvider, MockLambdaContext, mock_model


class TestHandlerDecorator:
    """Test @handler in isolation - no AWS needed."""

    @mock_model(responses=["This is a summary of the document."])
    def test_handler_returns_200_on_success(self):
        """Handler should return 200 with properly formatted response."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            result = context.invoke("Summarize: {text}", text="hello")
            return {"statusCode": 200, "body": {"result": result}}

        result = my_handler({"body": "{}"}, MockLambdaContext())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "result" in body

    @mock_model(responses=["Response text"])
    def test_handler_auto_formats_dict_response(self):
        """If handler returns a dict without statusCode, framework wraps it."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            return {"answer": "42"}

        result = my_handler({}, MockLambdaContext())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["answer"] == "42"

    @mock_model(responses=["ignored"])
    def test_handler_catches_user_exceptions(self):
        """Handler should catch exceptions and return 500."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            raise ValueError("Something broke")

        result = my_handler({}, MockLambdaContext())

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "error" in body

    @mock_model(responses=["response"])
    def test_handler_tracks_cost(self):
        """Handler should track cumulative cost across invocations."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            context.invoke("First call")
            context.invoke("Second call")
            return {"statusCode": 200, "body": {"cost": context.total_cost}}

        result = my_handler({}, MockLambdaContext())

        body = json.loads(result["body"])
        assert body["cost"] > 0

    @mock_model(responses=["response"])
    def test_handler_respects_timeout_buffer(self):
        """Handler should be aware of remaining Lambda time."""

        @handler(model=Model.CLAUDE_3_HAIKU, timeout_buffer=10)
        def my_handler(event, context):
            return {"statusCode": 200, "body": {"remaining_ms": context.remaining_time_ms}}

        # MockLambdaContext defaults to 30000ms
        result = my_handler({}, MockLambdaContext(timeout_ms=30000))

        body = json.loads(result["body"])
        assert body["remaining_ms"] == 30000

    def test_handler_metadata_attached(self):
        """Handler should have framework metadata for introspection."""

        @handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="checkpoint")
        def my_handler(event, context):
            pass

        assert my_handler._lambdallm_handler is True
        assert my_handler._lambdallm_config["timeout_strategy"] == "checkpoint"


class TestHandlerRetries:
    """Test retry behavior with mock providers."""

    def test_retries_on_transient_error(self):
        """Should retry on retryable errors."""
        from lambdallm.core.exceptions import ModelInvocationError

        call_count = {"n": 0}

        class FailThenSucceedProvider:
            def invoke(self, prompt, config):
                call_count["n"] += 1
                if call_count["n"] < 3:
                    raise ModelInvocationError("Throttled", retryable=True)
                return ModelResponse(
                    content="Success", model_id="test",
                    tokens_in=5, tokens_out=5, latency_ms=100, cost_usd=0.0001
                )
            def supports_streaming(self):
                return False

        with patch("lambdallm.providers.get_provider", return_value=FailThenSucceedProvider()):

            @handler(model=Model.CLAUDE_3_HAIKU, max_retries=5)
            def my_handler(event, context):
                result = context.invoke("test")
                return {"statusCode": 200, "body": {"result": result}}

            result = my_handler({}, MockLambdaContext())
            assert result["statusCode"] == 200
            assert call_count["n"] == 3  # Failed twice, succeeded on third

    def test_uses_fallback_model_on_failure(self):
        """Should fall back to secondary model when primary exhausts retries."""
        from lambdallm.core.exceptions import ModelInvocationError

        class AlwaysFailProvider:
            def invoke(self, prompt, config):
                if "haiku" in config.model_id:
                    raise ModelInvocationError("Primary down", retryable=True)
                return ModelResponse(
                    content="Fallback response", model_id="titan",
                    tokens_in=5, tokens_out=5, latency_ms=100, cost_usd=0.0001
                )
            def supports_streaming(self):
                return False

        with patch("lambdallm.providers.get_provider", return_value=AlwaysFailProvider()):

            @handler(
                model=Model.CLAUDE_3_HAIKU,
                fallback_model=Model.TITAN_TEXT_EXPRESS,
                max_retries=2,
            )
            def my_handler(event, context):
                result = context.invoke("test")
                return {"statusCode": 200, "body": {"result": result}}

            result = my_handler({}, MockLambdaContext())
            assert result["statusCode"] == 200
