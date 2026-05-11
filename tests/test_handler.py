"""Tests for the @handler decorator."""

import json
import pytest
from unittest.mock import patch, MagicMock

from lambdallm import handler, Model
from lambdallm.core.models import ModelResponse


class MockLambdaContext:
    """Mock AWS Lambda context object."""

    function_name = "test-function"
    memory_limit_in_mb = 256
    invoked_function_arn = "arn:aws:lambda:us-east-1:123456789:function:test"

    def get_remaining_time_in_millis(self):
        return 30000  # 30 seconds


@pytest.fixture
def mock_provider():
    """Mock the Bedrock provider to avoid real API calls."""
    with patch("lambdallm.providers.get_provider") as mock:
        provider = MagicMock()
        provider.invoke.return_value = ModelResponse(
            content="This is a test response.",
            model_id="anthropic.claude-3-haiku-20240307-v1:0",
            tokens_in=10,
            tokens_out=20,
            latency_ms=150.0,
            cost_usd=0.00003,
        )
        mock.return_value = provider
        yield provider


class TestHandler:
    """Test the @handler decorator."""

    def test_basic_handler_returns_200(self, mock_provider):
        """Handler should return 200 with formatted response."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            return {"statusCode": 200, "body": {"message": "hello"}}

        result = my_handler({"body": "test"}, MockLambdaContext())

        assert result["statusCode"] == 200
        assert "hello" in result["body"]

    def test_handler_auto_wraps_response(self, mock_provider):
        """Handler should auto-wrap non-dict responses."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            return {"result": "success"}

        result = my_handler({}, MockLambdaContext())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["result"] == "success"

    def test_handler_catches_exceptions(self, mock_provider):
        """Handler should catch exceptions and return error response."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            raise ValueError("Something went wrong")

        result = my_handler({}, MockLambdaContext())

        assert result["statusCode"] == 500
        body = json.loads(result["body"])
        assert "error" in body

    def test_handler_has_metadata(self, mock_provider):
        """Handler should expose framework metadata."""

        @handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="checkpoint")
        def my_handler(event, context):
            return {"statusCode": 200}

        assert my_handler._lambdallm_handler is True
        assert my_handler._lambdallm_config["timeout_strategy"] == "checkpoint"

    def test_context_invoke(self, mock_provider):
        """Context.invoke should call the provider and return content."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            result = context.invoke("Hello {name}", name="World")
            return {"statusCode": 200, "body": {"response": result}}

        result = my_handler({"body": "test"}, MockLambdaContext())

        assert result["statusCode"] == 200
        mock_provider.invoke.assert_called_once()

    def test_context_tracks_cost(self, mock_provider):
        """Context should track cumulative cost."""

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            context.invoke("First call")
            context.invoke("Second call")
            return {
                "statusCode": 200,
                "body": {"total_cost": context.total_cost},
            }

        result = my_handler({}, MockLambdaContext())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert body["total_cost"] > 0
