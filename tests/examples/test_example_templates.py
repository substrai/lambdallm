"""Example Test Templates for LambdaLLM Users

Copy these templates and adapt them for your own handlers.
All examples use mock providers - no AWS credentials needed.

Usage:
    1. Copy this file to your project's tests/ directory
    2. Replace the handler imports with your own
    3. Adjust mock responses to match your expected outputs
    4. Run: pytest tests/ -v
"""

import json
import pytest
from unittest.mock import patch

from lambdallm import handler, Prompt, Model
from lambdallm.testing import MockProvider, MockLambdaContext, mock_model


# ============================================================
# TEMPLATE 1: Testing a Basic Handler
# ============================================================

class TestBasicHandler:
    """Template for testing a simple LLM handler.

    Copy this and replace with your handler.
    """

    # Define your handler (or import from your module)
    @staticmethod
    def create_handler():
        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            body = json.loads(event.get("body", "{}"))
            result = context.invoke("Answer: {question}", question=body.get("question", ""))
            return {"statusCode": 200, "body": {"answer": result}}
        return my_handler

    @mock_model(responses=["The answer is 42."])
    def test_returns_200(self):
        handler_func = self.create_handler()
        event = {"body": json.dumps({"question": "What is the meaning of life?"})}
        result = handler_func(event, MockLambdaContext())
        assert result["statusCode"] == 200

    @mock_model(responses=["The answer is 42."])
    def test_response_has_answer(self):
        handler_func = self.create_handler()
        event = {"body": json.dumps({"question": "What is 6 * 7?"})}
        result = handler_func(event, MockLambdaContext())
        body = json.loads(result["body"])
        assert "answer" in body
        assert len(body["answer"]) > 0

    @mock_model(responses=["response"])
    def test_handles_empty_input(self):
        handler_func = self.create_handler()
        result = handler_func({"body": "{}"}, MockLambdaContext())
        assert result["statusCode"] == 200


# ============================================================
# TEMPLATE 2: Testing a Handler with Structured Output
# ============================================================

class TestStructuredOutputHandler:
    """Template for testing handlers that return structured JSON."""

    @staticmethod
    def create_handler():
        analyze = Prompt(
            template="Analyze sentiment: {text}",
            input_schema={"text": str},
            output_schema={"sentiment": str, "confidence": float},
        )

        @handler(model=Model.CLAUDE_3_HAIKU)
        def sentiment_handler(event, context):
            body = json.loads(event.get("body", "{}"))
            result = context.invoke_structured(
                "Analyze sentiment of: {text}",
                {"sentiment": str, "confidence": float},
                text=body.get("text", ""),
            )
            return {"statusCode": 200, "body": result}
        return sentiment_handler

    @mock_model(responses=['{"sentiment": "positive", "confidence": 0.95}'])
    def test_returns_structured_output(self):
        handler_func = self.create_handler()
        event = {"body": json.dumps({"text": "I love this product!"})}
        result = handler_func(event, MockLambdaContext())
        body = json.loads(result["body"])
        assert "sentiment" in body
        assert "confidence" in body


# ============================================================
# TEMPLATE 3: Testing Error Handling
# ============================================================

class TestErrorHandling:
    """Template for testing error scenarios."""

    @staticmethod
    def create_handler():
        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            body = json.loads(event.get("body", "{}"))
            if not body.get("text"):
                raise ValueError("text field is required")
            result = context.invoke("Process: {text}", text=body["text"])
            return {"statusCode": 200, "body": {"result": result}}
        return my_handler

    @mock_model(responses=["ignored"])
    def test_missing_field_returns_500(self):
        handler_func = self.create_handler()
        result = handler_func({"body": "{}"}, MockLambdaContext())
        assert result["statusCode"] == 500

    @mock_model(responses=["Success"])
    def test_valid_input_returns_200(self):
        handler_func = self.create_handler()
        event = {"body": json.dumps({"text": "valid input"})}
        result = handler_func(event, MockLambdaContext())
        assert result["statusCode"] == 200


# ============================================================
# TEMPLATE 4: Testing with Multiple Mock Responses
# ============================================================

class TestMultipleInvocations:
    """Template for handlers that make multiple LLM calls."""

    @mock_model(responses=["First response", "Second response", "Third response"])
    def test_multiple_calls_get_different_responses(self):
        @handler(model=Model.CLAUDE_3_HAIKU)
        def multi_handler(event, context):
            r1 = context.invoke("First")
            r2 = context.invoke("Second")
            r3 = context.invoke("Third")
            return {"statusCode": 200, "body": {"responses": [r1, r2, r3]}}

        result = multi_handler({}, MockLambdaContext())
        body = json.loads(result["body"])
        assert len(body["responses"]) == 3
        assert body["responses"][0] == "First response"
        assert body["responses"][1] == "Second response"


# ============================================================
# HOW TO RUN
# ============================================================
# pytest tests/examples/test_example_templates.py -v
# pytest tests/examples/ -v --tb=short
