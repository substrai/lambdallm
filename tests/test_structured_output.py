"""Tests for structured output parsing with Pydantic model validation."""

import json
from enum import Enum
from typing import Optional
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from lambdallm.core.structured_output import (
    ResponseParser,
    SchemaInstruction,
    StructuredOutputError,
    StructuredOutputParser,
    create_parser,
)


# --- Test Models ---


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class Address(BaseModel):
    street: str
    city: str
    zip_code: str
    country: str = "US"


class Person(BaseModel):
    name: str
    age: int
    email: Optional[str] = None
    address: Optional[Address] = None


class ReviewAnalysis(BaseModel):
    sentiment: Sentiment
    confidence: float
    summary: str
    keywords: list[str]


# --- Tests ---


class TestSchemaInstruction:
    """Tests for SchemaInstruction generation."""

    def test_from_model_generates_valid_instruction(self):
        """Test that schema instruction contains JSON schema."""
        instruction = SchemaInstruction.from_model(Person)
        assert "```json" in instruction
        assert '"name"' in instruction
        assert '"age"' in instruction
        assert "required" in instruction.lower()

    def test_from_model_includes_enum_constraints(self):
        """Test that enum fields are documented in instructions."""
        instruction = SchemaInstruction.from_model(ReviewAnalysis)
        assert "sentiment" in instruction
        assert "positive" in instruction or "Enum" in instruction

    def test_get_compact_schema_resolves_nested_refs(self):
        """Test that nested model references are resolved."""
        schema = SchemaInstruction.get_compact_schema(Person)
        assert "properties" in schema
        assert "name" in schema["properties"]

    def test_from_model_without_examples(self):
        """Test instruction generation without examples."""
        instruction = SchemaInstruction.from_model(Person, include_examples=False)
        assert "```json" in instruction
        assert "Required fields:" not in instruction


class TestResponseParser:
    """Tests for JSON extraction from LLM responses."""

    def test_extract_json_from_code_block(self):
        """Test extraction from markdown code block."""
        response = '```json\n{"name": "Alice", "age": 30}\n```'
        result = ResponseParser.parse_json(response)
        assert result == {"name": "Alice", "age": 30}

    def test_extract_json_from_raw_object(self):
        """Test extraction from raw JSON in response."""
        response = 'Here is the result: {"name": "Bob", "age": 25}'
        result = ResponseParser.parse_json(response)
        assert result["name"] == "Bob"

    def test_extract_json_raises_on_no_json(self):
        """Test that ValueError is raised when no JSON found."""
        response = "I cannot provide that information."
        with pytest.raises(ValueError, match="Could not extract JSON"):
            ResponseParser.extract_json(response)

    def test_extract_json_raises_on_invalid_json(self):
        """Test that ValueError is raised for malformed JSON."""
        response = '{"name": "Alice", "age":}'
        with pytest.raises(ValueError, match="Invalid JSON"):
            ResponseParser.parse_json(response)


class TestStructuredOutputParser:
    """Tests for the main parser with validation and retry."""

    def test_parse_valid_response(self):
        """Test successful parsing of a valid response."""
        parser = create_parser(Person)
        response = json.dumps({"name": "Alice", "age": 30})
        result = parser.parse(response)
        assert result.name == "Alice"
        assert result.age == 30

    def test_parse_nested_model(self):
        """Test parsing with nested Pydantic models."""
        parser = create_parser(Person)
        response = json.dumps({
            "name": "Bob",
            "age": 40,
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zip_code": "62701",
            },
        })
        result = parser.parse(response)
        assert result.address is not None
        assert result.address.city == "Springfield"
        assert result.address.country == "US"

    def test_parse_enum_field(self):
        """Test parsing with enum field validation."""
        parser = create_parser(ReviewAnalysis)
        response = json.dumps({
            "sentiment": "positive",
            "confidence": 0.95,
            "summary": "Great product",
            "keywords": ["quality", "value"],
        })
        result = parser.parse(response)
        assert result.sentiment == Sentiment.POSITIVE

    def test_parse_with_retry_on_validation_failure(self):
        """Test that retry callback is invoked on validation failure."""
        valid_response = json.dumps({"name": "Alice", "age": 30})
        callback = MagicMock(return_value=valid_response)

        parser = create_parser(Person, max_retries=3, retry_callback=callback)
        invalid_response = json.dumps({"name": "Alice"})  # missing 'age'

        result = parser.parse(invalid_response)
        assert result.name == "Alice"
        assert callback.called

    def test_parse_raises_after_max_retries(self):
        """Test that StructuredOutputError is raised after exhausting retries."""
        callback = MagicMock(return_value='{"invalid": true}')
        parser = create_parser(Person, max_retries=2, retry_callback=callback)

        with pytest.raises(StructuredOutputError) as exc_info:
            parser.parse("not json at all")

        assert exc_info.value.attempts == 2

    def test_parse_raw_returns_none_on_failure(self):
        """Test parse_raw returns error tuple instead of raising."""
        parser = create_parser(Person, max_retries=1)
        result, error = parser.parse_raw("garbage")
        assert result is None
        assert error is not None

    def test_parse_optional_fields(self):
        """Test that optional fields default to None."""
        parser = create_parser(Person)
        response = json.dumps({"name": "Charlie", "age": 22})
        result = parser.parse(response)
        assert result.email is None
        assert result.address is None

    def test_get_format_instructions(self):
        """Test that format instructions are generated correctly."""
        parser = create_parser(Person)
        instructions = parser.get_format_instructions()
        assert "JSON" in instructions
        assert "name" in instructions
