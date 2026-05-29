"""Structured output parsing with Pydantic model validation.

This module provides utilities to extract structured data from LLM responses
by auto-generating JSON schema instructions from Pydantic models, parsing
the LLM output, and validating it against the schema with retry support.

Supports nested models, optional fields, enums, and complex type hierarchies.
"""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Callable, Generic, Optional, Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class StructuredOutputError(Exception):
    """Raised when structured output parsing fails after all retries."""

    def __init__(self, message: str, attempts: int, last_error: Optional[Exception] = None):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(message)


class SchemaInstruction:
    """Generates JSON schema instructions from Pydantic models for LLM prompts."""

    @staticmethod
    def from_model(model: Type[BaseModel], include_examples: bool = True) -> str:
        """Generate a JSON schema instruction string from a Pydantic model.

        Args:
            model: The Pydantic model class to generate instructions for.
            include_examples: Whether to include field descriptions and examples.

        Returns:
            A formatted instruction string describing the expected JSON output.
        """
        schema = model.model_json_schema()
        instruction_parts = [
            "You must respond with a valid JSON object that conforms to the following schema:",
            "",
            "```json",
            json.dumps(schema, indent=2),
            "```",
            "",
            "Important rules:",
            "- Respond ONLY with the JSON object, no additional text",
            "- All required fields must be present",
            "- Field types must match exactly as specified",
            "- Enum fields must use one of the allowed values",
        ]

        if include_examples:
            required_fields = schema.get("required", [])
            properties = schema.get("properties", {})
            if required_fields:
                instruction_parts.append("")
                instruction_parts.append("Required fields: " + ", ".join(required_fields))

            enum_fields = {
                name: prop
                for name, prop in properties.items()
                if "enum" in prop or (prop.get("allOf") and any("enum" in ref for ref in prop.get("allOf", [])))
            }
            if enum_fields:
                instruction_parts.append("")
                instruction_parts.append("Enum constraints:")
                for name, prop in enum_fields.items():
                    if "enum" in prop:
                        instruction_parts.append(f"  - {name}: one of {prop['enum']}")

        return "\n".join(instruction_parts)

    @staticmethod
    def get_compact_schema(model: Type[BaseModel]) -> dict[str, Any]:
        """Get a compact representation of the model schema.

        Args:
            model: The Pydantic model class.

        Returns:
            A simplified schema dictionary.
        """
        schema = model.model_json_schema()
        defs = schema.pop("$defs", {})
        return _resolve_refs(schema, defs)


def _resolve_refs(schema: dict[str, Any], defs: dict[str, Any]) -> dict[str, Any]:
    """Recursively resolve $ref references in a JSON schema."""
    if "$ref" in schema:
        ref_name = schema["$ref"].split("/")[-1]
        if ref_name in defs:
            return _resolve_refs(defs[ref_name], defs)
    result = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            result[key] = _resolve_refs(value, defs)
        elif isinstance(value, list):
            result[key] = [
                _resolve_refs(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value
    return result


class ResponseParser:
    """Parses JSON from LLM responses, handling common formatting issues."""

    _JSON_BLOCK_PATTERN = re.compile(r"```(?:json)?\s*\n?(.*?)\n?\s*```", re.DOTALL)
    _JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)

    @classmethod
    def extract_json(cls, response: str) -> str:
        """Extract JSON string from an LLM response.

        Handles responses wrapped in markdown code blocks or mixed with text.

        Args:
            response: The raw LLM response string.

        Returns:
            The extracted JSON string.

        Raises:
            ValueError: If no valid JSON structure is found.
        """
        match = cls._JSON_BLOCK_PATTERN.search(response)
        if match:
            return match.group(1).strip()

        match = cls._JSON_OBJECT_PATTERN.search(response)
        if match:
            return match.group(0).strip()

        stripped = response.strip()
        if stripped.startswith("{"):
            return stripped

        raise ValueError(
            "Could not extract JSON from response. "
            "Expected a JSON object or markdown code block containing JSON."
        )

    @classmethod
    def parse_json(cls, response: str) -> dict[str, Any]:
        """Parse a JSON object from an LLM response.

        Args:
            response: The raw LLM response string.

        Returns:
            The parsed JSON as a dictionary.

        Raises:
            ValueError: If JSON extraction or parsing fails.
        """
        json_str = cls.extract_json(response)
        try:
            result = json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}") from e

        if not isinstance(result, dict):
            raise ValueError(f"Expected JSON object, got {type(result).__name__}")

        return result


class StructuredOutputParser(Generic[T]):
    """Parses and validates LLM responses against a Pydantic model.

    Supports automatic retry on validation failure with customizable
    retry logic and error feedback to the LLM.

    Args:
        model: The Pydantic model class to validate against.
        max_retries: Maximum number of retry attempts on validation failure.
        retry_callback: Optional callback invoked on retry with the error message.
            Should return a new LLM response string.
    """

    def __init__(
        self,
        model: Type[T],
        max_retries: int = 3,
        retry_callback: Optional[Callable[[str, str], str]] = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.retry_callback = retry_callback

    def get_format_instructions(self, include_examples: bool = True) -> str:
        """Get the format instructions to include in the LLM prompt.

        Args:
            include_examples: Whether to include field descriptions.

        Returns:
            Formatted instruction string.
        """
        return SchemaInstruction.from_model(self.model, include_examples=include_examples)

    def parse(self, response: str) -> T:
        """Parse and validate an LLM response.

        Attempts to extract JSON from the response and validate it against
        the Pydantic model. On failure, retries using the retry_callback
        if provided.

        Args:
            response: The raw LLM response string.

        Returns:
            A validated instance of the Pydantic model.

        Raises:
            StructuredOutputError: If parsing fails after all retry attempts.
        """
        last_error: Optional[Exception] = None
        current_response = response

        for attempt in range(1, self.max_retries + 1):
            try:
                data = ResponseParser.parse_json(current_response)
                return self.model.model_validate(data)
            except (ValueError, ValidationError) as e:
                last_error = e
                error_msg = self._format_error(e)

                if attempt < self.max_retries and self.retry_callback:
                    current_response = self.retry_callback(current_response, error_msg)
                elif attempt >= self.max_retries:
                    break

        raise StructuredOutputError(
            f"Failed to parse structured output after {self.max_retries} attempts. "
            f"Last error: {last_error}",
            attempts=self.max_retries,
            last_error=last_error,
        )

    def parse_raw(self, response: str) -> tuple[Optional[T], Optional[str]]:
        """Parse without raising exceptions.

        Args:
            response: The raw LLM response string.

        Returns:
            A tuple of (parsed_model, error_message). One will be None.
        """
        try:
            result = self.parse(response)
            return result, None
        except StructuredOutputError as e:
            return None, str(e)

    def _format_error(self, error: Exception) -> str:
        """Format a validation error into a human-readable retry prompt."""
        if isinstance(error, ValidationError):
            errors = error.errors()
            parts = ["Validation failed with the following errors:"]
            for err in errors:
                loc = " -> ".join(str(x) for x in err["loc"])
                parts.append(f"  - Field '{loc}': {err['msg']} (type: {err['type']})")
            parts.append("")
            parts.append("Please fix these errors and respond with valid JSON.")
            return "\n".join(parts)
        return f"Parsing error: {error}. Please respond with valid JSON only."


def create_parser(
    model: Type[T],
    max_retries: int = 3,
    retry_callback: Optional[Callable[[str, str], str]] = None,
) -> StructuredOutputParser[T]:
    """Factory function to create a StructuredOutputParser.

    Args:
        model: The Pydantic model class to validate against.
        max_retries: Maximum number of retry attempts.
        retry_callback: Optional callback for retry logic.

    Returns:
        A configured StructuredOutputParser instance.
    """
    return StructuredOutputParser(model=model, max_retries=max_retries, retry_callback=retry_callback)
