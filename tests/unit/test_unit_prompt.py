"""Unit Tests: Prompt Template System

Proves Prompt works in isolation with type validation and serialization.
"""

import pytest
from lambdallm import Prompt, Model
from lambdallm.core.exceptions import ConfigurationError


class TestPromptValidation:
    """Test input validation and type safety."""

    def test_creates_with_valid_template(self):
        p = Prompt(template="Hello {name}")
        assert p.template == "Hello {name}"
        assert p._variables == ["name"]

    def test_rejects_empty_template(self):
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            Prompt(template="")

    def test_validates_schema_covers_all_variables(self):
        with pytest.raises(ConfigurationError, match="not defined"):
            Prompt(template="{a} {b}", input_schema={"a": str})

    def test_validates_input_types(self):
        p = Prompt(template="{count} items", input_schema={"count": int})
        with pytest.raises(ConfigurationError, match="expected int"):
            p.format(count="not a number")

    def test_format_substitutes_variables(self):
        p = Prompt(template="Hello {name}, age {age}")
        result = p.format(name="Gaurav", age=30)
        assert result == "Hello Gaurav, age 30"

    def test_detects_multiple_variables(self):
        p = Prompt(template="{a} and {b} and {c}")
        assert set(p._variables) == {"a", "b", "c"}


class TestPromptSerialization:
    """Test to_dict/from_dict round-trip."""

    def test_round_trip(self):
        original = Prompt(
            template="Summarize: {text}",
            input_schema={"text": str},
            output_schema={"summary": str},
            name="summarize",
            version="2.0.0",
        )

        data = original.to_dict()
        restored = Prompt.from_dict(data)

        assert restored.template == original.template
        assert restored.name == original.name
        assert restored.version == original.version

    def test_to_dict_includes_all_fields(self):
        p = Prompt(
            template="{x}",
            input_schema={"x": str},
            name="test",
            version="1.0.0",
            tags=["production"],
        )
        data = p.to_dict()
        assert data["name"] == "test"
        assert data["version"] == "1.0.0"
        assert data["tags"] == ["production"]
