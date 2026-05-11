"""Tests for the Prompt template system."""

import pytest
from lambdallm.core.prompt import Prompt
from lambdallm.core.exceptions import ConfigurationError


class TestPrompt:
    """Test Prompt template creation and validation."""

    def test_basic_prompt_creation(self):
        """Prompt should be created with a template."""
        p = Prompt(template="Hello {name}")
        assert p.template == "Hello {name}"

    def test_prompt_extracts_variables(self):
        """Prompt should auto-detect template variables."""
        p = Prompt(template="Summarize {document} in {max_words} words")
        assert set(p._variables) == {"document", "max_words"}

    def test_prompt_format(self):
        """Prompt.format() should substitute variables."""
        p = Prompt(template="Hello {name}, you are {age} years old")
        result = p.format(name="Gaurav", age=30)
        assert result == "Hello Gaurav, you are 30 years old"

    def test_prompt_validates_missing_variables(self):
        """Prompt should raise error for missing required variables."""
        p = Prompt(
            template="Hello {name}",
            input_schema={"name": str},
        )
        with pytest.raises(ConfigurationError, match="Missing required variables"):
            p.format()

    def test_prompt_validates_types(self):
        """Prompt should validate variable types against schema."""
        p = Prompt(
            template="Count to {number}",
            input_schema={"number": int},
        )
        with pytest.raises(ConfigurationError, match="expected int"):
            p.format(number="not a number")

    def test_prompt_empty_template_raises(self):
        """Empty template should raise ConfigurationError."""
        with pytest.raises(ConfigurationError, match="cannot be empty"):
            Prompt(template="")

    def test_prompt_schema_mismatch_raises(self):
        """Template vars not in schema should raise error."""
        with pytest.raises(ConfigurationError, match="not defined in input_schema"):
            Prompt(
                template="Hello {name} {age}",
                input_schema={"name": str},  # missing 'age'
            )

    def test_prompt_serialization(self):
        """Prompt should serialize to/from dict."""
        p = Prompt(
            template="Summarize: {text}",
            input_schema={"text": str},
            output_schema={"summary": str},
            name="summarize",
            version="1.0.0",
        )

        data = p.to_dict()
        assert data["name"] == "summarize"
        assert data["version"] == "1.0.0"
        assert data["input_schema"] == {"text": "str"}

        # Round-trip
        p2 = Prompt.from_dict(data)
        assert p2.template == p.template
        assert p2.name == p.name

    def test_prompt_with_output_schema(self):
        """Prompt with output_schema should be created successfully."""
        p = Prompt(
            template="Analyze: {text}",
            input_schema={"text": str},
            output_schema={"sentiment": str, "score": float},
        )
        assert p.output_schema == {"sentiment": str, "score": float}
