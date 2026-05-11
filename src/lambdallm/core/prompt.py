"""Prompt template system for LambdaLLM.

Type-safe prompt definitions with variable injection, validation,
and structured output parsing.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.core.models import Model, ModelConfig
from lambdallm.core.exceptions import ConfigurationError

logger = logging.getLogger("lambdallm")


@dataclass
class Prompt:
    """A reusable, type-safe prompt template.

    Prompts are the declarative building blocks of LambdaLLM.
    Define once, invoke anywhere with type safety.

    Example:
        summarize = Prompt(
            template="Summarize in {max_words} words:\\n\\n{document}",
            input_schema={"document": str, "max_words": int},
            output_schema={"summary": str, "key_points": list}
        )

        result = summarize.invoke(document="...", max_words=100)
    """

    template: str
    input_schema: Optional[dict[str, type]] = None
    output_schema: Optional[dict[str, type]] = None
    system_prompt: Optional[str] = None
    model: Optional[Model] = None
    max_tokens: int = 1024
    temperature: float = 0.7
    name: Optional[str] = None
    description: Optional[str] = None
    version: str = "1.0.0"
    tags: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate the prompt template on creation."""
        if not self.template:
            raise ConfigurationError("Prompt template cannot be empty")

        # Extract template variables
        self._variables = self._extract_variables()

        # Validate input_schema matches template variables
        if self.input_schema:
            schema_keys = set(self.input_schema.keys())
            template_vars = set(self._variables)
            missing = template_vars - schema_keys
            if missing:
                raise ConfigurationError(
                    f"Template variables {missing} not defined in input_schema"
                )

    def format(self, **kwargs) -> str:
        """Format the template with provided variables.

        Args:
            **kwargs: Variable values matching the template placeholders.

        Returns:
            Formatted prompt string.

        Raises:
            ConfigurationError: If required variables are missing or wrong type.
        """
        self._validate_inputs(kwargs)
        return self.template.format(**kwargs)

    def invoke(self, _context=None, **kwargs) -> Any:
        """Invoke the prompt with an LLM and return the result.

        If output_schema is defined, returns a parsed dict.
        Otherwise returns the raw string response.

        Args:
            _context: Optional LambdaLLMContext. If None, creates a default one.
            **kwargs: Variables to substitute into the template.

        Returns:
            Parsed response (dict if output_schema defined, str otherwise).
        """
        formatted = self.format(**kwargs)

        if _context is None:
            # Create a minimal context for standalone usage
            from lambdallm.core.context import LambdaLLMContext

            _context = LambdaLLMContext(
                model=ModelConfig.from_model(self.model or Model.CLAUDE_3_HAIKU),
                timeout_strategy="fail-fast",
                timeout_buffer=5,
                max_retries=3,
                fallback_model=None,
                lambda_context=None,
                middleware=[],
            )

        if self.output_schema:
            return _context.invoke_structured(formatted, self.output_schema)
        else:
            return _context.invoke(formatted)

    def _validate_inputs(self, kwargs: dict) -> None:
        """Validate input variables against schema."""
        if not self.input_schema:
            return

        # Check for missing required variables
        required = set(self._variables)
        provided = set(kwargs.keys())
        missing = required - provided
        if missing:
            raise ConfigurationError(f"Missing required variables: {missing}")

        # Type checking
        for key, expected_type in self.input_schema.items():
            if key in kwargs and not isinstance(kwargs[key], expected_type):
                raise ConfigurationError(
                    f"Variable '{key}' expected {expected_type.__name__}, "
                    f"got {type(kwargs[key]).__name__}"
                )

    def _extract_variables(self) -> list[str]:
        """Extract variable names from the template string."""
        import re

        # Match {variable_name} but not {{escaped}}
        pattern = r"(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})"
        return re.findall(pattern, self.template)

    def to_dict(self) -> dict:
        """Serialize prompt to dictionary (for versioning/storage)."""
        return {
            "name": self.name,
            "version": self.version,
            "template": self.template,
            "input_schema": {k: v.__name__ for k, v in (self.input_schema or {}).items()},
            "output_schema": {k: v.__name__ for k, v in (self.output_schema or {}).items()},
            "system_prompt": self.system_prompt,
            "model": self.model.value if self.model else None,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Prompt":
        """Deserialize prompt from dictionary."""
        type_map = {"str": str, "int": int, "float": float, "list": list, "dict": dict, "bool": bool}

        input_schema = None
        if data.get("input_schema"):
            input_schema = {k: type_map.get(v, str) for k, v in data["input_schema"].items()}

        output_schema = None
        if data.get("output_schema"):
            output_schema = {k: type_map.get(v, str) for k, v in data["output_schema"].items()}

        return cls(
            template=data["template"],
            input_schema=input_schema,
            output_schema=output_schema,
            system_prompt=data.get("system_prompt"),
            model=Model(data["model"]) if data.get("model") else None,
            max_tokens=data.get("max_tokens", 1024),
            temperature=data.get("temperature", 0.7),
            name=data.get("name"),
            version=data.get("version", "1.0.0"),
            tags=data.get("tags", []),
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Prompt":
        """Load a prompt from a YAML file."""
        try:
            import yaml
        except ImportError:
            raise ConfigurationError("PyYAML required for YAML prompts: pip install pyyaml")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)
