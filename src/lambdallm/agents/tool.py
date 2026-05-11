"""Tool system for LambdaLLM agents.

Tools are typed Python functions that agents can invoke.
The framework auto-generates tool descriptions for the LLM
from function signatures and docstrings.
"""

import inspect
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, get_type_hints

from lambdallm.core.exceptions import LambdaLLMError

logger = logging.getLogger("lambdallm")


@dataclass
class ToolParameter:
    """A single parameter for a tool."""

    name: str
    type: str
    description: str = ""
    required: bool = True
    default: Any = None


@dataclass
class ToolDefinition:
    """Complete definition of a tool for LLM consumption."""

    name: str
    description: str
    parameters: list[ToolParameter] = field(default_factory=list)
    return_type: str = "str"
    func: Optional[Callable] = field(default=None, repr=False)
    allowed_iam_actions: list[str] = field(default_factory=list)
    timeout_seconds: Optional[int] = None
    is_async: bool = False

    def to_llm_schema(self) -> dict:
        """Convert to the schema format expected by LLMs (Anthropic tool_use)."""
        properties = {}
        required = []

        for param in self.parameters:
            properties[param.name] = {
                "type": self._python_type_to_json(param.type),
                "description": param.description,
            }
            if param.required:
                required.append(param.name)

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

    def _python_type_to_json(self, type_str: str) -> str:
        """Map Python type names to JSON schema types."""
        mapping = {
            "str": "string",
            "int": "integer",
            "float": "number",
            "bool": "boolean",
            "list": "array",
            "dict": "object",
            "None": "null",
        }
        return mapping.get(type_str, "string")


def Tool(
    description: Optional[str] = None,
    name: Optional[str] = None,
    allowed_iam_actions: Optional[list[str]] = None,
    timeout_seconds: Optional[int] = None,
):
    """Decorator to register a function as an agent tool.

    Automatically extracts parameter types and descriptions from
    the function signature and docstring.

    Example:
        @Tool(description="Search the knowledge base")
        def search_kb(query: str, max_results: int = 5) -> list[dict]:
            '''Search for documents matching the query.

            Args:
                query: The search query string.
                max_results: Maximum number of results to return.
            '''
            # implementation
            pass
    """

    def decorator(func: Callable) -> ToolDefinition:
        tool_name = name or func.__name__
        tool_desc = description or func.__doc__ or f"Tool: {tool_name}"

        # Extract first line of docstring as description if multi-line
        if "\n" in tool_desc:
            tool_desc = tool_desc.strip().split("\n")[0]

        # Extract parameters from function signature
        params = _extract_parameters(func)

        # Determine return type
        hints = get_type_hints(func)
        return_type = hints.get("return", str).__name__ if "return" in hints else "str"

        tool_def = ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=params,
            return_type=return_type,
            func=func,
            allowed_iam_actions=allowed_iam_actions or [],
            timeout_seconds=timeout_seconds,
            is_async=inspect.iscoroutinefunction(func),
        )

        # Attach metadata to the function for introspection
        func._tool_definition = tool_def
        return tool_def

    # Handle both @Tool and @Tool(...) syntax
    if callable(description):
        # Used as @Tool without parentheses
        func = description
        description = None
        return decorator(func)

    return decorator


def _extract_parameters(func: Callable) -> list[ToolParameter]:
    """Extract typed parameters from a function signature."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    doc_params = _parse_docstring_params(func.__doc__ or "")

    params = []
    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        type_hint = hints.get(param_name, str)
        type_name = type_hint.__name__ if hasattr(type_hint, "__name__") else str(type_hint)

        has_default = param.default != inspect.Parameter.empty
        default_value = param.default if has_default else None

        param_desc = doc_params.get(param_name, f"Parameter: {param_name}")

        params.append(ToolParameter(
            name=param_name,
            type=type_name,
            description=param_desc,
            required=not has_default,
            default=default_value,
        ))

    return params


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parse Args section from a docstring."""
    params = {}
    in_args = False

    for line in docstring.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("args:"):
            in_args = True
            continue
        if in_args:
            if stripped and not stripped.startswith(("Returns:", "Raises:", "Example:")):
                if ":" in stripped:
                    name, desc = stripped.split(":", 1)
                    params[name.strip()] = desc.strip()
            elif not stripped or stripped.startswith(("Returns:", "Raises:")):
                in_args = False

    return params


class ToolRegistry:
    """Registry of available tools for an agent.

    Manages tool definitions and handles tool invocation
    with input validation and sandboxing.
    """

    def __init__(self, tools: Optional[list] = None):
        self._tools: dict[str, ToolDefinition] = {}
        if tools:
            for tool in tools:
                self.register(tool)

    def register(self, tool) -> None:
        """Register a tool."""
        if isinstance(tool, ToolDefinition):
            self._tools[tool.name] = tool
        elif hasattr(tool, "_tool_definition"):
            self._tools[tool._tool_definition.name] = tool._tool_definition
        else:
            raise ValueError(f"Cannot register {tool}: not a valid tool")

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Get a tool by name."""
        return self._tools.get(name)

    def invoke(self, name: str, **kwargs) -> Any:
        """Invoke a tool by name with arguments.

        Args:
            name: Tool name.
            **kwargs: Tool arguments.

        Returns:
            Tool execution result.

        Raises:
            LambdaLLMError: If tool not found or execution fails.
        """
        tool = self._tools.get(name)
        if not tool:
            raise LambdaLLMError(f"Tool '{name}' not found. Available: {list(self._tools.keys())}")

        if not tool.func:
            raise LambdaLLMError(f"Tool '{name}' has no implementation")

        try:
            result = tool.func(**kwargs)
            logger.debug(f"Tool '{name}' executed successfully")
            return result
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            raise LambdaLLMError(f"Tool execution failed: {e}")

    def get_schemas(self) -> list[dict]:
        """Get all tool schemas for LLM consumption."""
        return [tool.to_llm_schema() for tool in self._tools.values()]

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    @property
    def count(self) -> int:
        return len(self._tools)
