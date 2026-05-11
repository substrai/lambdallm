"""Unit Tests: Tool System

Proves @Tool decorator and ToolRegistry work in isolation.
"""

import pytest
from lambdallm.agents import Tool, ToolRegistry
from lambdallm.agents.tool import ToolDefinition
from lambdallm.core.exceptions import LambdaLLMError


class TestToolDecorator:
    """Test @Tool auto-generates schemas from functions."""

    def test_basic_tool(self):
        @Tool(description="Add numbers")
        def add(a: int, b: int) -> int:
            return a + b

        assert isinstance(add, ToolDefinition)
        assert add.name == "add"
        assert add.description == "Add numbers"
        assert len(add.parameters) == 2

    def test_extracts_parameter_types(self):
        @Tool(description="Search")
        def search(query: str, limit: int = 10) -> list:
            return []

        assert search.parameters[0].name == "query"
        assert search.parameters[0].type == "str"
        assert search.parameters[0].required is True
        assert search.parameters[1].name == "limit"
        assert search.parameters[1].required is False
        assert search.parameters[1].default == 10

    def test_generates_llm_schema(self):
        @Tool(description="Get weather")
        def weather(city: str) -> str:
            return "sunny"

        schema = weather.to_llm_schema()
        assert schema["name"] == "weather"
        assert schema["description"] == "Get weather"
        assert "city" in schema["input_schema"]["properties"]
        assert "city" in schema["input_schema"]["required"]

    def test_tool_without_parens(self):
        @Tool
        def greet(name: str) -> str:
            return f"Hello {name}"

        assert isinstance(greet, ToolDefinition)
        assert greet.name == "greet"


class TestToolRegistry:
    """Test tool registration and invocation."""

    def test_register_and_invoke(self):
        @Tool(description="Multiply")
        def multiply(x: int, y: int) -> int:
            return x * y

        registry = ToolRegistry([multiply])
        result = registry.invoke("multiply", x=3, y=7)
        assert result == 21

    def test_invoke_unknown_tool_raises(self):
        registry = ToolRegistry([])
        with pytest.raises(LambdaLLMError, match="not found"):
            registry.invoke("nonexistent", x=1)

    def test_get_all_schemas(self):
        @Tool(description="A")
        def tool_a(x: str) -> str:
            return x

        @Tool(description="B")
        def tool_b(y: int) -> int:
            return y

        registry = ToolRegistry([tool_a, tool_b])
        schemas = registry.get_schemas()
        assert len(schemas) == 2
        names = [s["name"] for s in schemas]
        assert "tool_a" in names
        assert "tool_b" in names
