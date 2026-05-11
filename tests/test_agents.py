"""Tests for the agent system."""

import pytest
from unittest.mock import MagicMock, patch

from lambdallm.agents.tool import Tool, ToolRegistry, ToolDefinition
from lambdallm.agents.agent import Agent, AgentResult
from lambdallm.agents.router import AgentRouter, RouteConfig
from lambdallm.agents.sandbox import ToolSandbox, SandboxPolicy


class TestTool:
    def test_tool_decorator_basic(self):
        @Tool(description="Add two numbers")
        def add(a: int, b: int) -> int:
            """Add two numbers together.

            Args:
                a: First number.
                b: Second number.
            """
            return a + b

        assert isinstance(add, ToolDefinition)
        assert add.name == "add"
        assert add.description == "Add two numbers"
        assert len(add.parameters) == 2
        assert add.parameters[0].name == "a"
        assert add.parameters[0].type == "int"

    def test_tool_decorator_no_parens(self):
        @Tool
        def greet(name: str) -> str:
            """Greet someone."""
            return f"Hello {name}"

        assert isinstance(greet, ToolDefinition)
        assert greet.name == "greet"

    def test_tool_schema_generation(self):
        @Tool(description="Search documents")
        def search(query: str, max_results: int = 5) -> list:
            """Search for documents.

            Args:
                query: Search query.
                max_results: Max results to return.
            """
            return []

        schema = search.to_llm_schema()
        assert schema["name"] == "search"
        assert "query" in schema["input_schema"]["properties"]
        assert "max_results" in schema["input_schema"]["properties"]
        assert "query" in schema["input_schema"]["required"]
        assert "max_results" not in schema["input_schema"]["required"]

    def test_tool_invocation(self):
        @Tool(description="Multiply")
        def multiply(x: int, y: int) -> int:
            return x * y

        registry = ToolRegistry([multiply])
        result = registry.invoke("multiply", x=3, y=4)
        assert result == 12


class TestToolRegistry:
    def test_register_and_get(self):
        @Tool(description="Test tool")
        def my_tool(x: str) -> str:
            return x

        registry = ToolRegistry([my_tool])
        assert registry.count == 1
        assert "my_tool" in registry.tool_names
        assert registry.get("my_tool") is not None

    def test_get_schemas(self):
        @Tool(description="Tool A")
        def tool_a(x: str) -> str:
            return x

        @Tool(description="Tool B")
        def tool_b(y: int) -> int:
            return y

        registry = ToolRegistry([tool_a, tool_b])
        schemas = registry.get_schemas()
        assert len(schemas) == 2


class TestAgent:
    def test_agent_creation(self):
        @Tool(description="Search")
        def search(query: str) -> str:
            return "result"

        agent = Agent(
            name="test-agent",
            system_prompt="You are helpful.",
            tools=[search],
            max_iterations=3,
        )

        assert agent.name == "test-agent"
        assert agent.registry.count == 1
        assert agent.config.max_iterations == 3

    def test_agent_run_with_mock(self):
        @Tool(description="Get weather")
        def get_weather(city: str) -> str:
            return "Sunny, 72F"

        agent = Agent(
            name="weather-agent",
            system_prompt="You help with weather.",
            tools=[get_weather],
            max_iterations=3,
        )

        # Mock context that returns a final answer immediately
        mock_context = MagicMock()
        mock_context.remaining_time_ms = 30000
        mock_context.total_cost = 0.001
        mock_context.invoke.return_value = '{"thought": "User wants weather", "final_answer": "It is sunny."}'

        result = agent.run(query="What is the weather?", context=mock_context)

        assert isinstance(result, AgentResult)
        assert result.status == "completed"
        assert "sunny" in result.answer.lower()


class TestAgentRouter:
    def test_keyword_routing(self):
        @Tool(description="Search")
        def search(q: str) -> str:
            return q

        finance_agent = Agent(name="finance", system_prompt="Finance", tools=[search])
        tech_agent = Agent(name="tech", system_prompt="Tech", tools=[search])

        router = AgentRouter(
            routes=[
                RouteConfig(agent=finance_agent, description="Finance", keywords=["revenue", "profit", "stock"]),
                RouteConfig(agent=tech_agent, description="Tech", keywords=["code", "api", "bug"]),
            ],
            strategy="keyword",
        )

        # Test routing logic (without actually running the agent)
        assert router._route_by_keyword("What was Q3 revenue?") == finance_agent
        assert router._route_by_keyword("Fix this bug in the API") == tech_agent
        assert router._route_by_keyword("Hello world") is None


class TestSandbox:
    def test_sandbox_executes_function(self):
        sandbox = ToolSandbox(SandboxPolicy(max_execution_time=5))

        def add(a, b):
            return a + b

        result = sandbox.execute(add, a=2, b=3)
        assert result == 5

    def test_sandbox_validates_actions(self):
        sandbox = ToolSandbox(SandboxPolicy(
            allowed_actions=["dynamodb:GetItem", "s3:GetObject"],
            denied_actions=["s3:DeleteObject"],
        ))

        assert sandbox.validate_action("dynamodb:GetItem") is True
        assert sandbox.validate_action("s3:GetObject") is True
        assert sandbox.validate_action("s3:DeleteObject") is False
        assert sandbox.validate_action("ec2:RunInstances") is False

    def test_sandbox_wildcard_actions(self):
        sandbox = ToolSandbox(SandboxPolicy(
            allowed_actions=["dynamodb:*"],
        ))

        assert sandbox.validate_action("dynamodb:GetItem") is True
        assert sandbox.validate_action("dynamodb:PutItem") is True
        assert sandbox.validate_action("s3:GetObject") is False
