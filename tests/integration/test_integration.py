"""Integration Tests: Components Working Together

Proves that handler + chains + sessions + agents work end-to-end.
Uses mock providers (no AWS needed) but tests real component interaction.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from lambdallm import handler, Chain, Step, Model, Session, Prompt
from lambdallm.agents import Agent, Tool
from lambdallm.core.models import ModelResponse
from lambdallm.testing import MockProvider, MockLambdaContext, mock_model
from lambdallm.state.memory import InMemoryStateStore


class TestHandlerWithPrompt:
    """Test handler + prompt integration."""

    @mock_model(responses=["The document discusses AI and serverless computing."])
    def test_handler_invokes_prompt(self):
        summarize = Prompt(
            template="Summarize: {text}",
            input_schema={"text": str},
        )

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            body = json.loads(event.get("body", "{}"))
            result = context.invoke("Summarize: {text}", text=body["text"])
            return {"statusCode": 200, "body": {"summary": result}}

        event = {"body": json.dumps({"text": "Long document about AI..."})}
        result = my_handler(event, MockLambdaContext())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "summary" in body
        assert len(body["summary"]) > 0


class TestChainExecution:
    """Test multi-step chain execution end-to-end."""

    @mock_model(responses=[
        "Entities: Person(John), Org(Acme)",
        "John: CEO, Acme: Technology company",
        "Summary: John leads Acme in tech innovation.",
    ])
    def test_three_step_chain_completes(self):
        chain = Chain(
            name="analysis",
            steps=[
                Step("extract", prompt="Extract entities: {input}"),
                Step("classify", prompt="Classify: {extract.output}"),
                Step("summarize", prompt="Summarize: {classify.output}"),
            ],
        )

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            result = chain.run(context=context, input="John is CEO of Acme Corp.")
            return {"statusCode": 200, "body": {"result": result.final_output, "steps": result.completed_steps}}

        result = my_handler({}, MockLambdaContext())
        body = json.loads(result["body"])
        assert body["steps"] == 3
        assert body["result"] is not None

    @mock_model(responses=["Step 1 done", "Step 2 done"])
    def test_chain_with_transform_step(self):
        chain = Chain(
            name="transform-chain",
            steps=[
                Step("generate", prompt="Generate: {input}"),
                Step("upper", func=lambda data: data["generate"].upper()),
                Step("enhance", prompt="Enhance: {upper.output}"),
            ],
        )

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            result = chain.run(context=context, input="test")
            return {"statusCode": 200, "body": {"status": result.status}}

        result = my_handler({}, MockLambdaContext())
        body = json.loads(result["body"])
        assert body["status"] == "completed"

    @mock_model(responses=["Only this runs"])
    def test_chain_conditional_step_skips(self):
        chain = Chain(
            name="conditional",
            steps=[
                Step("always", prompt="Do: {input}"),
                Step("never", prompt="Skip: {always.output}", condition=lambda _: False),
            ],
        )

        @handler(model=Model.CLAUDE_3_HAIKU)
        def my_handler(event, context):
            result = chain.run(context=context, input="test")
            return {"statusCode": 200, "body": {"completed": result.completed_steps}}

        result = my_handler({}, MockLambdaContext())
        body = json.loads(result["body"])
        assert body["completed"] == 1


class TestSessionIntegration:
    """Test session state persistence across simulated invocations."""

    def test_session_persists_messages(self):
        store = InMemoryStateStore()

        session = Session(store="memory", max_messages=10)
        session._store_instance = store
        session.session_id = "test-session-123"

        # First invocation
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")
        session.save()

        # Second invocation (simulates new Lambda invocation)
        session2 = Session(store="memory", max_messages=10)
        session2._store_instance = store
        session2.load("test-session-123")

        assert session2.message_count == 2
        assert session2.messages[0].content == "Hello"
        assert session2.messages[1].content == "Hi there!"

    def test_sliding_window_trims_old_messages(self):
        session = Session(store="memory", max_messages=3)

        for i in range(5):
            session.add_message("user", f"Message {i}")

        assert session.message_count == 3
        assert session.messages[0].content == "Message 2"


class TestAgentIntegration:
    """Test agent with tools end-to-end."""

    @mock_model(responses=['{"thought": "I should search", "tool_name": "search", "tool_input": {"query": "AI"}}',
                           '{"thought": "Found info", "final_answer": "AI is artificial intelligence."}'])
    def test_agent_uses_tool_and_answers(self):
        @Tool(description="Search knowledge base")
        def search(query: str) -> str:
            return f"Results for {query}: AI is artificial intelligence."

        agent = Agent(
            name="test-agent",
            system_prompt="You are helpful.",
            tools=[search],
            max_iterations=5,
        )

        @handler(model=Model.CLAUDE_3_SONNET)
        def my_handler(event, context):
            result = agent.run(query="What is AI?", context=context)
            return {"statusCode": 200, "body": {"answer": result.answer, "tool_calls": result.total_tool_calls}}

        result = my_handler({}, MockLambdaContext())
        body = json.loads(result["body"])
        assert "AI" in body["answer"] or "artificial" in body["answer"].lower()
        assert body["tool_calls"] >= 1
