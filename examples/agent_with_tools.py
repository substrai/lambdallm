"""Example: AI agent with tools for research and analysis.

Demonstrates how to build a ReAct-style agent that uses tools
to answer complex questions, all within Lambda's constraints.
"""

from lambdallm import handler, Model
from lambdallm.agents import Tool, Agent


# Define tools the agent can use
@Tool(description="Search the knowledge base for relevant documents")
def search_kb(query: str, max_results: int = 3) -> list[dict]:
    """Search for documents matching the query.

    Args:
        query: The search query string.
        max_results: Maximum number of results to return.
    """
    # In production, this would call OpenSearch, Bedrock KB, etc.
    return [
        {"title": f"Result for '{query}'", "content": f"Relevant information about {query}..."},
    ]


@Tool(description="Calculate a mathematical expression")
def calculate(expression: str) -> float:
    """Safely evaluate a mathematical expression.

    Args:
        expression: Mathematical expression to evaluate (e.g., '2 + 2', '100 * 0.15').
    """
    # Safe eval for basic math
    allowed_chars = set("0123456789+-*/.() ")
    if not all(c in allowed_chars for c in expression):
        raise ValueError(f"Invalid expression: {expression}")
    return eval(expression)


@Tool(description="Get current date and time")
def get_current_time() -> str:
    """Get the current UTC date and time."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Create the agent
research_agent = Agent(
    name="research-analyst",
    system_prompt="""You are a research analyst. Use the available tools to 
    gather information and perform calculations to answer questions accurately.
    Always cite your sources and show your work for calculations.""",
    tools=[search_kb, calculate, get_current_time],
    max_iterations=5,
    timeout_buffer=30,  # Reserve 30s before Lambda timeout
    max_cost_usd=0.10,  # Stop if cost exceeds $0.10
    verbose=True,
)


@handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    """Research agent Lambda handler.

    Expected event body:
    {
        "question": "What is the total revenue if we grow 15% from $1M?"
    }
    """
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    question = body.get("question", "")

    # Run the agent
    result = research_agent.run(query=question, context=context)

    return {
        "statusCode": 200,
        "body": {
            "answer": result.answer,
            "status": result.status,
            "iterations": result.total_iterations,
            "tool_calls": result.total_tool_calls,
            "cost_usd": result.total_cost_usd,
            "latency_ms": result.total_latency_ms,
            "steps": [
                {
                    "thought": s.thought[:200],
                    "tool": s.tool_name,
                    "tool_output": str(s.tool_output)[:200] if s.tool_output else None,
                }
                for s in result.steps
            ],
        },
    }
