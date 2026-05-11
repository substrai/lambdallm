"""lambdallm init - Project scaffolding command.

Generates a complete project structure with sensible defaults.
Convention Over Configuration: works immediately after init.
"""

import os
import sys

TEMPLATES = {
    "basic": {
        "description": "Basic LLM handler with prompt template",
        "handler_code": '''"""Basic LambdaLLM handler."""

from lambdallm import handler, Prompt, Model

summarize = Prompt(
    name="summarize",
    template="""Summarize the following in {max_words} words:

{document}""",
    input_schema={"document": str, "max_words": int},
    output_schema={"summary": str, "key_points": list},
)


@handler(model=Model.CLAUDE_3_HAIKU)
def lambda_handler(event, context):
    """Main Lambda handler."""
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    result = summarize.invoke(
        _context=context,
        document=body.get("text", ""),
        max_words=body.get("max_words", 100),
    )

    return {
        "statusCode": 200,
        "body": {"result": result, "cost_usd": context.total_cost},
    }
''',
    },
    "chat": {
        "description": "Multi-turn chat with session memory",
        "handler_code": '''"""Chat handler with conversation memory."""

from lambdallm import handler, Model
from lambdallm.state import Session, MemoryStrategy


@handler(
    model=Model.CLAUDE_3_SONNET,
    session=Session(store="dynamodb", ttl_hours=24, max_messages=20),
)
def lambda_handler(event, context):
    """Multi-turn chat Lambda handler."""
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    # Load session (auto-loads from DynamoDB)
    session = context.session
    if session:
        session.load(session_id)
        session.add_message("user", message)

    # Invoke with conversation history
    history = session.format_history() if session else message
    response = context.invoke(
        "You are a helpful assistant.\\n\\nConversation:\\n{history}\\n\\nassistant:",
        history=history,
    )

    if session:
        session.add_message("assistant", response)
        session.save()

    return {
        "statusCode": 200,
        "body": {"reply": response, "session_id": session_id},
    }
''',
    },
    "agent": {
        "description": "AI agent with tool usage (Phase 3 preview)",
        "handler_code": '''"""Agent handler with tools (Phase 3 preview)."""

from lambdallm import handler, Model


@handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    """Agent Lambda handler - tools coming in Phase 3."""
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    question = body.get("question", "")

    response = context.invoke(
        "You are a helpful agent. Answer this question: {question}",
        question=question,
    )

    return {
        "statusCode": 200,
        "body": {"answer": response, "cost_usd": context.total_cost},
    }
''',
    },
    "rag": {
        "description": "RAG (Retrieval-Augmented Generation) handler",
        "handler_code": '''"""RAG handler - retrieval augmented generation."""

from lambdallm import handler, Prompt, Model

answer_with_context = Prompt(
    name="rag-answer",
    template="""Answer the question based on the provided context.
If the context does not contain the answer, say "I don't have enough information."

Context:
{context}

Question: {question}""",
    input_schema={"context": str, "question": str},
)


@handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    """RAG Lambda handler."""
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    question = body.get("question", "")
    # In production, retrieve context from vector DB / Bedrock KB
    retrieved_context = body.get("context", "No context provided.")

    result = answer_with_context.invoke(
        _context=context,
        context=retrieved_context,
        question=question,
    )

    return {
        "statusCode": 200,
        "body": {"answer": result, "cost_usd": context.total_cost},
    }
''',
    },
}

CONFIG_TEMPLATE = '''project:
  name: "{name}"
  version: "0.1.0"
  runtime: python3.12

defaults:
  model: bedrock/claude-3-haiku
  region: us-east-1
  timeout: 30
  memory: 256
  state:
    provider: dynamodb
    table_prefix: "lambdallm-"
    ttl: 86400

models:
  fast:
    provider: bedrock
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    max_tokens: 1000
    temperature: 0.3
  smart:
    provider: bedrock
    model_id: anthropic.claude-3-sonnet-20240229-v1:0
    max_tokens: 4000
    temperature: 0.7

cost:
  budget:
    daily: 50.00
    monthly: 1000.00
  on_budget_exceeded: downgrade

observability:
  tracing: xray
  metrics: cloudwatch
  log_level: INFO
  log_prompts: false

environments:
  dev:
    model: fast
    cost:
      budget:
        daily: 5.00
  prod:
    model: smart
    cost:
      budget:
        daily: 50.00
'''

TEST_TEMPLATE = '''"""Tests for {name} handler."""

import json
import pytest
from unittest.mock import patch, MagicMock
from lambdallm.core.models import ModelResponse


class MockLambdaContext:
    function_name = "test"
    def get_remaining_time_in_millis(self):
        return 30000


@pytest.fixture
def mock_provider():
    with patch("lambdallm.providers.get_provider") as mock:
        provider = MagicMock()
        provider.invoke.return_value = ModelResponse(
            content="Test response",
            model_id="test",
            tokens_in=10,
            tokens_out=20,
            latency_ms=100.0,
            cost_usd=0.0001,
        )
        mock.return_value = provider
        yield provider


def test_handler_returns_200(mock_provider):
    from handlers.main import lambda_handler

    result = lambda_handler({{"body": json.dumps({{"text": "test"}})}}, MockLambdaContext())
    assert result["statusCode"] == 200
'''


def init_project(name: str, template: str = "basic"):
    """Scaffold a new LambdaLLM project."""
    project_dir = os.path.join(os.getcwd(), name)

    if os.path.exists(project_dir):
        print(f"Error: Directory '{name}' already exists.")
        sys.exit(1)

    template_data = TEMPLATES.get(template, TEMPLATES["basic"])

    print(f"Creating LambdaLLM project: {name}")
    print(f"Template: {template} - {template_data['description']}")
    print()

    # Create directory structure
    dirs = [
        project_dir,
        os.path.join(project_dir, "handlers"),
        os.path.join(project_dir, "prompts"),
        os.path.join(project_dir, "tests"),
        os.path.join(project_dir, "tests", "golden"),
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)

    # Write files
    _write(os.path.join(project_dir, "lambdallm.yaml"), CONFIG_TEMPLATE.format(name=name))
    _write(os.path.join(project_dir, "handlers", "__init__.py"), "")
    _write(os.path.join(project_dir, "handlers", "main.py"), template_data["handler_code"])
    _write(os.path.join(project_dir, "tests", "__init__.py"), "")
    _write(os.path.join(project_dir, "tests", "test_main.py"), TEST_TEMPLATE.format(name=name))
    _write(os.path.join(project_dir, "tests", "golden", ".gitkeep"), "")

    # Write requirements
    _write(os.path.join(project_dir, "requirements.txt"), "substrai-lambdallm[bedrock]>=1.0.0\n")

    # Write README
    readme = f"""# {name}

Built with [LambdaLLM](https://github.com/substrai/lambdallm) by SubstrAI.

## Quick Start

```bash
pip install -r requirements.txt
lambdallm dev
```

## Deploy

```bash
lambdallm deploy --env dev
```

## Test

```bash
lambdallm test
```
"""
    _write(os.path.join(project_dir, "README.md"), readme)

    print(f"  Created {name}/")
    print(f"  Created {name}/lambdallm.yaml")
    print(f"  Created {name}/handlers/main.py ({template} template)")
    print(f"  Created {name}/tests/test_main.py")
    print(f"  Created {name}/requirements.txt")
    print(f"  Created {name}/README.md")
    print()
    print("Done! Next steps:")
    print(f"  cd {name}")
    print(f"  pip install -r requirements.txt")
    print(f"  lambdallm dev")
    print()


def _write(path: str, content: str):
    """Write content to a file."""
    with open(path, "w") as f:
        f.write(content)
