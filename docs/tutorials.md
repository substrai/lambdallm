# Tutorials

Hands-on guides for building real applications with LambdaLLM.

---

## Tutorial 1: Build a Document Summarization API

**Time:** 5 minutes | **Difficulty:** Beginner

### What You Will Build

A REST API that accepts documents and returns structured summaries with key points.

### Step 1: Initialize

```bash
lambdallm init summarizer --template basic
cd summarizer
```

### Step 2: Define Your Prompt

Edit handlers/main.py:

```python
from lambdallm import handler, Prompt, Model

summarize = Prompt(
    name="document-summarizer",
    template="""You are an expert summarizer. Summarize the following document
in exactly {max_words} words or fewer. Extract the key points.

Document:
{document}

Respond in JSON format with fields: summary, key_points (list of strings)""",
    input_schema={"document": str, "max_words": int},
    output_schema={"summary": str, "key_points": list},
)

@handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="truncate")
def lambda_handler(event, context):
    import json
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    result = summarize.invoke(
        _context=context,
        document=body.get("text", ""),
        max_words=body.get("max_words", 100),
    )

    return {
        "statusCode": 200,
        "body": {
            "result": result,
            "model": "claude-3-haiku",
            "cost_usd": context.total_cost,
        },
    }
```

### Step 3: Test Locally

```bash
lambdallm dev
# In another terminal:
curl -X POST http://localhost:3000 -d '{"text": "Your long document here...", "max_words": 50}'
```

### Step 4: Deploy

```bash
lambdallm deploy --env dev
```

---

## Tutorial 2: Build a Multi-Turn Chatbot

**Time:** 10 minutes | **Difficulty:** Intermediate

### What You Will Build

A chatbot that remembers conversation history across Lambda invocations using DynamoDB.

### Step 1: Initialize

```bash
lambdallm init chatbot --template chat
cd chatbot
```

### Step 2: The Chat Handler

```python
from lambdallm import handler, Model
from lambdallm.state import Session, MemoryStrategy

@handler(
    model=Model.CLAUDE_3_SONNET,
    session=Session(
        store="dynamodb",
        ttl_hours=24,
        max_messages=20,
    ),
)
def lambda_handler(event, context):
    import json
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    message = body.get("message", "")
    session_id = body.get("session_id", "default")

    # Load conversation history
    session = context.session
    if session:
        session.load(session_id)
        session.add_message("user", message)

    # Build prompt with history
    history = session.format_history() if session else message
    response = context.invoke(
        "You are a helpful assistant.\n\nConversation:\n{history}\n\nassistant:",
        history=history,
    )

    # Save to DynamoDB
    if session:
        session.add_message("assistant", response)
        session.save()

    return {
        "statusCode": 200,
        "body": {"reply": response, "session_id": session_id, "messages": session.message_count if session else 0},
    }
```

### Key Concepts

- **Session**: Auto-persists to DynamoDB between invocations
- **Memory Strategy**: Sliding window keeps last 20 messages (prevents token overflow)
- **session_id**: Client sends this to maintain conversation continuity

---

## Tutorial 3: Build an AI Agent with Tools

**Time:** 15 minutes | **Difficulty:** Advanced

### What You Will Build

A research agent that can search a knowledge base and perform calculations.

### Step 1: Initialize

```bash
lambdallm init researcher --template agent
cd researcher
```

### Step 2: Define Tools

```python
from lambdallm import handler, Model
from lambdallm.agents import Agent, Tool

@Tool(description="Search the knowledge base for relevant documents")
def search_kb(query: str, max_results: int = 3) -> list:
    """Search for documents matching the query.

    Args:
        query: The search query string.
        max_results: Maximum results to return.
    """
    # Replace with your actual search (OpenSearch, Bedrock KB, etc.)
    return [{"title": f"Result for {query}", "content": f"Information about {query}..."}]

@Tool(description="Calculate a mathematical expression")
def calculate(expression: str) -> float:
    """Safely evaluate a math expression.

    Args:
        expression: Math expression like '100 * 1.15'
    """
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        raise ValueError(f"Invalid expression: {expression}")
    return eval(expression)

# Create the agent
researcher = Agent(
    name="research-analyst",
    system_prompt="You are a research analyst. Use tools to find information and calculate.",
    tools=[search_kb, calculate],
    max_iterations=5,
    timeout_buffer=30,
    max_cost_usd=0.10,
)

@handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    import json
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    result = researcher.run(query=body.get("question", ""), context=context)

    return {
        "statusCode": 200,
        "body": {
            "answer": result.answer,
            "iterations": result.total_iterations,
            "tool_calls": result.total_tool_calls,
            "cost_usd": result.total_cost_usd,
        },
    }
```

### Key Concepts

- **@Tool decorator**: Auto-generates LLM tool schemas from function signatures
- **Agent**: ReAct loop (Reason-Act-Observe) with timeout awareness
- **timeout_buffer**: Reserves 30s before Lambda timeout for cleanup
- **max_cost_usd**: Stops agent if cost exceeds limit

---

## Tutorial 4: Build a Multi-Step Analysis Chain

**Time:** 10 minutes | **Difficulty:** Intermediate

### What You Will Build

A document analysis pipeline that extracts entities, classifies them, and generates a summary.

```python
from lambdallm import handler, Chain, Step, Model

analysis = Chain(
    name="document-analysis",
    steps=[
        Step("extract", prompt="Extract all key entities (people, orgs, dates) from:\n\n{input}"),
        Step("classify", prompt="Classify each entity by type and importance:\n\n{extract.output}"),
        Step("summarize", prompt="Create executive summary from entities and classifications:\n\n{classify.output}"),
    ],
    timeout_strategy="checkpoint",
    max_total_cost=0.50,
)

@handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    import json
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    result = analysis.run(context=context, input=body.get("document", ""))

    return {
        "statusCode": 200 if result.status == "completed" else 202,
        "body": {
            "status": result.status,
            "result": result.final_output,
            "steps_completed": result.completed_steps,
            "cost_usd": result.total_cost_usd,
        },
    }
```

### Key Concepts

- **Chain**: Declarative multi-step pipeline
- **{step_name.output}**: Reference previous step results
- **timeout_strategy="checkpoint"**: Saves progress if Lambda is about to timeout
- **max_total_cost**: Auto-stops chain if budget exceeded
