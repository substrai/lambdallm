# API Reference

Complete reference for all LambdaLLM public APIs.

---

## Core

### handler(model, timeout_strategy, timeout_buffer, max_retries, fallback_model, middleware, session, router)

Decorator that wraps a Lambda handler with LLM orchestration.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model | Model or str | CLAUDE_3_HAIKU | Default model to use |
| timeout_strategy | str | "fail-fast" | How to handle timeout: fail-fast, truncate, checkpoint |
| timeout_buffer | int | 5 | Seconds to reserve before Lambda timeout |
| max_retries | int | 3 | Retries on transient model errors |
| fallback_model | Model or str | None | Model to use if primary fails |
| middleware | list | None | Middleware instances to apply |
| session | Session | None | Session config for state management |
| router | Router | None | Cost-aware model router |

**Example:**

```python
@handler(model=Model.CLAUDE_3_HAIKU, max_retries=3, fallback_model=Model.TITAN_TEXT_EXPRESS)
def lambda_handler(event, context):
    result = context.invoke("Hello {name}", name="World")
    return {"statusCode": 200, "body": result}
```

---

### Prompt(template, input_schema, output_schema, system_prompt, model, max_tokens, temperature, name, version)

Type-safe prompt template with validation and structured output.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| template | str | required | Prompt template with {variable} placeholders |
| input_schema | dict | None | Expected input types: {"var": type} |
| output_schema | dict | None | Expected output structure (enables JSON parsing) |
| system_prompt | str | None | System prompt prepended to template |
| model | Model | None | Override default model for this prompt |
| max_tokens | int | 1024 | Maximum response tokens |
| temperature | float | 0.7 | Model temperature |
| name | str | None | Prompt name (for analytics tracking) |
| version | str | "1.0.0" | Prompt version (for A/B testing) |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| format(**kwargs) | str | Format template with variables |
| invoke(_context, **kwargs) | str or dict | Invoke LLM and return result |
| to_dict() | dict | Serialize prompt to dictionary |
| from_dict(data) | Prompt | Deserialize from dictionary |
| from_yaml(path) | Prompt | Load from YAML file |

---

### Model (Enum)

Supported model identifiers.

| Value | Model ID |
|-------|----------|
| Model.CLAUDE_3_HAIKU | anthropic.claude-3-haiku-20240307-v1:0 |
| Model.CLAUDE_3_SONNET | anthropic.claude-3-sonnet-20240229-v1:0 |
| Model.CLAUDE_3_5_SONNET | anthropic.claude-3-5-sonnet-20241022-v2:0 |
| Model.CLAUDE_3_OPUS | anthropic.claude-3-opus-20240229-v1:0 |
| Model.TITAN_TEXT_EXPRESS | amazon.titan-text-express-v1 |
| Model.TITAN_TEXT_LITE | amazon.titan-text-lite-v1 |
| Model.LLAMA3_8B | meta.llama3-8b-instruct-v1:0 |
| Model.LLAMA3_70B | meta.llama3-70b-instruct-v1:0 |

---

### LambdaLLMContext

Context object passed to handler functions. Provides model invocation and state access.

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| remaining_time_ms | int | Milliseconds before Lambda timeout |
| should_checkpoint | bool | Whether to save progress now |
| total_cost | float | Cumulative cost in USD |
| session | Session | Loaded session state |
| metrics | Metrics | Metrics collector |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| invoke(prompt, **kwargs) | str | Invoke LLM with prompt template |
| invoke_structured(prompt, schema, **kwargs) | dict | Invoke and parse JSON response |
| get_raw_client(service) | boto3.Client | Escape hatch: raw AWS client |

---

## Chains

### Chain(name, steps, timeout_strategy, max_total_cost)

Declarative multi-step LLM pipeline.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | required | Chain identifier |
| steps | list[Step] | required | Ordered list of steps |
| timeout_strategy | str | "fail-fast" | checkpoint, truncate, or fail-fast |
| max_total_cost | float | None | USD limit for entire chain |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| run(context, **kwargs) | ChainResult | Execute the chain |
| get_step(name) | Step | Get step by name |

---

### Step(name, prompt, func, model, output_schema, condition)

A single step in a chain.

| Parameter | Type | Description |
|-----------|------|-------------|
| name | str | Unique step identifier |
| prompt | str | LLM prompt template (use {var} and {step.output}) |
| func | Callable | Python transform function (alternative to prompt) |
| output_schema | dict | Expected JSON output structure |
| condition | Callable | Skip step if returns False |

---

## State

### Session(store, ttl_hours, memory, max_messages)

Conversation session with persistent state.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| store | str | "dynamodb" | State backend: "dynamodb" or "memory" |
| ttl_hours | int | 24 | Session expiration time |
| memory | MemoryStrategy | SLIDING_WINDOW | How to manage history |
| max_messages | int | 20 | Max messages to keep |

**Methods:**

| Method | Description |
|--------|-------------|
| load(session_id) | Load session from store |
| save() | Persist session (only if modified) |
| add_message(role, content) | Add message to history |
| get_history() | Get messages as list of dicts |
| format_history() | Get messages as formatted string |
| clear() | Remove all messages |

---

## Agents

### Agent(name, system_prompt, tools, max_iterations, timeout_buffer, max_cost_usd)

ReAct-style AI agent with tool usage.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| name | str | required | Agent identifier |
| system_prompt | str | required | Agent instructions |
| tools | list | required | List of @Tool decorated functions |
| max_iterations | int | 10 | Max reasoning loops |
| timeout_buffer | int | 30 | Seconds reserved before timeout |
| max_cost_usd | float | None | Cost limit per execution |

**Methods:**

| Method | Returns | Description |
|--------|---------|-------------|
| run(query, context) | AgentResult | Execute agent reasoning loop |

---

### @Tool(description, name, timeout_seconds)

Decorator to register a function as an agent tool.

```python
@Tool(description="Search documents")
def search(query: str, max_results: int = 5) -> list:
    """Search the knowledge base.

    Args:
        query: Search query string.
        max_results: Max results to return.
    """
    pass
```

Parameters and descriptions are auto-extracted from the function signature and docstring.

---

## Middleware

### Middleware (Base Class)

```python
class MyMiddleware(Middleware):
    def before_invoke(self, event, context):
        # Process before handler
        return event

    def after_invoke(self, event, result, context):
        # Process after handler
        return result

    def on_error(self, event, error, context):
        # Handle errors
        pass
```

### Built-in Middleware

| Class | Description |
|-------|-------------|
| LoggingMiddleware | Structured JSON logging |
| CostTrackingMiddleware | Budget enforcement |

---

## Observability

### Tracer

```python
from lambdallm.observability import Tracer

tracer = Tracer()
with tracer.span("model.invoke") as span:
    span.set_attribute("model_id", "claude-3-haiku")
    # ... do work
```

### MetricsEmitter

```python
from lambdallm.observability import MetricsEmitter

emitter = MetricsEmitter(namespace="MyApp")
emitter.record("custom.metric", 42.0, unit="Count")
emitter.flush()
```

### CostTracker

```python
from lambdallm.observability import CostTracker

tracker = CostTracker(daily_budget=50.0)
tracker.check_budget()  # Raises BudgetExceededError if over
```

---

## Testing

### MockProvider

```python
from lambdallm.testing import MockProvider, mock_model, MockLambdaContext

@mock_model(responses=["Test response"])
def test_my_handler():
    result = my_handler({"body": '{"text": "test"}'}, MockLambdaContext())
    assert result["statusCode"] == 200
```

### GoldenDatasetRunner

```python
from lambdallm.testing import GoldenDatasetRunner

runner = GoldenDatasetRunner()
result = runner.run("tests/golden/cases.json", handler=my_handler)
assert result.pass_rate >= 0.95
```
