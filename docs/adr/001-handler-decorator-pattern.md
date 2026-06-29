# ADR 001: Handler Decorator Pattern

**Status:** Accepted  
**Date:** 2026-05-25  
**Author:** Gaurav Kumar Sinha

## Context

AWS Lambda requires a handler function with a specific signature `(event, context)`.
Existing LLM frameworks (LangChain, LlamaIndex) don't account for this — they assume
long-running server processes. Teams repeatedly write boilerplate to bridge framework
abstractions to Lambda's execution model.

We needed an approach that:
- Preserves Lambda's native handler signature
- Adds LLM-specific capabilities (model selection, cost tracking, observability)
- Requires zero configuration for basic usage
- Allows progressive complexity (opt-in to advanced features)

### Alternatives Considered

1. **Class-based handler** (`class MyHandler(LambdaLLMHandler)`): Too verbose for simple cases, requires understanding inheritance hierarchy.
2. **Configuration file approach** (YAML/JSON defines behavior): Loses IDE support, type safety, and makes debugging harder.
3. **Middleware-only pattern** (wrap any function): Doesn't provide enough structure for consistent observability.

## Decision

Use a **decorator pattern** that wraps the standard Lambda handler signature:

```python
from lambdallm import handler, Model

@handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="checkpoint")
def lambda_handler(event, context):
    result = context.invoke("Summarize: {text}", text=event["body"]["text"])
    return {"statusCode": 200, "body": result}
```

The decorator:
- Intercepts the Lambda invocation lifecycle
- Injects an enhanced `context` object with LLM methods (`.invoke()`, `.stream()`)
- Manages model connections, cost tracking, and observability transparently
- Preserves the standard `(event, context)` signature for Lambda compatibility

## Consequences

### Positive
- **Zero learning curve**: Developers familiar with Lambda immediately understand the pattern
- **Progressive disclosure**: Simple cases require only `@handler()`, complex cases add parameters
- **Type safety**: Full IDE autocomplete and type checking on the enhanced context
- **Testability**: Functions remain pure — mock the context for unit tests
- **Framework-agnostic**: The handler is still a valid Lambda handler without the decorator

### Negative
- **Magic behavior**: The decorator injects capabilities that aren't visible from the signature
- **Debugging complexity**: Stack traces include decorator internals
- **Single model per handler**: Each decorated handler binds to one model (use chains for multi-model)

### Mitigations
- Comprehensive docstrings explain what the decorator adds
- `LambdaLLMContext` is a well-documented class with explicit methods
- Error messages include "LambdaLLM:" prefix for easy identification in logs
