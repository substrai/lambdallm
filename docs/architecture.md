# Architecture Guide

How LambdaLLM works internally, design decisions, and extension points.

---

## Design Philosophy

LambdaLLM is built on 7 core principles:

1. **Convention Over Configuration** - Works with zero config
2. **Inversion of Control** - Framework calls user code
3. **Plugin Architecture** - Extend without modifying source
4. **Declarative Over Imperative** - Users say what, framework does how
5. **Observable by Default** - Everything traced and metered
6. **Infrastructure as Byproduct** - One command deploys everything
7. **Escape Hatches** - Never trap the user

---

## System Architecture

```
+------------------------------------------------------------------+
|                        LambdaLLM Framework                        |
+------------------------------------------------------------------+
|  CLI Layer          |  Runtime Layer       |  Deploy Layer        |
|  - init             |  - @handler          |  - SAM generator     |
|  - dev              |  - Prompt            |  - CDK generator     |
|  - deploy           |  - Chain/Step        |  - Canary deployer   |
|  - test             |  - Agent/Tool        |  - Rollback          |
+---------------------+----------------------+----------------------+
|                       Plugin Layer                                 |
|  Providers | Middleware | State Adapters | Routers | Observers    |
+------------------------------------------------------------------+
|                       AWS Services                                 |
|  Lambda | Bedrock | DynamoDB | API GW | CloudWatch | X-Ray | SQS |
+------------------------------------------------------------------+
```

---

## Package Structure

```
src/lambdallm/
+-- __init__.py              # Public API exports
+-- core/
|   +-- handler.py           # @handler decorator (IoC entry point)
|   +-- context.py           # LambdaLLMContext (user-facing interface)
|   +-- prompt.py            # Prompt template system
|   +-- models.py            # Model enum, ModelConfig, ModelResponse
|   +-- config.py            # YAML config loader
|   +-- streaming.py         # Lambda Response Streaming
|   +-- exceptions.py        # Exception hierarchy
+-- providers/
|   +-- base.py              # BaseProvider (plugin interface)
|   +-- bedrock.py           # AWS Bedrock implementation
+-- middleware/
|   +-- base.py              # Middleware base class
|   +-- logging.py           # Structured logging
|   +-- cost.py              # Cost enforcement
+-- state/
|   +-- session.py           # Session + MemoryStrategy
|   +-- dynamodb.py          # DynamoDB state store
|   +-- memory.py            # In-memory store (dev/test)
|   +-- context_window.py    # Context window manager
|   +-- auto_session.py      # Auto load/save integration
+-- chains/
|   +-- chain.py             # Chain + Step definitions
|   +-- runner.py            # ChainRunner with checkpoint/resume
+-- agents/
|   +-- tool.py              # @Tool decorator + ToolRegistry
|   +-- agent.py             # Agent (ReAct loop)
|   +-- router.py            # Multi-agent router
|   +-- sandbox.py           # Tool sandboxing
|   +-- async_tools.py       # SQS dispatch + Human-in-the-loop
+-- observability/
|   +-- tracer.py            # Distributed tracing (X-Ray)
|   +-- metrics.py           # CloudWatch metrics emitter
|   +-- cost_tracker.py      # Persistent cost tracking
|   +-- router.py            # Cost-aware model router
|   +-- ab_testing.py        # A/B experiment system
|   +-- prompt_analytics.py  # Prompt performance tracking
+-- deploy/
|   +-- generator.py         # SAM/CDK template generation
|   +-- deployer.py          # Deployment orchestrator
|   +-- canary.py            # Canary deployment
+-- testing/
|   +-- mocks.py             # MockProvider, MockLambdaContext
|   +-- golden.py            # Golden dataset runner
+-- cli/
    +-- main.py              # CLI entry point (argparse)
    +-- init.py              # Project scaffolding
    +-- dev.py               # Local dev server
```

---

## Request Lifecycle

When a Lambda invocation hits a @handler-decorated function:

```
1. Lambda invokes handler
2. @handler decorator activates
3. LambdaLLMContext is created (model, config, timeout info)
4. Middleware: before_invoke() runs (logging, cost check, auth)
5. User function executes
   - context.invoke() calls provider
   - Provider formats request for model family (Claude/Titan/Llama)
   - Bedrock API called with retry + exponential backoff
   - Response parsed, cost calculated
   - Metrics recorded
6. Middleware: after_invoke() runs (cost tracking, response logging)
7. Response formatted as Lambda response
8. Metrics flushed to CloudWatch
```

---

## Extension Points

### Adding a New Model Provider

```python
from lambdallm.providers.base import BaseProvider
from lambdallm.core.models import ModelConfig, ModelResponse

class OpenAIProvider(BaseProvider):
    def invoke(self, prompt: str, config: ModelConfig) -> ModelResponse:
        # Call OpenAI API
        response = openai.chat.completions.create(...)
        return ModelResponse(
            content=response.choices[0].message.content,
            model_id=config.model_id,
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            latency_ms=elapsed,
            cost_usd=calculated_cost,
        )

    def supports_streaming(self) -> bool:
        return True
```

### Adding Custom Middleware

```python
from lambdallm.middleware.base import Middleware

class AuthMiddleware(Middleware):
    def before_invoke(self, event, context):
        token = event.get("headers", {}).get("Authorization")
        if not self.validate(token):
            raise UnauthorizedError("Invalid token")
        return event
```

### Adding a Custom State Store

```python
class RedisStateStore:
    def get(self, session_id: str) -> dict:
        return json.loads(self.redis.get(session_id))

    def put(self, session_id: str, data: dict, ttl_seconds: int):
        self.redis.setex(session_id, ttl_seconds, json.dumps(data))
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| No required dependencies | Core is zero-dep | Cold start optimization |
| Module-level client caching | Reuse across invocations | Lambda container reuse |
| Lazy imports everywhere | Import only when needed | Faster cold starts |
| DynamoDB over Redis | Serverless, pay-per-use | Matches Lambda philosophy |
| Hatchling over setuptools | Modern, fast, minimal config | Better developer experience |
| Conventional commits | Structured history | Auto-changelog generation |
| Dataclasses over Pydantic | Zero dependencies | Keep core tiny |

---

## Cold Start Optimization

LambdaLLM adds < 50ms to cold start because:

1. **Zero required dependencies** - Core imports only stdlib
2. **Lazy provider loading** - boto3 imported only on first invoke()
3. **Module-level client caching** - Survives across warm invocations
4. **No heavy frameworks** - No LangChain, no Pydantic in core
5. **Minimal __init__.py** - Only imports lightweight dataclasses

---

## Contributing

See [CONTRIBUTING.md](../CONTRIBUTING.md) for development workflow, commit conventions, and PR process.
