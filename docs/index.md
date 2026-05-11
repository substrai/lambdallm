# LambdaLLM

**Serverless-native LLM orchestration framework for AWS Lambda.**

<div style="text-align: center; margin: 2rem 0;">
  <a href="getting-started/" style="padding: 12px 24px; background: #3f51b5; color: white; text-decoration: none; border-radius: 4px; margin: 0 8px;">Get Started</a>
  <a href="https://github.com/substrai/lambdallm" style="padding: 12px 24px; background: #333; color: white; text-decoration: none; border-radius: 4px; margin: 0 8px;">GitHub</a>
  <a href="https://pypi.org/project/substrai-lambdallm/" style="padding: 12px 24px; background: #006dad; color: white; text-decoration: none; border-radius: 4px; margin: 0 8px;">PyPI</a>
  <a href="https://www.npmjs.com/package/substrai-lambdallm" style="padding: 12px 24px; background: #cb3837; color: white; text-decoration: none; border-radius: 4px; margin: 0 8px;">npm</a>
</div>

## The Problem

Existing LLM frameworks (LangChain, LlamaIndex) assume long-running servers. They **break on Lambda**:

- :material-timer-alert: **Cold starts**: 500MB+ dependency trees add seconds
- :material-memory: **Stateless**: No conversation memory between invocations
- :material-clock-alert: **15-min timeout**: Long agent loops crash
- :material-package-variant: **250MB limit**: LangChain alone exceeds this

## The Solution

```python
from lambdallm import handler, Prompt, Model

summarize = Prompt(
    template="Summarize in {max_words} words:\n\n{document}",
    output_schema={"summary": str, "key_points": list}
)

@handler(model=Model.CLAUDE_3_HAIKU)
def lambda_handler(event, context):
    return summarize.invoke(
        _context=context,
        document=event["body"]["text"],
        max_words=100
    )
```

## Install

=== "Python (pip)"

    ```bash
    pip install "substrai-lambdallm[bedrock]"
    ```

=== "npm"

    ```bash
    npm install substrai-lambdallm
    ```

[![PyPI](https://badge.fury.io/py/substrai-lambdallm.svg)](https://pypi.org/project/substrai-lambdallm/)
[![npm](https://badge.fury.io/js/substrai-lambdallm.svg)](https://www.npmjs.com/package/substrai-lambdallm)

## TypeScript Quick Start

```typescript
import { handler, Model, Chain, Step } from 'substrai-lambdallm';

export const lambdaHandler = handler(
  { model: Model.CLAUDE_3_HAIKU },
  async (event, context) => {
    const result = await context.invoke('Summarize: {text}', { text: event.body.text });
    return { statusCode: 200, body: { result, cost: context.totalCost } };
  }
);
```

## Key Features

| Feature | Description |
|---------|-------------|
| **< 5MB package** | vs 400MB+ for LangChain |
| **Cold-start optimized** | Lazy imports, connection pooling |
| **DynamoDB state** | Conversation memory that persists |
| **Multi-step chains** | Checkpoint/resume on timeout |
| **AI Agents** | ReAct loop with tool sandboxing |
| **Cost-aware routing** | Auto-select cheapest model |
| **One-command deploy** | `lambdallm deploy` |
| **A/B testing** | Compare prompt versions |
| **Full observability** | X-Ray + CloudWatch built-in |

## Quick Start

```bash
lambdallm init my-app --template basic
cd my-app
lambdallm dev
```

Then test:

```bash
curl -X POST http://localhost:3000 -d '{"text": "Hello world"}'
```

---

Built by [SubstrAI](https://github.com/substrai) — Open-source GenAI frameworks for serverless infrastructure.

**Author:** Gaurav Kumar Sinha ([gaurav@substrai.dev](mailto:gaurav@substrai.dev))
