# LambdaLLM

**Serverless-native LLM orchestration framework for AWS Lambda.**

> Built by [SubstrAI](https://github.com/substrai) — Open-source GenAI frameworks for serverless infrastructure.

[![PyPI version](https://badge.fury.io/py/substrai-lambdallm.svg)](https://pypi.org/project/substrai-lambdallm/)
[![Tests](https://github.com/substrai/lambdallm/actions/workflows/ci.yml/badge.svg)](https://github.com/substrai/lambdallm/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation](https://img.shields.io/badge/docs-substrai.github.io-blue)](https://substrai.github.io/lambdallm)
[![codecov](https://codecov.io/gh/substrai/lambdallm/branch/main/graph/badge.svg)](https://codecov.io/gh/substrai/lambdallm)
[![npm version](https://badge.fury.io/js/substrai-lambdallm.svg)](https://www.npmjs.com/package/substrai-lambdallm)

## The Problem

Existing LLM frameworks (LangChain, LlamaIndex) assume long-running servers. They break on Lambda:
- Cold starts: 500MB+ dependency trees add seconds
- Stateless: No conversation memory between invocations
- 15-min timeout: Long agent loops crash
- 250MB limit: LangChain alone exceeds this

## The Solution

LambdaLLM is purpose-built for Lambda's constraints:

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

## Features

- **< 5MB** package size (vs 400MB+ for LangChain)
- **Cold-start optimized** — lazy imports, connection pooling
- **DynamoDB-native state** — conversation memory that survives stateless execution
- **Cost-aware routing** — auto-select cheapest model that meets quality threshold
- **Multi-step chains** — declarative pipelines with checkpoint/resume on timeout
- **AI Agents** — ReAct-style agents with tool sandboxing and timeout awareness
- **One-command deploy** — `lambdallm deploy` generates all AWS infrastructure
- **Timeout handling** — checkpoint/resume for long chains
- **A/B testing** — route traffic between prompt versions, compare metrics
- **Full observability** — X-Ray tracing, CloudWatch metrics, cost tracking built-in

## Installation

### Python (primary)

```bash
pip install substrai-lambdallm
```

With AWS Bedrock support (recommended):

```bash
pip install "substrai-lambdallm[bedrock]"
```

With all optional dependencies:

```bash
pip install "substrai-lambdallm[all]"
```

### npm

```bash
npm install substrai-lambdallm
```

## Quick Start

```bash
# Initialize a new project
lambdallm init my-project --template basic
cd my-project

# Start local development server
lambdallm dev

# Test your handler
curl -X POST http://localhost:3000 -d '{"text": "Hello world"}'

# Deploy to AWS
lambdallm deploy --env dev
```

## Available Templates

```bash
lambdallm init my-app --template basic   # Simple LLM handler
lambdallm init my-app --template chat    # Multi-turn chat with memory
lambdallm init my-app --template agent   # AI agent with tools
lambdallm init my-app --template rag     # Retrieval-augmented generation
```

## Core Concepts

### Handlers
```python
from lambdallm import handler, Model

@handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="checkpoint")
def lambda_handler(event, context):
    result = context.invoke("Summarize: {text}", text=event["body"]["text"])
    return {"statusCode": 200, "body": result}
```

### Chains
```python
from lambdallm import Chain, Step

pipeline = Chain(
    name="analysis",
    steps=[
        Step("extract", prompt="Extract entities from: {input}"),
        Step("classify", prompt="Classify: {extract.output}"),
        Step("summarize", prompt="Summarize: {classify.output}"),
    ],
    timeout_strategy="checkpoint",
)
```

### Agents
```python
from lambdallm.agents import Agent, Tool

@Tool(description="Search the knowledge base")
def search(query: str, max_results: int = 5) -> list:
    # your implementation
    pass

agent = Agent(
    name="researcher",
    system_prompt="You are a research assistant.",
    tools=[search],
    max_iterations=5,
    timeout_buffer=30,
)
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `lambdallm init` | Scaffold a new project |
| `lambdallm dev` | Start local development server |
| `lambdallm deploy` | Deploy to AWS (SAM/CDK) |
| `lambdallm test` | Run tests |
| `lambdallm cost` | Show cost summary and forecast |
| `lambdallm status` | Check deployment status |
| `lambdallm rollback` | Rollback to previous version |
| `lambdallm eject` | Export raw SAM/CDK templates |
| `lambdallm logs` | Tail CloudWatch logs |
| `lambdallm metrics` | Show key metrics |

## Documentation

- [**Full Documentation**](https://substrai.github.io/lambdallm) — Getting started, tutorials, API reference
- [Getting Started](https://substrai.github.io/lambdallm/getting-started/)
- [Tutorials](https://substrai.github.io/lambdallm/tutorials/)
- [API Reference](https://substrai.github.io/lambdallm/api-reference/)
- [Architecture Guide](https://substrai.github.io/lambdallm/architecture/)
- [Migration Guide](https://substrai.github.io/lambdallm/migration-guide/)
- [Examples](https://github.com/substrai/lambdallm/tree/main/examples)
- [Contributing](https://github.com/substrai/lambdallm/blob/main/CONTRIBUTING.md)
- [Changelog](https://github.com/substrai/lambdallm/blob/main/CHANGELOG.md)

## License

MIT — see [LICENSE](LICENSE)

## Author

**Gaurav Kumar Sinha** — Founder, [SubstrAI](https://github.com/substrai)

- Email: gaurav@substrai.dev
- GitHub: [@substrai](https://github.com/substrai)
