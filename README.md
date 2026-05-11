# LambdaLLM

**Serverless-native LLM orchestration framework for AWS Lambda.**

> Built by [SubstrAI](https://github.com/substrai) — Open-source GenAI frameworks for serverless infrastructure.

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
        document=event["body"]["text"],
        max_words=100
    )
```

## Features

- **< 5MB** package size (vs 400MB+ for LangChain)
- **Cold-start optimized** — lazy imports, connection pooling
- **DynamoDB-native state** — conversation memory that survives stateless execution
- **Cost-aware routing** — auto-select cheapest model that meets quality threshold
- **One-command deploy** — `lambdallm deploy` generates all AWS infrastructure
- **Timeout handling** — checkpoint/resume for long chains

## Installation

```bash
pip install lambdallm[bedrock]
```

## Quick Start

```bash
lambdallm init my-project
cd my-project
lambdallm dev
```

## Documentation

- [Getting Started](https://docs.substrai.dev/lambdallm/getting-started)
- [API Reference](https://docs.substrai.dev/lambdallm/api)
- [Examples](https://github.com/substrai/lambdallm/tree/main/examples)

## License

MIT — see [LICENSE](LICENSE)

## Author

**Gaurav Kumar Sinha** — Founder, [SubstrAI](https://github.com/substrai)
