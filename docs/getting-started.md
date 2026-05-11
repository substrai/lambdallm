# Getting Started with LambdaLLM

Get from zero to a deployed GenAI API in under 10 minutes.

## Prerequisites

- Python 3.10+
- AWS account with Bedrock access
- AWS CLI configured (aws configure)

## Step 1: Install LambdaLLM (30 seconds)

**Python (primary):**

```bash
pip install "substrai-lambdallm[bedrock]"
```

**npm (TypeScript SDK - coming soon):**

```bash
npm install substrai-lambdallm
```

Verify installation:

```bash
lambdallm --version
# Output: lambdallm 1.0.0
```

## Step 2: Create a New Project (30 seconds)

```bash
lambdallm init my-genai-api --template basic
cd my-genai-api
```

This creates:

```
my-genai-api/
+-- lambdallm.yaml          # Configuration
+-- handlers/
|   +-- main.py             # Your Lambda handler
+-- tests/
|   +-- test_main.py        # Tests
+-- requirements.txt
+-- README.md
```

## Step 3: Explore the Generated Handler (1 minute)

Open handlers/main.py:

```python
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
    import json
    body = json.loads(event.get("body", "{}"))
    result = summarize.invoke(
        _context=context,
        document=body.get("text", ""),
        max_words=body.get("max_words", 100),
    )
    return {"statusCode": 200, "body": {"result": result, "cost_usd": context.total_cost}}
```

Key concepts:
- **@handler**: Wraps your function with LLM orchestration, error handling, retries
- **Prompt**: Type-safe template with input/output validation
- **Model**: Enum of supported Bedrock models
- **context**: Provides invoke(), cost tracking, timeout awareness

## Step 4: Run Locally (2 minutes)

```bash
lambdallm dev
```

Output:
```
LambdaLLM dev server running on http://localhost:3000
Handler: handlers.main
Press Ctrl+C to stop
```

Test it:

```bash
curl -X POST http://localhost:3000   -H "Content-Type: application/json"   -d '{"text": "LambdaLLM is a serverless framework for LLM orchestration.", "max_words": 20}'
```

## Step 5: Run Tests (1 minute)

```bash
lambdallm test
```

Tests use mock providers by default - no AWS credentials needed for testing.

## Step 6: Configure for Your Needs (2 minutes)

Edit lambdallm.yaml:

```yaml
project:
  name: "my-genai-api"

defaults:
  model: bedrock/claude-3-haiku
  region: us-east-1
  timeout: 30
  memory: 256

cost:
  budget:
    daily: 10.00
  on_budget_exceeded: downgrade
```

## Step 7: Deploy to AWS (3 minutes)

```bash
# Deploy to dev environment
lambdallm deploy --env dev

# Output:
# Deploying my-genai-api to dev...
#   Validated configuration
#   Generated template
#   Deployed stack: my-genai-api-dev
#   Endpoint: https://xxx.execute-api.us-east-1.amazonaws.com/dev/
```

## Next Steps

- [Tutorials](tutorials.md) - Build a chatbot, agent, or RAG system
- [API Reference](api-reference.md) - Full API documentation
- [Examples](https://github.com/substrai/lambdallm/tree/main/examples) - Working code examples

## Need Help?

- [Troubleshooting](troubleshooting.md) - Common issues and fixes
- [GitHub Discussions](https://github.com/substrai/lambdallm/discussions) - Ask questions
- Email: gaurav@substrai.dev
