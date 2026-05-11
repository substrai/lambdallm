# Migration Guide

How to migrate from existing tools to LambdaLLM.

---

## Migrating from Raw Boto3 + Lambda

### Before (Raw Boto3)

```python
import json
import boto3

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def lambda_handler(event, context):
    try:
        body = json.loads(event["body"])
        prompt = f"Summarize: {body['text']}"

        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": prompt}]
            }),
            contentType="application/json",
        )

        result = json.loads(response["body"].read())
        content = result["content"][0]["text"]

        return {"statusCode": 200, "body": json.dumps({"result": content})}

    except Exception as e:
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
```

### After (LambdaLLM)

```python
from lambdallm import handler, Prompt, Model

summarize = Prompt(template="Summarize: {text}")

@handler(model=Model.CLAUDE_3_HAIKU, max_retries=3)
def lambda_handler(event, context):
    import json
    body = json.loads(event["body"])
    result = summarize.invoke(_context=context, text=body["text"])
    return {"statusCode": 200, "body": {"result": result}}
```

### What You Gain

| Concern | Before (manual) | After (LambdaLLM) |
|---------|-----------------|-------------------|
| Error handling | Manual try/except | Automatic with retries |
| Model formatting | Know each model API format | Framework handles it |
| Cost tracking | None | Automatic per-request |
| Timeout handling | Hope it finishes | Checkpoint/resume |
| Retries | Manual implementation | Built-in exponential backoff |
| Observability | Manual logging | Structured logs + X-Ray + metrics |

---

## Migrating from LangChain on Lambda

### Before (LangChain - problematic on Lambda)

```python
# WARNING: This causes 500MB+ package, 5s+ cold starts
from langchain_aws import ChatBedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = ChatBedrock(model_id="anthropic.claude-3-haiku-20240307-v1:0")
prompt = PromptTemplate(template="Summarize: {text}", input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)

def lambda_handler(event, context):
    import json
    body = json.loads(event["body"])
    result = chain.run(text=body["text"])
    return {"statusCode": 200, "body": json.dumps({"result": result})}
```

### After (LambdaLLM)

```python
from lambdallm import handler, Prompt, Model

summarize = Prompt(template="Summarize: {text}")

@handler(model=Model.CLAUDE_3_HAIKU)
def lambda_handler(event, context):
    import json
    body = json.loads(event["body"])
    result = summarize.invoke(_context=context, text=body["text"])
    return {"statusCode": 200, "body": {"result": result}}
```

### Comparison

| Metric | LangChain on Lambda | LambdaLLM |
|--------|--------------------:|----------:|
| Package size | ~400MB | < 5MB |
| Cold start | 3-8 seconds | < 200ms |
| Dependencies | 50+ packages | 0 (core) |
| Memory usage | 256MB+ | < 128MB |
| State management | In-memory (lost) | DynamoDB (persisted) |
| Cost tracking | None | Built-in |

---

## Migrating Multi-Step Chains

### LangChain SequentialChain -> LambdaLLM Chain

```python
# Before (LangChain)
from langchain.chains import SequentialChain
chain = SequentialChain(chains=[chain1, chain2, chain3])
result = chain.run(input="...")
# Problem: crashes if Lambda times out mid-chain

# After (LambdaLLM)
from lambdallm import Chain, Step
pipeline = Chain(
    name="analysis",
    steps=[
        Step("extract", prompt="Extract: {input}"),
        Step("classify", prompt="Classify: {extract.output}"),
        Step("summarize", prompt="Summarize: {classify.output}"),
    ],
    timeout_strategy="checkpoint",  # Saves progress on timeout!
)
result = pipeline.run(context=context, input="...")
# If Lambda times out: saves progress, resumes on next invocation
```

---

## Migration Checklist

1. [ ] Install: pip install substrai-lambdallm[bedrock]
2. [ ] Replace boto3 bedrock calls with @handler + context.invoke()
3. [ ] Replace manual prompts with Prompt() templates
4. [ ] Replace in-memory state with Session(store="dynamodb")
5. [ ] Replace manual error handling with framework retries
6. [ ] Add lambdallm.yaml configuration
7. [ ] Run tests: lambdallm test
8. [ ] Deploy: lambdallm deploy --env dev
9. [ ] Remove LangChain dependencies (if applicable)
10. [ ] Verify cold start improvement

---

## Keeping the Escape Hatch

If LambdaLLM does not cover a specific use case, you can always drop down to raw boto3:

```python
@handler(model=Model.CLAUDE_3_HAIKU)
def lambda_handler(event, context):
    # Use framework for most things
    summary = context.invoke("Summarize: {text}", text=event["body"]["text"])

    # Drop to raw boto3 for edge cases
    raw_client = context.get_raw_client("bedrock-runtime")
    custom_response = raw_client.invoke_model(
        modelId="custom-model-arn",
        body=custom_payload,
    )

    return {"statusCode": 200, "body": {"summary": summary}}
```
