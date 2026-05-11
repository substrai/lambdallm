# Troubleshooting

Common errors, debugging tips, and FAQ.

---

## Common Errors

### ImportError: No module named 'boto3'

**Cause:** boto3 is an optional dependency (keeps core lightweight).

**Fix:**
```bash
pip install "substrai-lambdallm[bedrock]"
```

---

### ModelInvocationError: Bedrock throttled

**Cause:** Too many requests to Bedrock API.

**Fix:**
- The framework auto-retries with exponential backoff (default: 3 retries)
- Increase max_retries: `@handler(max_retries=5)`
- Add a fallback model: `@handler(fallback_model=Model.TITAN_TEXT_EXPRESS)`
- Request a Bedrock quota increase in AWS console

---

### BudgetExceededError: Daily budget exceeded

**Cause:** Your daily cost limit was reached.

**Fix:**
- Increase budget in lambdallm.yaml: `cost.budget.daily: 100.00`
- Change action: `cost.on_budget_exceeded: downgrade` (uses cheaper model)
- Check costs: `lambdallm cost --report daily`

---

### TimeoutError: Approaching Lambda timeout

**Cause:** Handler is about to exceed Lambda timeout.

**Fix:**
- Increase timeout: `defaults.timeout: 60` in lambdallm.yaml
- Use checkpoint: `@handler(timeout_strategy="checkpoint")`
- Use faster model (Haiku instead of Sonnet)
- Reduce max_tokens

---

### ConfigurationError: Template variables not defined in input_schema

**Cause:** Prompt template has {variables} not in input_schema.

**Fix:**
```python
# All template variables must be in schema
Prompt(
    template="Hello {name}, summarize {text}",
    input_schema={"text": str, "name": str}  # Both variables declared
)
```

---

### Cold start is slow (> 1 second)

**Fix:**
- Move heavy imports inside functions
- Increase Lambda memory (more memory = more CPU)
- Use provisioned concurrency for latency-sensitive endpoints

---

### Session not persisting

**Fix:**
1. Deploy infrastructure: `lambdallm deploy --env dev` (creates DynamoDB tables)
2. Ensure Lambda has DynamoDB permissions
3. Check table_prefix in lambdallm.yaml

---

### Agent stuck (max_iterations reached)

**Fix:**
- Increase: `Agent(..., max_iterations=10)`
- Improve system_prompt
- Check tool outputs are useful

---

## Debugging Tips

### Enable verbose logging

```python
import logging
logging.getLogger("lambdallm").setLevel(logging.DEBUG)
```

### Test without AWS credentials

```python
from lambdallm.testing import mock_model, MockLambdaContext

@mock_model(responses=["Mock response"])
def test_handler():
    result = my_handler({"body": '{"text": "test"}'}, MockLambdaContext())
    assert result["statusCode"] == 200
```

### Check cost per request

```python
@handler(model=Model.CLAUDE_3_HAIKU)
def lambda_handler(event, context):
    result = context.invoke("Hello")
    print(f"Cost: ${context.total_cost:.6f}")
    return {"statusCode": 200, "body": result}
```

---

## FAQ

**Q: Does it work with models outside Bedrock?**
A: Yes. Implement BaseProvider for any LLM API.

**Q: Can I use it without Lambda?**
A: Yes. `lambdallm dev` runs locally. Works anywhere Python runs.

**Q: LambdaLLM vs Bedrock Agents?**
A: Bedrock Agents = managed (less control). LambdaLLM = framework (full control, open-source).

**Q: Minimum Lambda memory?**
A: 128MB works. 256MB recommended for agents/chains.

**Q: TypeScript support?**
A: Not yet. On the roadmap.

**Q: How to handle secrets?**
A: Use AWS Secrets Manager. Never put secrets in lambdallm.yaml. Bedrock uses IAM (no keys needed).

---

## Getting Help

- Issues: github.com/substrai/lambdallm/issues
- Discussions: github.com/substrai/lambdallm/discussions
- Email: gaurav@substrai.dev
