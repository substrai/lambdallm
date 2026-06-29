# ADR 004: Checkpoint/Resume for Timeout Handling

**Status:** Accepted  
**Date:** 2026-05-28  
**Author:** Gaurav Kumar Sinha

## Context

Lambda has a hard 15-minute execution timeout. Multi-step LLM chains (e.g., a
5-step analysis pipeline where each step calls Bedrock) can exceed this limit,
especially with large context windows or complex agent loops.

Current behavior when timeout hits: Lambda kills the process, all intermediate
results are lost, the user gets a 503 error. This is unacceptable for production
workloads that may have accumulated $0.50+ in LLM costs before the timeout.

We need a mechanism to:
- Detect approaching timeouts before they hit
- Save intermediate state (completed steps, accumulated context)
- Resume from the last checkpoint on the next invocation
- Make this transparent to the developer

### Alternatives Considered

1. **Step Functions orchestration**: External state machine — adds complexity, cost, and latency between steps.
2. **Increase timeout to 15 min**: Only delays the problem, doesn't solve it for long chains.
3. **Break into multiple Lambdas**: Requires manual orchestration, loses simplicity of single-handler model.
4. **Background processing with SQS**: Async only — can't return results to synchronous API calls.

## Decision

Implement **transparent checkpoint/resume** within the handler using DynamoDB state:

```python
@handler(model=Model.CLAUDE_3_SONNET, timeout_strategy="checkpoint")
def lambda_handler(event, context):
    # Chain automatically checkpoints between steps
    chain = Chain(steps=[
        Step("extract", prompt="Extract entities: {input}"),
        Step("analyze", prompt="Analyze: {extract.output}"),
        Step("summarize", prompt="Summarize: {analyze.output}"),
    ])
    return chain.run(input=event["body"]["text"])
```

### How it works:

1. **Timeout detection**: Monitor `context.get_remaining_time_in_millis()` — checkpoint when <30s remain
2. **State serialization**: After each step completes, serialize the chain state to DynamoDB
3. **Resume detection**: On next invocation with same `request_id`, load checkpoint and skip completed steps
4. **Idempotency**: Each step has a deterministic ID — re-running a checkpointed step returns cached result
5. **Client signaling**: Return HTTP 202 with `X-LambdaLLM-Checkpoint: step-2-of-5` header

### Timeout budget allocation:
```
Total: 900s (15 min)
├── Step execution: 840s (reserve 60s buffer)
├── Checkpoint write: 20s (DynamoDB write + serialization)
└── Safety margin: 40s (network variance)
```

## Consequences

### Positive
- **No lost work**: Intermediate results preserved even on timeout
- **Transparent to developers**: `timeout_strategy="checkpoint"` is the only config needed
- **Cost-efficient**: Don't re-run expensive LLM calls for completed steps
- **Composable**: Works with chains, agents, and custom multi-step logic
- **Observable**: Checkpoint events emitted as metrics/traces

### Negative
- **Increased latency**: Checkpoint writes add ~50ms per step
- **State size limits**: DynamoDB 400KB item limit caps checkpoint size
- **Complexity**: Resume logic must handle partially-completed steps carefully
- **Client awareness**: Callers must handle 202 responses and poll/retry
- **Cold start on resume**: Second invocation pays cold start cost again

### Mitigations
- Checkpoint writes are async (fire-and-forget with retry)
- Large intermediate results stored in S3, DynamoDB holds reference
- Steps are atomic — either fully complete or not started
- SDK client handles 202 + retry automatically
- Provisioned concurrency eliminates cold start for resumed invocations
