# ADR 002: DynamoDB State Backend

**Status:** Accepted  
**Date:** 2026-05-26  
**Author:** Gaurav Kumar Sinha

## Context

Lambda functions are stateless — each invocation starts fresh. LLM applications
require conversational state (message history, session context, accumulated costs).
We need a state backend that:

- Survives across Lambda invocations
- Handles concurrent access safely (multiple Lambda instances)
- Supports TTL for automatic session expiration
- Has sub-10ms latency for cold-start-sensitive workloads
- Requires zero infrastructure management
- Works within Lambda's execution environment (no long-lived connections)

### Alternatives Considered

1. **Redis/ElastiCache**: Fast but requires VPC configuration, connection management, and monthly cost regardless of usage.
2. **S3**: Cheap storage but high latency (50-200ms), no atomic operations, eventual consistency.
3. **RDS/Aurora**: Relational overhead, connection pooling complexity in Lambda, cold start penalty.
4. **Lambda Extensions with local state**: Not durable across invocations, lost on cold start.

## Decision

Use **DynamoDB** as the primary production state backend with a **JSON file fallback** for local development.

```python
# Production: DynamoDB (zero config with IAM role)
store = DynamoDBStateStore(table_name="lambdallm-sessions")

# Development: Local JSON file
store = InMemoryStateStore()  # or JsonFileStateStore("./state.json")
```

Key design choices:
- **Single-table design**: One DynamoDB table stores all session types (messages, metadata, costs)
- **Composite keys**: `PK=SESSION#{session_id}`, `SK=MSG#{timestamp}` for efficient range queries
- **TTL attribute**: DynamoDB native TTL handles session expiration without cron jobs
- **Conditional writes**: `ConditionExpression` prevents lost updates from concurrent invocations
- **Interface abstraction**: `StateStore` protocol allows swapping backends without code changes

## Consequences

### Positive
- **Serverless-native**: Pay-per-request pricing, no idle costs, auto-scaling
- **Sub-5ms latency**: DynamoDB on-demand reads are consistently fast
- **Atomic operations**: Conditional writes prevent race conditions
- **Zero infrastructure**: Table auto-created on first use (or via `lambdallm deploy`)
- **TTL built-in**: Sessions automatically expire without cleanup jobs
- **Dev/prod parity**: Same interface, different backend — easy local testing

### Negative
- **25KB item limit**: Long conversation histories must be paginated or summarized
- **AWS lock-in**: DynamoDB is AWS-specific (mitigated by interface abstraction)
- **Cost at scale**: High-throughput applications may prefer provisioned capacity
- **Query limitations**: No complex queries — must know the session ID

### Mitigations
- Context window management auto-summarizes when approaching 25KB limit
- `StateStore` interface enables future backends (Redis, PostgreSQL)
- Monitoring emits item size metrics to alert before hitting limits
