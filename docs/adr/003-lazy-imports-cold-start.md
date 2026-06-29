# ADR 003: Lazy Imports for Cold Start Optimization

**Status:** Accepted  
**Date:** 2026-05-27  
**Author:** Gaurav Kumar Sinha

## Context

Lambda cold starts are the critical performance bottleneck. Every millisecond of
import time directly impacts user-perceived latency. The Python import system
eagerly loads all modules at `import` time, including transitive dependencies.

Measured import times for common LLM frameworks:
- LangChain: ~2,400ms (imports torch, numpy, pydantic, dozens of providers)
- LlamaIndex: ~1,800ms (similar dependency tree)
- LambdaLLM target: **<50ms** for core imports

We need to minimize import-time work while maintaining a clean developer API.

### Alternatives Considered

1. **Vendoring/bundling**: Copy dependencies inline — reduces I/O but doesn't reduce parse time.
2. **Compiled extensions (Cython)**: Faster execution but complex build pipeline, platform-specific.
3. **Module-level `__getattr__`**: Python 3.7+ feature for lazy module attributes — clean but limited.
4. **Import everything eagerly**: Simple but unacceptable cold start impact.

## Decision

Use a **multi-level lazy import strategy**:

### Level 1: Module `__getattr__` for optional components
```python
# src/lambdallm/__init__.py
def __getattr__(name):
    if name == "DynamoDBStateStore":
        from lambdallm.state.store import DynamoDBStateStore
        return DynamoDBStateStore
    raise AttributeError(f"module 'lambdallm' has no attribute {name!r}")
```

### Level 2: Deferred provider initialization
```python
# Bedrock client created on first .invoke(), not at import
class LambdaLLMContext:
    @cached_property
    def _client(self):
        import boto3
        return boto3.client("bedrock-runtime")
```

### Level 3: Conditional imports guarded by feature flags
```python
# Heavy observability imports only when tracing is enabled
if config.tracing_enabled:
    from lambdallm.observability import XRayTracer
```

## Consequences

### Positive
- **<50ms cold start contribution**: Core imports measured at 12-35ms
- **Sub-5MB package**: No heavy transitive dependencies pulled in eagerly
- **Feature-gated costs**: Users only pay import time for features they use
- **Compatible with Lambda Layers**: Heavy deps (boto3) already in Lambda runtime

### Negative
- **Delayed error discovery**: Import errors surface at runtime, not at deploy time
- **IDE limitations**: Some lazy imports don't appear in autocomplete until accessed
- **Complexity**: Developers must understand which imports are lazy vs eager
- **Testing overhead**: Tests must exercise lazy paths to catch import errors

### Mitigations
- `lambdallm validate` CLI command checks all imports eagerly (run in CI)
- Type stubs (`__init__.pyi`) provide full IDE autocomplete
- Integration tests import every public symbol to catch broken lazy paths
- Documentation clearly marks which components require optional dependencies
