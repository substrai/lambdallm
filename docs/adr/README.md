# Architecture Decision Records

This directory contains Architecture Decision Records (ADRs) for LambdaLLM.

ADRs document the key architectural decisions made during the development of LambdaLLM,
including the context, decision, and consequences of each choice.

## Index

| ADR | Title | Status | Date |
|-----|-------|--------|------|
| [001](001-handler-decorator-pattern.md) | Handler Decorator Pattern | Accepted | 2026-05-25 |
| [002](002-dynamodb-state-backend.md) | DynamoDB State Backend | Accepted | 2026-05-26 |
| [003](003-lazy-imports-cold-start.md) | Lazy Imports for Cold Start Optimization | Accepted | 2026-05-27 |
| [004](004-checkpoint-resume.md) | Checkpoint/Resume for Timeout Handling | Accepted | 2026-05-28 |

## Format

Each ADR follows this structure:
- **Status**: Proposed, Accepted, Deprecated, Superseded
- **Context**: What is the issue that we're seeing that motivates this decision?
- **Decision**: What is the change that we're proposing and/or doing?
- **Consequences**: What becomes easier or more difficult as a result?
