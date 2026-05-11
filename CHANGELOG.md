# Changelog

All notable changes to LambdaLLM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-10

### Added
- `@handler` decorator with IoC pattern for Lambda functions
- `Prompt` class with type-safe templates and structured output
- `Model` enum with Bedrock model IDs (Claude, Titan, Llama)
- `BedrockProvider` with lazy initialization and cost tracking
- Middleware system: `LoggingMiddleware`, `CostTrackingMiddleware`
- Retry with exponential backoff and fallback model support
- Timeout awareness with checkpoint/truncate/fail-fast strategies
- GitHub Actions CI/CD (test + publish)
- Examples: basic summarizer, chat with memory
- Full test suite with mocked providers

### Architecture
- Core package size: < 50KB (no dependencies in core)
- Plugin-based provider system (BaseProvider interface)
- Middleware pipeline (before/after hooks)
- Convention over configuration (works with zero config)
- Escape hatch: `context.get_raw_client()` for direct boto3 access
