# Contributing to LambdaLLM

Thank you for your interest in contributing to LambdaLLM! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/lambdallm.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[dev]"`
5. Create a branch: `git checkout -b feat/your-feature`

## Development Workflow

```bash
# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/

# Format code
ruff format src/ tests/
```

## Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `ci:` CI/CD changes
- `refactor:` Code refactoring
- `chore:` Maintenance

## Pull Request Process

1. Ensure all tests pass
2. Update documentation if needed
3. Add tests for new features
4. Keep PRs focused — one feature per PR
5. Reference any related issues

## Code Style

- Python 3.10+ type hints
- Docstrings on all public functions/classes
- Max line length: 100 characters
- Use `ruff` for linting and formatting

## Architecture Principles

When contributing, keep these framework principles in mind:

1. **Convention over configuration** — sensible defaults everywhere
2. **< 5MB package size** — no heavy dependencies in core
3. **Lambda-first** — every feature must work within Lambda constraints
4. **Observable by default** — log and trace without user configuration
5. **Escape hatches** — never trap the user

## Questions?

Open a [Discussion](https://github.com/substrai/lambdallm/discussions) or reach out at contact@substrai.dev.
