# Contributing

Thank you for your interest in contributing to LambdaLLM!

For full contributing guidelines, see [CONTRIBUTING.md on GitHub](https://github.com/substrai/lambdallm/blob/main/CONTRIBUTING.md).

## Quick Start for Contributors

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/lambdallm.git
cd lambdallm

# Set up dev environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run linter
ruff check src/ tests/
```

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `test:` Tests
- `ci:` CI/CD
- `refactor:` Refactoring

## Architecture Principles

When contributing, keep these in mind:

1. **Zero dependencies in core** — keeps cold starts fast
2. **Lambda-first** — every feature must work within Lambda constraints
3. **Convention over configuration** — sensible defaults everywhere
4. **Observable by default** — log and trace without user config
5. **Escape hatches** — never trap the user
