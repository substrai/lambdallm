"""Tests for configuration loader."""

import pytest
from lambdallm.core.config import LambdaLLMConfig, DEFAULT_CONFIG


class TestConfig:
    """Test configuration loading and defaults."""

    def test_default_config_loads(self):
        """Config should load with all defaults when no file exists."""
        config = LambdaLLMConfig.load(path="/nonexistent/path.yaml")
        assert config.project.name == "my-lambdallm-app"
        assert config.defaults.model == "bedrock/claude-3-haiku"
        assert config.defaults.region == "us-east-1"
        assert config.defaults.timeout == 30
        assert config.defaults.memory == 256

    def test_default_state_config(self):
        """State config should default to DynamoDB."""
        config = LambdaLLMConfig.load(path="/nonexistent/path.yaml")
        assert config.defaults.state.provider == "dynamodb"
        assert config.defaults.state.table_prefix == "lambdallm-"
        assert config.defaults.state.ttl == 86400

    def test_default_cost_config(self):
        """Cost config should have sensible budget defaults."""
        config = LambdaLLMConfig.load(path="/nonexistent/path.yaml")
        assert config.cost.budget.daily == 50.0
        assert config.cost.budget.monthly == 1000.0
        assert config.cost.on_budget_exceeded == "downgrade"

    def test_default_observability_config(self):
        """Observability should be enabled by default."""
        config = LambdaLLMConfig.load(path="/nonexistent/path.yaml")
        assert config.observability.tracing == "xray"
        assert config.observability.metrics == "cloudwatch"
        assert config.observability.log_prompts is False

    def test_models_config(self):
        """Models should include fast/smart/powerful presets."""
        config = LambdaLLMConfig.load(path="/nonexistent/path.yaml")
        assert "fast" in config.models
        assert "smart" in config.models
        assert "powerful" in config.models
        assert "haiku" in config.models["fast"]["model_id"]
        assert "sonnet" in config.models["smart"]["model_id"]
