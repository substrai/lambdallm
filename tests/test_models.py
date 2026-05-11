"""Tests for model definitions."""

from lambdallm.core.models import Model, ModelConfig, ModelResponse


class TestModel:
    """Test Model enum and config."""

    def test_model_enum_values(self):
        """Model enum should have correct Bedrock model IDs."""
        assert "anthropic" in Model.CLAUDE_3_HAIKU.value
        assert "anthropic" in Model.CLAUDE_3_SONNET.value
        assert "amazon" in Model.TITAN_TEXT_EXPRESS.value
        assert "meta" in Model.LLAMA3_8B.value

    def test_model_config_from_enum(self):
        """ModelConfig should be creatable from Model enum."""
        config = ModelConfig.from_model(Model.CLAUDE_3_HAIKU)
        assert config.model_id == Model.CLAUDE_3_HAIKU.value
        assert config.max_tokens == 1024
        assert config.temperature == 0.7

    def test_model_config_overrides(self):
        """ModelConfig should accept overrides."""
        config = ModelConfig.from_model(
            Model.CLAUDE_3_SONNET, max_tokens=4000, temperature=0.3
        )
        assert config.max_tokens == 4000
        assert config.temperature == 0.3

    def test_model_response_total_tokens(self):
        """ModelResponse should calculate total tokens."""
        response = ModelResponse(
            content="test",
            model_id="test",
            tokens_in=100,
            tokens_out=50,
            latency_ms=200.0,
        )
        assert response.total_tokens == 150
