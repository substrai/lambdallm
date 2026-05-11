"""Model providers for LambdaLLM.

Providers are the bridge between the framework and LLM services.
The plugin architecture allows swapping providers without changing user code.
"""

from lambdallm.core.models import ModelConfig
from lambdallm.providers.base import BaseProvider


def get_provider(model_config: ModelConfig) -> BaseProvider:
    """Get the appropriate provider for a model config.

    Uses convention: Bedrock model IDs contain dots (anthropic.claude-3...)
    """
    model_id = model_config.model_id

    # Bedrock models have format: provider.model-name-version
    if "." in model_id:
        from lambdallm.providers.bedrock import BedrockProvider
        return BedrockProvider(model_config)

    # Default to Bedrock
    from lambdallm.providers.bedrock import BedrockProvider
    return BedrockProvider(model_config)


__all__ = ["get_provider", "BaseProvider"]
