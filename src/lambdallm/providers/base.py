"""Base provider interface for LambdaLLM.

All model providers must implement this interface.
This enables the plugin architecture - swap providers without changing user code.
"""

from abc import ABC, abstractmethod
from lambdallm.core.models import ModelConfig, ModelResponse


class BaseProvider(ABC):
    """Abstract base class for model providers.

    Implement this to add support for new LLM providers.

    Example:
        class MyProvider(BaseProvider):
            def invoke(self, prompt, config):
                response = my_api.call(prompt)
                return ModelResponse(content=response.text, ...)

            def supports_streaming(self):
                return True
    """

    def __init__(self, config: ModelConfig):
        self.config = config

    @abstractmethod
    def invoke(self, prompt: str, config: ModelConfig) -> ModelResponse:
        """Invoke the model with a prompt.

        Args:
            prompt: The formatted prompt string.
            config: Model configuration (tokens, temperature, etc.)

        Returns:
            ModelResponse with content, token counts, and cost.
        """
        pass

    @abstractmethod
    def supports_streaming(self) -> bool:
        """Whether this provider supports streaming responses."""
        pass

    def invoke_streaming(self, prompt: str, config: ModelConfig):
        """Stream response tokens. Override if supports_streaming() is True."""
        raise NotImplementedError("Streaming not supported by this provider")

    @property
    def name(self) -> str:
        """Provider name for logging and metrics."""
        return self.__class__.__name__
