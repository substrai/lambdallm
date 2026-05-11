"""LambdaLLM exception hierarchy."""


class LambdaLLMError(Exception):
    """Base exception for all LambdaLLM errors."""

    pass


class ModelInvocationError(LambdaLLMError):
    """Raised when a model invocation fails."""

    def __init__(self, message: str, retryable: bool = True):
        super().__init__(message)
        self.retryable = retryable


class TimeoutError(LambdaLLMError):
    """Raised when approaching Lambda timeout."""

    pass


class BudgetExceededError(LambdaLLMError):
    """Raised when cost budget is exceeded."""

    pass


class ConfigurationError(LambdaLLMError):
    """Raised when framework configuration is invalid."""

    pass


class ProviderError(LambdaLLMError):
    """Raised when a model provider encounters an error."""

    pass


class StateError(LambdaLLMError):
    """Raised when state management fails."""

    pass
