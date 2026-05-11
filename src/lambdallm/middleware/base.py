"""Base middleware interface for LambdaLLM.

Middleware follows the before/after pattern:
1. before_invoke: Process request before it reaches the handler
2. after_invoke: Process response before it's returned to the caller

Users extend this class to add custom cross-cutting concerns.
"""

from abc import ABC
from typing import Any


class Middleware(ABC):
    """Base class for LambdaLLM middleware.

    Example:
        class AuthMiddleware(Middleware):
            def before_invoke(self, event, context):
                token = event.get("headers", {}).get("Authorization")
                if not self.validate_token(token):
                    raise UnauthorizedError("Invalid token")
                return event

            def after_invoke(self, event, result, context):
                return result
    """

    def before_invoke(self, event: dict, context: Any) -> dict:
        """Process the event before the handler executes.

        Args:
            event: The Lambda event dict.
            context: The LambdaLLMContext.

        Returns:
            The (possibly modified) event dict.
        """
        return event

    def after_invoke(self, event: dict, result: Any, context: Any) -> Any:
        """Process the result after the handler executes.

        Args:
            event: The original Lambda event dict.
            result: The handler's return value.
            context: The LambdaLLMContext.

        Returns:
            The (possibly modified) result.
        """
        return result

    def on_error(self, event: dict, error: Exception, context: Any) -> None:
        """Called when an error occurs during handler execution.

        Args:
            event: The original Lambda event dict.
            error: The exception that was raised.
            context: The LambdaLLMContext.
        """
        pass
