"""LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda.

The first framework purpose-built for Lambda's stateless, cold-start-optimized,
timeout-constrained execution model.

Quick Start:
    from lambdallm import handler, Prompt, Model

    @handler(model=Model.CLAUDE_3_HAIKU)
    def lambda_handler(event, context):
        result = context.invoke("Summarize: {text}", text=event["body"]["text"])
        return {"statusCode": 200, "body": result}

GitHub: https://github.com/substrai/lambdallm
Docs: https://docs.substrai.dev/lambdallm
"""

__version__ = "0.1.0"
__author__ = "Gaurav Kumar Sinha"
__email__ = "gaurav@substrai.dev"

from lambdallm.core.handler import handler
from lambdallm.core.prompt import Prompt
from lambdallm.core.models import Model, ModelConfig, ModelResponse
from lambdallm.core.config import LambdaLLMConfig
from lambdallm.core.exceptions import (
    LambdaLLMError,
    ModelInvocationError,
    TimeoutError,
    BudgetExceededError,
    ConfigurationError,
)
from lambdallm.state.session import Session, MemoryStrategy
from lambdallm.middleware.base import Middleware

__all__ = [
    # Core
    "handler",
    "Prompt",
    "Model",
    "ModelConfig",
    "ModelResponse",
    "LambdaLLMConfig",
    # State
    "Session",
    "MemoryStrategy",
    # Middleware
    "Middleware",
    # Exceptions
    "LambdaLLMError",
    "ModelInvocationError",
    "TimeoutError",
    "BudgetExceededError",
    "ConfigurationError",
    # Meta
    "__version__",
]
