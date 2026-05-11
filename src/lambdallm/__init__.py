"""LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda.

The first framework purpose-built for Lambda's stateless, cold-start-optimized,
timeout-constrained execution model.

Quick Start:
    from lambdallm import handler, Prompt, Model

    @handler(model=Model.CLAUDE_3_HAIKU)
    def lambda_handler(event, context):
        result = context.invoke("Summarize: {text}", text=event["body"]["text"])
        return {"statusCode": 200, "body": result}

Chains:
    from lambdallm import handler, Chain, Step

    pipeline = Chain(
        name="analysis",
        steps=[
            Step("extract", prompt="Extract entities: {input}"),
            Step("classify", prompt="Classify: {extract.output}"),
        ],
        timeout_strategy="checkpoint",
    )

GitHub: https://github.com/substrai/lambdallm
Docs: https://docs.substrai.dev/lambdallm
"""

__version__ = "0.2.0"
__author__ = "Gaurav Kumar Sinha"
__email__ = "gaurav@substrai.dev"

from lambdallm.core.handler import handler
from lambdallm.core.prompt import Prompt
from lambdallm.core.models import Model, ModelConfig, ModelResponse
from lambdallm.core.config import LambdaLLMConfig
from lambdallm.core.streaming import StreamingResponse, stream_handler
from lambdallm.core.exceptions import (
    LambdaLLMError,
    ModelInvocationError,
    TimeoutError,
    BudgetExceededError,
    ConfigurationError,
)
from lambdallm.state.session import Session, MemoryStrategy
from lambdallm.state.context_window import ContextWindowManager
from lambdallm.chains.chain import Chain, Step
from lambdallm.chains.runner import ChainResult
from lambdallm.middleware.base import Middleware

__all__ = [
    # Core
    "handler",
    "stream_handler",
    "Prompt",
    "Model",
    "ModelConfig",
    "ModelResponse",
    "LambdaLLMConfig",
    "StreamingResponse",
    # Chains
    "Chain",
    "Step",
    "ChainResult",
    # State
    "Session",
    "MemoryStrategy",
    "ContextWindowManager",
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
