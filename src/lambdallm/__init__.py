"""LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda.

The first framework purpose-built for Lambda's stateless, cold-start-optimized,
timeout-constrained execution model.

GitHub: https://github.com/substrai/lambdallm
Docs: https://docs.substrai.dev/lambdallm
Author: Gaurav Kumar Sinha <gaurav@substrai.dev>
"""

__version__ = "2.2.0"
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
