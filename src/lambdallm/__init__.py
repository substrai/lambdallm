"""LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda.

The first framework purpose-built for Lambda's stateless, cold-start-optimized,
timeout-constrained execution model.

Quick Start:
    from lambdallm import handler, Prompt, Model

    @handler(model=Model.CLAUDE_3_HAIKU)
    def lambda_handler(event, context):
        result = context.invoke("Summarize: {text}", text=event["body"]["text"])
        return {"statusCode": 200, "body": result}

Agents:
    from lambdallm import handler, Model
    from lambdallm.agents import Agent, Tool

    @Tool(description="Search documents")
    def search(query: str) -> list:
        pass

    agent = Agent(name="researcher", system_prompt="...", tools=[search])

    @handler(model=Model.CLAUDE_3_SONNET)
    def lambda_handler(event, context):
        result = agent.run(query=event["body"]["question"], context=context)
        return {"statusCode": 200, "body": result.answer}

GitHub: https://github.com/substrai/lambdallm
Docs: https://docs.substrai.dev/lambdallm
"""

__version__ = "0.3.0"
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
