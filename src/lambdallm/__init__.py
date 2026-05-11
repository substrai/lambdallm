"""LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda."""

__version__ = "0.1.0"
__author__ = "Gaurav Kumar Sinha"
__email__ = "gaurav@substrai.dev"

from lambdallm.core.handler import handler
from lambdallm.core.prompt import Prompt
from lambdallm.core.models import Model

__all__ = ["handler", "Prompt", "Model", "__version__"]
