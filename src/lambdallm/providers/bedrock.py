"""AWS Bedrock provider for LambdaLLM.

Optimized for Lambda execution:
- Lazy client initialization (reduces cold start)
- Connection reuse across invocations
- Cost calculation per model
"""

import json
import time
import logging
from typing import Optional

from lambdallm.core.models import ModelConfig, ModelResponse
from lambdallm.core.exceptions import ModelInvocationError, ProviderError
from lambdallm.providers.base import BaseProvider

logger = logging.getLogger("lambdallm")

# Cost per 1K tokens (input/output) as of 2024
MODEL_COSTS = {
    "anthropic.claude-3-haiku-20240307-v1:0": {"input": 0.00025, "output": 0.00125},
    "anthropic.claude-3-sonnet-20240229-v1:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-5-sonnet-20241022-v2:0": {"input": 0.003, "output": 0.015},
    "anthropic.claude-3-opus-20240229-v1:0": {"input": 0.015, "output": 0.075},
    "amazon.titan-text-express-v1": {"input": 0.0002, "output": 0.0006},
    "amazon.titan-text-lite-v1": {"input": 0.00015, "output": 0.0002},
    "meta.llama3-8b-instruct-v1:0": {"input": 0.0003, "output": 0.0006},
    "meta.llama3-70b-instruct-v1:0": {"input": 0.00265, "output": 0.0035},
}

# Lazy-loaded client (persists across Lambda invocations via container reuse)
_client: Optional[object] = None


def _get_client(region: str = "us-east-1"):
    """Get or create the Bedrock runtime client.

    Uses module-level caching for Lambda container reuse optimization.
    """
    global _client
    if _client is None:
        try:
            import boto3

            _client = boto3.client("bedrock-runtime", region_name=region)
        except ImportError:
            raise ProviderError(
                "boto3 is required for Bedrock provider. "
                "Install with: pip install lambdallm[bedrock]"
            )
    return _client


class BedrockProvider(BaseProvider):
    """AWS Bedrock model provider.

    Supports Claude, Titan, and Llama models via the Bedrock runtime API.
    Optimized for Lambda with lazy initialization and connection reuse.
    """

    def invoke(self, prompt: str, config: ModelConfig) -> ModelResponse:
        """Invoke a Bedrock model.

        Automatically formats the request based on the model family
        (Anthropic, Amazon, Meta each have different request formats).
        """
        client = _get_client(config.region or "us-east-1")
        model_id = config.model_id

        # Build request body based on model family
        body = self._build_request_body(prompt, config)

        start_time = time.time()

        try:
            response = client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
        except client.exceptions.ThrottlingException as e:
            raise ModelInvocationError(f"Bedrock throttled: {e}", retryable=True)
        except client.exceptions.ModelTimeoutException as e:
            raise ModelInvocationError(f"Bedrock timeout: {e}", retryable=True)
        except client.exceptions.ValidationException as e:
            raise ModelInvocationError(f"Bedrock validation error: {e}", retryable=False)
        except Exception as e:
            raise ModelInvocationError(f"Bedrock invocation failed: {e}", retryable=True)

        latency_ms = (time.time() - start_time) * 1000

        # Parse response based on model family
        response_body = json.loads(response["body"].read())
        content, tokens_in, tokens_out = self._parse_response(response_body, model_id)

        # Calculate cost
        cost = self._calculate_cost(model_id, tokens_in, tokens_out)

        return ModelResponse(
            content=content,
            model_id=model_id,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=cost,
            raw_response=response_body,
        )

    def supports_streaming(self) -> bool:
        return True

    def _build_request_body(self, prompt: str, config: ModelConfig) -> dict:
        """Build the request body based on model family."""
        model_id = config.model_id

        if "anthropic" in model_id:
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
                "messages": [{"role": "user", "content": prompt}],
            }
        elif "amazon.titan" in model_id:
            return {
                "inputText": prompt,
                "textGenerationConfig": {
                    "maxTokenCount": config.max_tokens,
                    "temperature": config.temperature,
                    "topP": config.top_p,
                },
            }
        elif "meta.llama" in model_id:
            return {
                "prompt": prompt,
                "max_gen_len": config.max_tokens,
                "temperature": config.temperature,
                "top_p": config.top_p,
            }
        else:
            # Generic fallback - Anthropic format
            return {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": config.max_tokens,
                "temperature": config.temperature,
                "messages": [{"role": "user", "content": prompt}],
            }

    def _parse_response(self, response_body: dict, model_id: str) -> tuple[str, int, int]:
        """Parse response body based on model family. Returns (content, tokens_in, tokens_out)."""
        if "anthropic" in model_id:
            content = response_body["content"][0]["text"]
            tokens_in = response_body.get("usage", {}).get("input_tokens", 0)
            tokens_out = response_body.get("usage", {}).get("output_tokens", 0)
        elif "amazon.titan" in model_id:
            content = response_body["results"][0]["outputText"]
            tokens_in = response_body.get("inputTextTokenCount", 0)
            tokens_out = response_body.get("results", [{}])[0].get("tokenCount", 0)
        elif "meta.llama" in model_id:
            content = response_body["generation"]
            tokens_in = response_body.get("prompt_token_count", 0)
            tokens_out = response_body.get("generation_token_count", 0)
        else:
            content = str(response_body)
            tokens_in = 0
            tokens_out = 0

        return content, tokens_in, tokens_out

    def _calculate_cost(self, model_id: str, tokens_in: int, tokens_out: int) -> float:
        """Calculate cost in USD based on token usage."""
        costs = MODEL_COSTS.get(model_id, {"input": 0.001, "output": 0.002})
        input_cost = (tokens_in / 1000) * costs["input"]
        output_cost = (tokens_out / 1000) * costs["output"]
        return input_cost + output_cost
