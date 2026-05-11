"""LambdaLLM Context - passed to handler functions.

Provides the interface for users to interact with LLM models,
manage state, and access framework utilities within their handler.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, Union

from lambdallm.core.models import ModelConfig, ModelResponse
from lambdallm.core.exceptions import ModelInvocationError, TimeoutError

logger = logging.getLogger("lambdallm")


@dataclass
class Metrics:
    """Simple metrics collector that emits to CloudWatch."""

    _data: dict = field(default_factory=dict)

    def record(self, name: str, value: float) -> None:
        """Record a metric value."""
        self._data[name] = value

    def get(self, name: str) -> Optional[float]:
        """Get a recorded metric."""
        return self._data.get(name)

    def all(self) -> dict:
        """Get all recorded metrics."""
        return self._data.copy()


class LambdaLLMContext:
    """Context object passed to handler functions.

    Provides:
    - Model invocation (invoke, chat)
    - Session/state access
    - Metrics recording
    - Timeout awareness
    - Raw client access (escape hatch)
    """

    def __init__(
        self,
        model: Optional[ModelConfig],
        timeout_strategy: str,
        timeout_buffer: int,
        max_retries: int,
        fallback_model: Optional[ModelConfig],
        lambda_context: Any,
        middleware: list,
        session_config: Any = None,
        router: Any = None,
    ):
        self.model = model
        self.timeout_strategy = timeout_strategy
        self.timeout_buffer = timeout_buffer
        self.max_retries = max_retries
        self.fallback_model = fallback_model
        self.lambda_context = lambda_context
        self.middleware = middleware
        self.session_config = session_config
        self.router = router
        self.metrics = Metrics()
        self.checkpoint: Optional[dict] = None
        self._provider = None
        self._session = None
        self._invocation_count = 0
        self._total_cost = 0.0

    @property
    def remaining_time_ms(self) -> int:
        """Milliseconds remaining before Lambda timeout."""
        if self.lambda_context and hasattr(self.lambda_context, "get_remaining_time_in_millis"):
            return self.lambda_context.get_remaining_time_in_millis()
        return 900_000  # Default 15 min for local dev

    @property
    def should_checkpoint(self) -> bool:
        """Whether we're approaching timeout and should save progress."""
        return self.remaining_time_ms < (self.timeout_buffer * 1000)

    @property
    def total_cost(self) -> float:
        """Total cost of all model invocations in this request."""
        return self._total_cost

    @property
    def session(self) -> Optional[Any]:
        """Lazy-loaded session state."""
        if self._session is None and self.session_config:
            # Session loading will be implemented in state module
            pass
        return self._session

    def invoke(self, prompt: str, **kwargs) -> str:
        """Invoke the LLM with a prompt string.

        Args:
            prompt: The prompt template string. Use {key} for variable substitution.
            **kwargs: Variables to substitute into the prompt.

        Returns:
            The model's response text.

        Raises:
            ModelInvocationError: If the model call fails after retries.
            TimeoutError: If approaching Lambda timeout.
        """
        # Check timeout
        if self.should_checkpoint:
            raise TimeoutError("Approaching Lambda timeout")

        # Format prompt with variables
        formatted_prompt = prompt.format(**kwargs) if kwargs else prompt

        # Select model (use router if available)
        model_config = self._select_model(formatted_prompt)

        # Invoke with retries
        response = self._invoke_with_retry(formatted_prompt, model_config)

        # Track costs
        self._invocation_count += 1
        self._total_cost += response.cost_usd
        self.metrics.record("model.invocations", self._invocation_count)
        self.metrics.record("model.total_cost_usd", self._total_cost)
        self.metrics.record("model.tokens_in", response.tokens_in)
        self.metrics.record("model.tokens_out", response.tokens_out)
        self.metrics.record("model.latency_ms", response.latency_ms)

        return response.content

    def invoke_structured(self, prompt: str, output_schema: dict, **kwargs) -> dict:
        """Invoke the LLM and parse response into a structured format.

        Args:
            prompt: The prompt template string.
            output_schema: Expected output schema (dict of field_name: type).
            **kwargs: Variables to substitute into the prompt.

        Returns:
            Parsed response as a dictionary matching the schema.
        """
        import json as json_module

        schema_instruction = f"\n\nRespond in JSON format with these fields: {output_schema}"
        full_prompt = prompt.format(**kwargs) + schema_instruction if kwargs else prompt + schema_instruction

        response_text = self.invoke(full_prompt)

        # Try to parse JSON from response
        try:
            # Handle markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            return json_module.loads(response_text.strip())
        except json_module.JSONDecodeError:
            # Retry once with explicit JSON instruction
            retry_prompt = f"{full_prompt}\n\nIMPORTANT: Return ONLY valid JSON, no other text."
            response_text = self.invoke(retry_prompt)
            return json_module.loads(response_text.strip())

    def get_raw_client(self, service: str = "bedrock-runtime"):
        """Escape hatch: get the raw boto3 client for direct access.

        Use this when the framework's abstraction doesn't fit your needs.
        """
        import boto3

        region = self.model.region if self.model and self.model.region else "us-east-1"
        return boto3.client(service, region_name=region)

    def _select_model(self, prompt: str) -> ModelConfig:
        """Select the best model based on router rules or default."""
        if self.router:
            return self.router.select(prompt, self)
        return self.model

    def _invoke_with_retry(self, prompt: str, model_config: ModelConfig) -> ModelResponse:
        """Invoke model with exponential backoff retry."""
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self._call_provider(prompt, model_config)
            except ModelInvocationError as e:
                last_error = e
                if not e.retryable:
                    raise
                # Exponential backoff
                wait_time = min(2**attempt * 0.5, 10)
                logger.warning(
                    f"Model invocation failed (attempt {attempt + 1}/{self.max_retries}), "
                    f"retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)

        # All retries exhausted - try fallback model
        if self.fallback_model:
            logger.info(f"Falling back to model: {self.fallback_model.model_id}")
            try:
                return self._call_provider(prompt, self.fallback_model)
            except ModelInvocationError:
                pass

        raise ModelInvocationError(
            f"All {self.max_retries} retries exhausted. Last error: {last_error}",
            retryable=False,
        )

    def _call_provider(self, prompt: str, model_config: ModelConfig) -> ModelResponse:
        """Call the model provider. Lazy-loads the provider on first use."""
        if self._provider is None:
            from lambdallm.providers import get_provider

            self._provider = get_provider(model_config)

        return self._provider.invoke(prompt, model_config)
