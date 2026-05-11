"""Chain execution engine for LambdaLLM.

Handles step-by-step execution with:
- Variable resolution between steps
- Timeout awareness and checkpointing
- Cost tracking across the chain
- Conditional step execution
- Error recovery and retries
"""

import re
import time
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.chains.chain import Chain, Step
from lambdallm.core.exceptions import TimeoutError, LambdaLLMError

logger = logging.getLogger("lambdallm")


@dataclass
class StepResult:
    """Result from a single chain step."""

    name: str
    output: Any
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    skipped: bool = False
    error: Optional[str] = None


@dataclass
class ChainResult:
    """Result from a complete chain execution."""

    chain_name: str
    steps: list[StepResult] = field(default_factory=list)
    status: str = "completed"  # completed | checkpointed | failed | truncated
    checkpoint: Optional[dict] = None
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    final_output: Any = None

    @property
    def completed_steps(self) -> int:
        return sum(1 for s in self.steps if not s.skipped and not s.error)

    @property
    def outputs(self) -> dict:
        """All step outputs as a dict."""
        return {s.name: s.output for s in self.steps if s.output is not None}


class ChainRunner:
    """Executes a chain with timeout awareness and checkpointing.

    The runner resolves variables between steps, manages state,
    and handles Lambda timeout constraints.
    """

    def __init__(self, chain: Chain, context: Optional[Any] = None):
        self.chain = chain
        self.context = context
        self._step_outputs: dict[str, Any] = {}
        self._start_time: float = 0
        self._total_cost: float = 0

    def execute(self, checkpoint: Optional[dict] = None, **kwargs) -> ChainResult:
        """Execute the chain from start or resume from checkpoint.

        Args:
            checkpoint: Previous checkpoint to resume from.
            **kwargs: Input variables for the chain.

        Returns:
            ChainResult with all step outputs and metadata.
        """
        self._start_time = time.time()
        self._step_outputs = {"input": kwargs.get("input", "")}
        self._step_outputs.update(kwargs)

        result = ChainResult(chain_name=self.chain.name)

        # Determine starting step (resume from checkpoint if provided)
        start_index = 0
        if checkpoint:
            start_index = checkpoint.get("next_step_index", 0)
            self._step_outputs = checkpoint.get("step_outputs", self._step_outputs)
            logger.info(f"Resuming chain '{self.chain.name}' from step {start_index}")

        for i, step in enumerate(self.chain.steps[start_index:], start=start_index):
            # Check timeout before each step
            if self._should_checkpoint():
                result.status = "checkpointed"
                result.checkpoint = {
                    "next_step_index": i,
                    "step_outputs": self._step_outputs,
                    "chain_name": self.chain.name,
                }
                logger.info(f"Chain '{self.chain.name}' checkpointed at step {i} ({step.name})")
                break

            # Check cost limit
            if self.chain.max_total_cost and self._total_cost >= self.chain.max_total_cost:
                result.status = "truncated"
                logger.warning(f"Chain '{self.chain.name}' truncated: cost limit reached")
                break

            # Execute step
            step_result = self._execute_step(step, i)
            result.steps.append(step_result)

            if step_result.error:
                result.status = "failed"
                logger.error(f"Chain '{self.chain.name}' failed at step '{step.name}': {step_result.error}")
                break

            # Store output for next steps
            if not step_result.skipped:
                self._step_outputs[step.name] = step_result.output
                # Also store as {step_name}.output for template resolution
                self._step_outputs[f"{step.name}.output"] = step_result.output

            self._total_cost += step_result.cost_usd

        # Finalize result
        result.total_cost_usd = self._total_cost
        result.total_latency_ms = (time.time() - self._start_time) * 1000

        if result.steps and not result.steps[-1].error:
            result.final_output = result.steps[-1].output

        if result.status == "completed" or (result.status != "checkpointed" and result.status != "failed"):
            if not any(s.error for s in result.steps):
                result.status = "completed"

        return result

    def _execute_step(self, step: Step, index: int) -> StepResult:
        """Execute a single step."""
        start = time.time()

        # Check condition
        if step.condition and not step.condition(self._step_outputs):
            logger.debug(f"Step '{step.name}' skipped (condition not met)")
            return StepResult(name=step.name, output=None, skipped=True)

        try:
            if step.is_llm_step:
                return self._execute_llm_step(step, start)
            else:
                return self._execute_transform_step(step, start)
        except Exception as e:
            latency = (time.time() - start) * 1000
            logger.error(f"Step '{step.name}' failed: {e}")
            return StepResult(name=step.name, output=None, latency_ms=latency, error=str(e))

    def _execute_llm_step(self, step: Step, start_time: float) -> StepResult:
        """Execute an LLM prompt step."""
        # Resolve variables in the prompt template
        prompt = self._resolve_variables(step.prompt)

        if self.context is None:
            raise LambdaLLMError("Chain LLM steps require a context. Pass context to chain.run()")

        # Invoke model
        if step.output_schema:
            output = self.context.invoke_structured(prompt, step.output_schema)
        else:
            output = self.context.invoke(prompt)

        latency = (time.time() - start_time) * 1000

        # Get metrics from context
        tokens_in = self.context.metrics.get("model.tokens_in") or 0
        tokens_out = self.context.metrics.get("model.tokens_out") or 0
        cost = self.context.metrics.get("model.total_cost_usd") or 0

        return StepResult(
            name=step.name,
            output=output,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost_usd=cost - self._total_cost,  # Delta cost for this step
            latency_ms=latency,
        )

    def _execute_transform_step(self, step: Step, start_time: float) -> StepResult:
        """Execute a Python transform step."""
        # Pass all previous outputs to the transform function
        output = step.func(self._step_outputs)
        latency = (time.time() - start_time) * 1000

        return StepResult(name=step.name, output=output, latency_ms=latency)

    def _resolve_variables(self, template: str) -> str:
        """Resolve {variable} and {step_name.output} references in a template."""
        def replacer(match):
            key = match.group(1)
            if key in self._step_outputs:
                value = self._step_outputs[key]
                return str(value) if not isinstance(value, str) else value
            return match.group(0)  # Leave unresolved variables as-is

        pattern = r"\{([a-zA-Z_][a-zA-Z0-9_.]*)\}"
        return re.sub(pattern, replacer, template)

    def _should_checkpoint(self) -> bool:
        """Check if we should checkpoint due to approaching timeout."""
        if self.chain.timeout_strategy != "checkpoint":
            return False

        if self.context and hasattr(self.context, "should_checkpoint"):
            return self.context.should_checkpoint

        return False
