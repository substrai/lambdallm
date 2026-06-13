"""Multi-model chain with conditional routing.

Routes to different models per step based on complexity, cost, and latency
requirements. Enables chains that use cheaper/faster models for simple steps
and more capable models for complex reasoning steps.

Features:
- Per-step model selection based on configurable strategies
- Conditional branching within chains
- Cost/latency/quality optimization
- Model selection strategies: lowest-cost, lowest-latency, highest-quality, threshold-based
- Fallback model support when primary selection is unavailable
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class SelectionStrategy(Enum):
    """Model selection strategy for routing decisions."""

    LOWEST_COST = "lowest_cost"
    LOWEST_LATENCY = "lowest_latency"
    HIGHEST_QUALITY = "highest_quality"
    THRESHOLD = "threshold"
    ROUND_ROBIN = "round_robin"
    CUSTOM = "custom"


class ComplexityLevel(Enum):
    """Task complexity classification for model routing."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ModelProfile:
    """Profile describing a model's capabilities and characteristics.

    Attributes:
        model_id: Unique model identifier (e.g., "gpt-4", "claude-3-haiku").
        cost_per_1k_tokens: Cost in USD per 1000 tokens.
        avg_latency_ms: Average response latency in milliseconds.
        quality_score: Quality rating from 0.0 to 1.0.
        max_tokens: Maximum token limit for this model.
        supports_json: Whether model supports structured JSON output.
        supports_streaming: Whether model supports streaming responses.
        complexity_ceiling: Maximum complexity level this model handles well.
        metadata: Additional model metadata.
    """

    model_id: str
    cost_per_1k_tokens: float = 0.01
    avg_latency_ms: float = 500.0
    quality_score: float = 0.8
    max_tokens: int = 4096
    supports_json: bool = True
    supports_streaming: bool = True
    complexity_ceiling: ComplexityLevel = ComplexityLevel.HIGH
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def cost_efficiency(self) -> float:
        """Quality per unit cost (higher is better)."""
        if self.cost_per_1k_tokens == 0:
            return float("inf")
        return self.quality_score / self.cost_per_1k_tokens

    @property
    def speed_quality_ratio(self) -> float:
        """Quality per unit latency (higher is better)."""
        if self.avg_latency_ms == 0:
            return float("inf")
        return self.quality_score / (self.avg_latency_ms / 1000.0)


@dataclass
class RoutingCondition:
    """Condition that determines when a branch should be taken.

    Attributes:
        name: Human-readable condition name.
        predicate: Function that evaluates context and returns bool.
        priority: Higher priority conditions are evaluated first.
    """

    name: str
    predicate: Callable[[Dict[str, Any]], bool]
    priority: int = 0


@dataclass
class ChainBranch:
    """A branch in a conditional chain.

    Attributes:
        name: Branch identifier.
        condition: When to take this branch.
        steps: Steps to execute in this branch.
        model_override: Optional model override for all steps in branch.
    """

    name: str
    condition: Optional[RoutingCondition] = None
    steps: List["MultiModelStep"] = field(default_factory=list)
    model_override: Optional[str] = None


@dataclass
class ModelSelectionResult:
    """Result of model selection process.

    Attributes:
        selected_model: The chosen model profile.
        reason: Why this model was selected.
        alternatives: Other models that were considered.
        estimated_cost: Estimated cost for this step.
        estimated_latency_ms: Estimated latency for this step.
    """

    selected_model: ModelProfile
    reason: str = ""
    alternatives: List[ModelProfile] = field(default_factory=list)
    estimated_cost: float = 0.0
    estimated_latency_ms: float = 0.0


@dataclass
class MultiModelStep:
    """A step in a multi-model chain with routing capabilities.

    Attributes:
        name: Step identifier (must be unique within chain).
        prompt: Prompt template with {variable} placeholders.
        func: Optional transform function (alternative to prompt).
        complexity: Expected complexity level of this step.
        strategy: Model selection strategy for this step.
        model_constraints: Hard constraints for model selection.
        branches: Conditional branches from this step.
        fallback_model: Model to use if primary selection fails.
        max_cost: Maximum cost budget for this step in USD.
        max_latency_ms: Maximum acceptable latency in milliseconds.
        min_quality: Minimum quality score required.
        retry_count: Number of retries on failure.
        metadata: Additional step metadata.
    """

    name: str
    prompt: Optional[str] = None
    func: Optional[Callable] = None
    complexity: ComplexityLevel = ComplexityLevel.MEDIUM
    strategy: SelectionStrategy = SelectionStrategy.LOWEST_COST
    model_constraints: Dict[str, Any] = field(default_factory=dict)
    branches: List[ChainBranch] = field(default_factory=list)
    fallback_model: Optional[str] = None
    max_cost: Optional[float] = None
    max_latency_ms: Optional[float] = None
    min_quality: Optional[float] = None
    retry_count: int = 2
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.prompt and not self.func:
            raise ValueError(f"Step '{self.name}' must have either 'prompt' or 'func'")
        if self.prompt and self.func:
            raise ValueError(f"Step '{self.name}' cannot have both 'prompt' and 'func'")

    @property
    def is_llm_step(self) -> bool:
        """Whether this step requires an LLM invocation."""
        return self.prompt is not None

    @property
    def is_branching_step(self) -> bool:
        """Whether this step has conditional branches."""
        return len(self.branches) > 0

    @property
    def has_constraints(self) -> bool:
        """Whether this step has explicit model constraints."""
        return any([self.max_cost, self.max_latency_ms, self.min_quality])


@dataclass
class StepResult:
    """Result of executing a single step.

    Attributes:
        step_name: Name of the step that produced this result.
        output: The step's output value.
        model_used: Which model was used (if LLM step).
        cost: Actual cost incurred.
        latency_ms: Actual latency in milliseconds.
        branch_taken: Which branch was taken (if branching step).
        success: Whether the step completed successfully.
        error: Error message if step failed.
    """

    step_name: str
    output: Any = None
    model_used: Optional[str] = None
    cost: float = 0.0
    latency_ms: float = 0.0
    branch_taken: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


@dataclass
class MultiModelChainResult:
    """Result of executing a multi-model chain.

    Attributes:
        chain_name: Name of the chain that was executed.
        step_results: Results from each step in execution order.
        total_cost: Total cost across all steps.
        total_latency_ms: Total latency across all steps.
        models_used: Set of unique models used during execution.
        success: Whether all steps completed successfully.
        final_output: Output from the last step.
    """

    chain_name: str
    step_results: List[StepResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_latency_ms: float = 0.0
    models_used: List[str] = field(default_factory=list)
    success: bool = True
    final_output: Any = None

    @property
    def step_count(self) -> int:
        """Number of steps executed."""
        return len(self.step_results)

    @property
    def failed_steps(self) -> List[StepResult]:
        """Steps that failed during execution."""
        return [r for r in self.step_results if not r.success]

    def get_step_result(self, step_name: str) -> Optional[StepResult]:
        """Get the result of a specific step by name."""
        for result in self.step_results:
            if result.step_name == step_name:
                return result
        return None


class ModelRouter:
    """Routes requests to appropriate models based on selection strategy.

    The router maintains a registry of available models and selects
    the optimal model for each step based on the step's requirements
    and the configured selection strategy.

    Args:
        models: List of available model profiles.
        default_strategy: Default selection strategy when not specified per-step.
        custom_selector: Optional custom selection function.
    """

    def __init__(
        self,
        models: Optional[List[ModelProfile]] = None,
        default_strategy: SelectionStrategy = SelectionStrategy.LOWEST_COST,
        custom_selector: Optional[Callable[[List[ModelProfile], MultiModelStep], ModelProfile]] = None,
    ):
        self.models = models or []
        self.default_strategy = default_strategy
        self.custom_selector = custom_selector
        self._round_robin_index = 0

    def add_model(self, model: ModelProfile) -> None:
        """Register a model with the router."""
        self.models.append(model)

    def remove_model(self, model_id: str) -> bool:
        """Remove a model from the router. Returns True if found and removed."""
        initial_len = len(self.models)
        self.models = [m for m in self.models if m.model_id != model_id]
        return len(self.models) < initial_len

    def get_model(self, model_id: str) -> Optional[ModelProfile]:
        """Get a model profile by ID."""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def select_model(self, step: MultiModelStep) -> ModelSelectionResult:
        """Select the optimal model for a given step.

        Args:
            step: The step requiring model selection.

        Returns:
            ModelSelectionResult with selected model and reasoning.

        Raises:
            ValueError: If no suitable model is found.
        """
        candidates = self._filter_candidates(step)

        if not candidates:
            if step.fallback_model:
                fallback = self.get_model(step.fallback_model)
                if fallback:
                    return ModelSelectionResult(
                        selected_model=fallback,
                        reason="Fallback model used: no candidates met constraints",
                        alternatives=[],
                    )
            raise ValueError(
                f"No suitable model found for step '{step.name}' "
                f"with constraints: cost<={step.max_cost}, "
                f"latency<={step.max_latency_ms}ms, quality>={step.min_quality}"
            )

        strategy = step.strategy if step.strategy != SelectionStrategy.CUSTOM else self.default_strategy

        if strategy == SelectionStrategy.CUSTOM and self.custom_selector:
            selected = self.custom_selector(candidates, step)
        elif strategy == SelectionStrategy.LOWEST_COST:
            selected = min(candidates, key=lambda m: m.cost_per_1k_tokens)
        elif strategy == SelectionStrategy.LOWEST_LATENCY:
            selected = min(candidates, key=lambda m: m.avg_latency_ms)
        elif strategy == SelectionStrategy.HIGHEST_QUALITY:
            selected = max(candidates, key=lambda m: m.quality_score)
        elif strategy == SelectionStrategy.ROUND_ROBIN:
            selected = candidates[self._round_robin_index % len(candidates)]
            self._round_robin_index += 1
        elif strategy == SelectionStrategy.THRESHOLD:
            min_quality = step.min_quality or 0.7
            qualified = [m for m in candidates if m.quality_score >= min_quality]
            if qualified:
                selected = min(qualified, key=lambda m: m.cost_per_1k_tokens)
            else:
                selected = max(candidates, key=lambda m: m.quality_score)
        else:
            selected = min(candidates, key=lambda m: m.cost_per_1k_tokens)

        alternatives = [m for m in candidates if m.model_id != selected.model_id]

        return ModelSelectionResult(
            selected_model=selected,
            reason=f"Selected by {strategy.value} strategy",
            alternatives=alternatives,
            estimated_cost=selected.cost_per_1k_tokens,
            estimated_latency_ms=selected.avg_latency_ms,
        )

    def _filter_candidates(self, step: MultiModelStep) -> List[ModelProfile]:
        """Filter models based on step constraints."""
        candidates = self.models.copy()

        complexity_order = [
            ComplexityLevel.LOW,
            ComplexityLevel.MEDIUM,
            ComplexityLevel.HIGH,
            ComplexityLevel.CRITICAL,
        ]
        step_complexity_idx = complexity_order.index(step.complexity)
        candidates = [
            m for m in candidates
            if complexity_order.index(m.complexity_ceiling) >= step_complexity_idx
        ]

        if step.max_cost is not None:
            candidates = [m for m in candidates if m.cost_per_1k_tokens <= step.max_cost]

        if step.max_latency_ms is not None:
            candidates = [m for m in candidates if m.avg_latency_ms <= step.max_latency_ms]

        if step.min_quality is not None:
            candidates = [m for m in candidates if m.quality_score >= step.min_quality]

        if step.model_constraints.get("requires_json"):
            candidates = [m for m in candidates if m.supports_json]

        if step.model_constraints.get("requires_streaming"):
            candidates = [m for m in candidates if m.supports_streaming]

        min_tokens = step.model_constraints.get("min_tokens")
        if min_tokens:
            candidates = [m for m in candidates if m.max_tokens >= min_tokens]

        return candidates


class MultiModelChain:
    """A multi-model chain with conditional routing between models.

    Executes a sequence of steps, selecting the optimal model for each
    step based on complexity, cost, and latency requirements. Supports
    conditional branching where the path through the chain depends on
    intermediate results.

    Args:
        name: Chain identifier.
        steps: Ordered list of steps in the chain.
        router: Model router for selection decisions.
        max_total_cost: Maximum total cost budget for the entire chain.
        max_total_latency_ms: Maximum total latency for the chain.
        stop_on_failure: Whether to halt execution on step failure.
        metadata: Additional chain metadata.

    Example:
        router = ModelRouter(models=[
            ModelProfile("gpt-4", cost_per_1k_tokens=0.03, quality_score=0.95),
            ModelProfile("gpt-3.5-turbo", cost_per_1k_tokens=0.002, quality_score=0.7),
        ])

        chain = MultiModelChain(
            name="analyze-and-summarize",
            steps=[
                MultiModelStep("classify", prompt="Classify: {input}",
                    complexity=ComplexityLevel.LOW,
                    strategy=SelectionStrategy.LOWEST_COST),
                MultiModelStep("analyze", prompt="Deep analysis: {classify.output}",
                    complexity=ComplexityLevel.HIGH,
                    strategy=SelectionStrategy.HIGHEST_QUALITY),
            ],
            router=router,
        )

        result = chain.execute(input="Some text to analyze")
    """

    def __init__(
        self,
        name: str,
        steps: List[MultiModelStep],
        router: ModelRouter,
        max_total_cost: Optional[float] = None,
        max_total_latency_ms: Optional[float] = None,
        stop_on_failure: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.steps = steps
        self.router = router
        self.max_total_cost = max_total_cost
        self.max_total_latency_ms = max_total_latency_ms
        self.stop_on_failure = stop_on_failure
        self.metadata = metadata or {}

        self._validate()

    def _validate(self) -> None:
        """Validate chain configuration."""
        if not self.steps:
            raise ValueError(f"Chain '{self.name}' must have at least one step")

        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Chain '{self.name}' has duplicate step names")

        if not self.router.models:
            raise ValueError(f"Chain '{self.name}' requires at least one model in router")

    @property
    def step_count(self) -> int:
        """Total number of steps in the chain."""
        return len(self.steps)

    @property
    def llm_step_count(self) -> int:
        """Number of steps that require LLM invocations."""
        return sum(1 for s in self.steps if s.is_llm_step)

    def get_step(self, name: str) -> Optional[MultiModelStep]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def estimate_cost(self) -> float:
        """Estimate total cost for the chain based on model selection."""
        total = 0.0
        for step in self.steps:
            if step.is_llm_step:
                try:
                    result = self.router.select_model(step)
                    total += result.estimated_cost
                except ValueError:
                    pass
        return total

    def estimate_latency(self) -> float:
        """Estimate total latency for the chain in milliseconds."""
        total = 0.0
        for step in self.steps:
            if step.is_llm_step:
                try:
                    result = self.router.select_model(step)
                    total += result.estimated_latency_ms
                except ValueError:
                    pass
        return total

    def execute(
        self,
        context: Optional[Dict[str, Any]] = None,
        model_invoker: Optional[Callable] = None,
        **kwargs: Any,
    ) -> MultiModelChainResult:
        """Execute the multi-model chain.

        Args:
            context: Optional execution context.
            model_invoker: Optional callable to invoke models.
                Signature: (model_id, prompt, **kwargs) -> str
                If not provided, steps return selection metadata only.
            **kwargs: Input variables for prompt templates.

        Returns:
            MultiModelChainResult with results from all steps.
        """
        chain_result = MultiModelChainResult(chain_name=self.name)
        step_outputs: Dict[str, Any] = {}
        step_outputs.update(kwargs)
        accumulated_cost = 0.0
        accumulated_latency = 0.0

        for step in self.steps:
            # Check budget constraints
            if self.max_total_cost and accumulated_cost >= self.max_total_cost:
                step_result = StepResult(
                    step_name=step.name,
                    success=False,
                    error="Chain cost budget exceeded",
                )
                chain_result.step_results.append(step_result)
                chain_result.success = False
                if self.stop_on_failure:
                    break
                continue

            if self.max_total_latency_ms and accumulated_latency >= self.max_total_latency_ms:
                step_result = StepResult(
                    step_name=step.name,
                    success=False,
                    error="Chain latency budget exceeded",
                )
                chain_result.step_results.append(step_result)
                chain_result.success = False
                if self.stop_on_failure:
                    break
                continue

            # Handle branching
            if step.is_branching_step:
                branch_result = self._execute_branch(step, step_outputs, model_invoker)
                chain_result.step_results.append(branch_result)
                if branch_result.success:
                    step_outputs[step.name] = branch_result.output
                elif self.stop_on_failure:
                    chain_result.success = False
                    break
                continue

            # Execute step
            step_result = self._execute_step(step, step_outputs, model_invoker)
            chain_result.step_results.append(step_result)

            if step_result.success:
                step_outputs[step.name] = step_result.output
                accumulated_cost += step_result.cost
                accumulated_latency += step_result.latency_ms
                if step_result.model_used:
                    chain_result.models_used.append(step_result.model_used)
            elif self.stop_on_failure:
                chain_result.success = False
                break

        chain_result.total_cost = accumulated_cost
        chain_result.total_latency_ms = accumulated_latency

        if chain_result.step_results:
            last_successful = [r for r in chain_result.step_results if r.success]
            if last_successful:
                chain_result.final_output = last_successful[-1].output

        return chain_result

    def _execute_step(
        self,
        step: MultiModelStep,
        step_outputs: Dict[str, Any],
        model_invoker: Optional[Callable],
    ) -> StepResult:
        """Execute a single step with model selection."""
        if step.func:
            try:
                output = step.func(step_outputs)
                return StepResult(step_name=step.name, output=output, success=True)
            except Exception as e:
                return StepResult(step_name=step.name, success=False, error=str(e))

        try:
            selection = self.router.select_model(step)
        except ValueError as e:
            return StepResult(step_name=step.name, success=False, error=str(e))

        model_id = selection.selected_model.model_id
        prompt = self._resolve_prompt(step.prompt or "", step_outputs)

        if model_invoker:
            try:
                output = model_invoker(model_id, prompt)
                return StepResult(
                    step_name=step.name,
                    output=output,
                    model_used=model_id,
                    cost=selection.estimated_cost,
                    latency_ms=selection.estimated_latency_ms,
                    success=True,
                )
            except Exception as e:
                return StepResult(
                    step_name=step.name,
                    model_used=model_id,
                    success=False,
                    error=str(e),
                )
        else:
            return StepResult(
                step_name=step.name,
                output=f"[Would invoke {model_id}]: {prompt}",
                model_used=model_id,
                cost=selection.estimated_cost,
                latency_ms=selection.estimated_latency_ms,
                success=True,
            )

    def _execute_branch(
        self,
        step: MultiModelStep,
        step_outputs: Dict[str, Any],
        model_invoker: Optional[Callable],
    ) -> StepResult:
        """Execute a branching step by evaluating conditions."""
        sorted_branches = sorted(
            step.branches,
            key=lambda b: b.condition.priority if b.condition else -1,
            reverse=True,
        )

        for branch in sorted_branches:
            if branch.condition is None:
                return StepResult(
                    step_name=step.name,
                    output=f"Branch taken: {branch.name}",
                    branch_taken=branch.name,
                    success=True,
                )

            try:
                if branch.condition.predicate(step_outputs):
                    return StepResult(
                        step_name=step.name,
                        output=f"Branch taken: {branch.name}",
                        branch_taken=branch.name,
                        success=True,
                    )
            except Exception:
                continue

        if step.prompt or step.func:
            return self._execute_step(
                MultiModelStep(
                    name=step.name,
                    prompt=step.prompt,
                    func=step.func,
                    complexity=step.complexity,
                    strategy=step.strategy,
                ),
                step_outputs,
                model_invoker,
            )

        return StepResult(
            step_name=step.name,
            output=None,
            success=False,
            error="No branch condition matched and no default provided",
        )

    def _resolve_prompt(self, template: str, variables: Dict[str, Any]) -> str:
        """Resolve a prompt template with available variables."""
        resolved = template
        for key, value in variables.items():
            placeholder = "{" + key + "}"
            if placeholder in resolved:
                resolved = resolved.replace(placeholder, str(value))
            output_placeholder = "{" + key + ".output}"
            if output_placeholder in resolved:
                resolved = resolved.replace(output_placeholder, str(value))
        return resolved
