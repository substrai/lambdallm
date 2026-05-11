"""Chain and Step definitions for multi-step LLM pipelines.

Chains are declarative: define the steps, the framework handles execution,
state persistence, timeout management, and error recovery.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Union


@dataclass
class Step:
    """A single step in a chain pipeline.

    Each step can be:
    - A prompt template (string with {variables})
    - A transform function (Python callable)
    - A conditional branch

    Steps can reference outputs from previous steps using {step_name.output} syntax.

    Example:
        Step("extract", prompt="Extract entities from: {input}")
        Step("classify", prompt="Classify: {extract.output}")
        Step("transform", func=lambda data: data.upper())
    """

    name: str
    prompt: Optional[str] = None
    func: Optional[Callable] = None
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    output_schema: Optional[dict] = None
    condition: Optional[Callable] = None  # Skip step if condition returns False
    retry_count: int = 2
    metadata: dict = field(default_factory=dict)

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
    def is_transform_step(self) -> bool:
        """Whether this step is a pure Python transform."""
        return self.func is not None


@dataclass
class Chain:
    """A declarative multi-step LLM pipeline.

    Chains execute steps sequentially, passing outputs between steps.
    They handle Lambda timeout constraints with checkpoint/resume.

    Example:
        analyze = Chain(
            name="document-analysis",
            steps=[
                Step("extract", prompt="Extract key entities from: {input}"),
                Step("classify", prompt="Classify entities: {extract.output}"),
                Step("summarize", prompt="Summarize findings: {classify.output}"),
            ],
            timeout_strategy="checkpoint",
        )

        result = analyze.run(input="...", context=context)
    """

    name: str
    steps: list[Step]
    timeout_strategy: str = "fail-fast"  # fail-fast | checkpoint | truncate
    description: Optional[str] = None
    version: str = "1.0.0"
    max_total_cost: Optional[float] = None  # USD limit for entire chain
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.steps:
            raise ValueError(f"Chain '{self.name}' must have at least one step")

        # Validate step names are unique
        names = [s.name for s in self.steps]
        if len(names) != len(set(names)):
            raise ValueError(f"Chain '{self.name}' has duplicate step names")

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def llm_step_count(self) -> int:
        """Number of steps that require LLM invocations."""
        return sum(1 for s in self.steps if s.is_llm_step)

    def get_step(self, name: str) -> Optional[Step]:
        """Get a step by name."""
        for step in self.steps:
            if step.name == name:
                return step
        return None

    def run(self, context=None, **kwargs) -> Any:
        """Execute the chain.

        Args:
            context: LambdaLLMContext for model invocations.
            **kwargs: Input variables for the first step.

        Returns:
            ChainResult with outputs from all steps.
        """
        from lambdallm.chains.runner import ChainRunner

        runner = ChainRunner(self, context)
        return runner.execute(**kwargs)
