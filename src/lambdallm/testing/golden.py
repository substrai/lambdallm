"""Golden dataset testing for LambdaLLM.

Run prompts against golden datasets (expected input/output pairs)
to detect regressions when prompts or models change.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger("lambdallm")


@dataclass
class GoldenCase:
    """A single test case in a golden dataset."""

    name: str
    input: dict
    expected_output: Optional[Any] = None
    expected_contains: Optional[list[str]] = None
    expected_not_contains: Optional[list[str]] = None
    max_latency_ms: Optional[float] = None
    max_cost_usd: Optional[float] = None
    tags: list[str] = field(default_factory=list)


@dataclass
class GoldenCaseResult:
    """Result of running a single golden test case."""

    case_name: str
    passed: bool
    actual_output: Any = None
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    failure_reason: Optional[str] = None


@dataclass
class GoldenResult:
    """Aggregated results from a golden dataset run."""

    dataset_name: str
    total_cases: int = 0
    passed: int = 0
    failed: int = 0
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0
    results: list[GoldenCaseResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return self.passed / max(self.total_cases, 1)

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / max(self.total_cases, 1)

    @property
    def avg_cost_usd(self) -> float:
        return self.total_cost_usd / max(self.total_cases, 1)

    def summary(self) -> str:
        return (
            f"Golden Dataset: {self.dataset_name}\n"
            f"  Pass rate: {self.pass_rate:.1%} ({self.passed}/{self.total_cases})\n"
            f"  Avg latency: {self.avg_latency_ms:.0f}ms\n"
            f"  Total cost: ${self.total_cost_usd:.4f}\n"
            f"  Failed: {self.failed}"
        )


class GoldenDatasetRunner:
    """Runs golden dataset tests against handlers or prompts.

    Golden datasets are JSON files with input/expected-output pairs.
    Use them to detect regressions when changing prompts or models.

    Dataset format (JSON):
    [
        {
            "name": "basic_summary",
            "input": {"text": "Long document...", "max_words": 50},
            "expected_contains": ["key point"],
            "max_latency_ms": 3000
        }
    ]

    Example:
        runner = GoldenDatasetRunner()
        result = runner.run("tests/golden/summarize_cases.json", handler=my_handler)
        assert result.pass_rate >= 0.95
    """

    def __init__(self):
        self._results: list[GoldenResult] = []

    def run(
        self,
        dataset_path: str,
        handler: Optional[Callable] = None,
        prompt: Optional[Any] = None,
        context: Optional[Any] = None,
    ) -> GoldenResult:
        """Run a golden dataset.

        Args:
            dataset_path: Path to JSON golden dataset file.
            handler: Lambda handler function to test.
            prompt: Prompt object to test (alternative to handler).
            context: Optional LambdaLLMContext.

        Returns:
            GoldenResult with pass/fail for each case.
        """
        cases = self._load_dataset(dataset_path)
        dataset_name = Path(dataset_path).stem

        result = GoldenResult(dataset_name=dataset_name, total_cases=len(cases))

        for case in cases:
            case_result = self._run_case(case, handler, prompt, context)
            result.results.append(case_result)
            result.total_latency_ms += case_result.latency_ms
            result.total_cost_usd += case_result.cost_usd

            if case_result.passed:
                result.passed += 1
            else:
                result.failed += 1
                logger.warning(f"  FAIL: {case_result.case_name} - {case_result.failure_reason}")

        self._results.append(result)
        logger.info(result.summary())
        return result

    def _run_case(
        self,
        case: GoldenCase,
        handler: Optional[Callable],
        prompt: Optional[Any],
        context: Optional[Any],
    ) -> GoldenCaseResult:
        """Run a single test case."""
        start = time.time()

        try:
            if handler:
                from lambdallm.testing.mocks import MockLambdaContext
                event = {"body": json.dumps(case.input)}
                output = handler(event, context or MockLambdaContext())
                if isinstance(output, dict) and "body" in output:
                    actual = output["body"]
                    if isinstance(actual, str):
                        actual = json.loads(actual)
                else:
                    actual = output
            elif prompt:
                actual = prompt.invoke(**case.input)
            else:
                return GoldenCaseResult(case_name=case.name, passed=False, failure_reason="No handler or prompt provided")

            latency = (time.time() - start) * 1000

            # Validate
            passed, reason = self._validate(case, actual, latency)

            return GoldenCaseResult(
                case_name=case.name,
                passed=passed,
                actual_output=actual,
                latency_ms=latency,
                failure_reason=reason,
            )

        except Exception as e:
            latency = (time.time() - start) * 1000
            return GoldenCaseResult(
                case_name=case.name,
                passed=False,
                latency_ms=latency,
                failure_reason=f"Exception: {e}",
            )

    def _validate(self, case: GoldenCase, actual: Any, latency_ms: float) -> tuple[bool, Optional[str]]:
        """Validate actual output against expected."""
        actual_str = json.dumps(actual) if not isinstance(actual, str) else actual

        # Check expected_contains
        if case.expected_contains:
            for expected in case.expected_contains:
                if expected.lower() not in actual_str.lower():
                    return False, f"Expected to contain '{expected}'"

        # Check expected_not_contains
        if case.expected_not_contains:
            for not_expected in case.expected_not_contains:
                if not_expected.lower() in actual_str.lower():
                    return False, f"Should not contain '{not_expected}'"

        # Check latency
        if case.max_latency_ms and latency_ms > case.max_latency_ms:
            return False, f"Latency {latency_ms:.0f}ms exceeds max {case.max_latency_ms}ms"

        # Check exact match
        if case.expected_output is not None:
            if actual != case.expected_output:
                return False, f"Output mismatch"

        return True, None

    def _load_dataset(self, path: str) -> list[GoldenCase]:
        """Load golden dataset from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)

        cases = []
        for item in data:
            cases.append(GoldenCase(
                name=item.get("name", f"case_{len(cases)}"),
                input=item.get("input", {}),
                expected_output=item.get("expected_output"),
                expected_contains=item.get("expected_contains"),
                expected_not_contains=item.get("expected_not_contains"),
                max_latency_ms=item.get("max_latency_ms"),
                max_cost_usd=item.get("max_cost_usd"),
                tags=item.get("tags", []),
            ))

        return cases
