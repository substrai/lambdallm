"""Golden Dataset Tests: Regression Testing for Prompts

Run prompts against known input/output pairs to detect regressions
when prompts, models, or configurations change.

Usage:
    pytest tests/golden/test_golden.py -v
    lambdallm test --golden
"""

import json
import pytest
from pathlib import Path

from lambdallm import handler, Prompt, Model
from lambdallm.testing import MockProvider, MockLambdaContext, mock_model, GoldenDatasetRunner


# Define the handler under test
summarize = Prompt(
    template="Summarize in {max_words} words: {text}",
    input_schema={"text": str, "max_words": int},
)

@handler(model=Model.CLAUDE_3_HAIKU)
def summarize_handler(event, context):
    import json as j
    body = j.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})
    result = context.invoke("Summarize in {max_words} words: {text}", text=body["text"], max_words=body["max_words"])
    return {"statusCode": 200, "body": {"summary": result}}


class TestGoldenDatasets:
    """Run golden dataset regression tests."""

    @mock_model(responses=[
        "LambdaLLM is a serverless framework for orchestrating LLMs on Lambda.",
        "Hello world greeting.",
        "AWS Lambda provides stateless compute with 15-min timeout and 250MB limit.",
    ])
    def test_summarize_golden_dataset(self):
        """All golden test cases should pass."""
        runner = GoldenDatasetRunner()
        dataset_path = str(Path(__file__).parent / "summarize_cases.json")

        result = runner.run(dataset_path, handler=summarize_handler)

        print(result.summary())
        assert result.pass_rate >= 0.9, f"Pass rate {result.pass_rate:.1%} below 90% threshold"
        assert result.avg_latency_ms < 5000, f"Avg latency {result.avg_latency_ms:.0f}ms too high"

    @mock_model(responses=["Quick summary of the text."])
    def test_single_golden_case(self):
        """Test a single case for quick validation."""
        runner = GoldenDatasetRunner()
        dataset_path = str(Path(__file__).parent / "summarize_cases.json")

        result = runner.run(dataset_path, handler=summarize_handler)
        # At minimum, no exceptions should occur
        assert result.total_cases > 0


class TestCustomGoldenDataset:
    """Show users how to create their own golden datasets."""

    @mock_model(responses=["Good morning summary.", "Python is a language for programming."])
    def test_custom_dataset_format(self, tmp_path):
        """Demonstrate creating a custom golden dataset."""
        # Create a custom dataset
        custom_cases = [
            {
                "name": "greeting",
                "input": {"text": "Good morning everyone!", "max_words": 5},
                "expected_contains": ["morning"],
            },
            {
                "name": "technical",
                "input": {"text": "Python is a programming language.", "max_words": 10},
                "expected_contains": ["Python"],
                "expected_not_contains": ["Java"],
            },
        ]

        dataset_file = tmp_path / "custom_cases.json"
        dataset_file.write_text(json.dumps(custom_cases))

        runner = GoldenDatasetRunner()
        result = runner.run(str(dataset_file), handler=summarize_handler)
        assert result.pass_rate >= 0.5  # At least some pass
