"""Example: Basic document summarizer using LambdaLLM.

Deploy this as a Lambda function to get a GenAI-powered
summarization API in under 5 minutes.

Usage:
    lambdallm init --template basic
    lambdallm deploy
"""

from lambdallm import handler, Prompt, Model

# Define a reusable, type-safe prompt
summarize = Prompt(
    name="summarize",
    template="""Summarize the following document in {max_words} words or less.
Focus on the key points and main conclusions.

Document:
{document}""",
    input_schema={"document": str, "max_words": int},
    output_schema={"summary": str, "key_points": list},
)


@handler(model=Model.CLAUDE_3_HAIKU, timeout_strategy="truncate")
def lambda_handler(event, context):
    """Lambda handler for document summarization.

    Expected event body:
    {
        "text": "The document to summarize...",
        "max_words": 100
    }
    """
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    result = summarize.invoke(
        _context=context,
        document=body.get("text", ""),
        max_words=body.get("max_words", 100),
    )

    return {
        "statusCode": 200,
        "body": {
            "result": result,
            "cost_usd": context.total_cost,
            "model": context.model.model_id,
        },
    }
