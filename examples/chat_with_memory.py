"""Example: Multi-turn chat with conversation memory.

Demonstrates how LambdaLLM handles stateless Lambda + stateful conversations
using DynamoDB-backed session management.
"""

from lambdallm import handler, Model
from lambdallm.core.context import LambdaLLMContext


@handler(
    model=Model.CLAUDE_3_SONNET,
    timeout_strategy="fail-fast",
    timeout_buffer=10,
)
def lambda_handler(event, context: LambdaLLMContext):
    """Multi-turn chat handler.

    Expected event body:
    {
        "message": "User's message",
        "session_id": "unique-session-id"
    }
    """
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    user_message = body.get("message", "")
    session_id = body.get("session_id", "default")

    # In Phase 2, this will auto-load from DynamoDB:
    # session = context.session  # auto-loaded via session_id

    # For now, simple single-turn response
    response = context.invoke(
        "You are a helpful assistant. Respond to: {message}",
        message=user_message,
    )

    return {
        "statusCode": 200,
        "body": {
            "reply": response,
            "session_id": session_id,
            "cost_usd": context.total_cost,
            "remaining_time_ms": context.remaining_time_ms,
        },
    }
