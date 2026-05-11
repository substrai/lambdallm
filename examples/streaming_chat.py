"""Example: Streaming chat with Lambda Response Streaming.

Demonstrates real-time token delivery using Lambda Function URLs
with RESPONSE_STREAM invocation type.
"""

from lambdallm import stream_handler, Model, Session
from lambdallm.core.streaming import StreamingResponse


@stream_handler(model=Model.CLAUDE_3_SONNET)
def lambda_handler(event, context):
    """Streaming chat handler for Lambda Function URL.

    Deploy with:
        Function URL invocation mode: RESPONSE_STREAM

    Client receives tokens in real-time as they're generated.
    """
    import json

    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})

    message = body.get("message", "Hello!")

    # Return a streaming response
    # The framework handles writing chunks to the Lambda response stream
    def generate():
        # In production, this would stream from the Bedrock API
        # For now, simulate streaming by yielding the full response
        response = context.invoke(
            "You are a helpful assistant. Respond to: {message}",
            message=message,
        )
        # Yield in chunks (simulating token-by-token delivery)
        words = response.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")

    return StreamingResponse(
        generate(),
        metadata={"model": "claude-3-sonnet", "session_id": body.get("session_id", "none")},
    )
