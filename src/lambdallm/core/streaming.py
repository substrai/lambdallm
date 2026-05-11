"""Lambda Response Streaming support for LambdaLLM.

Enables real-time token delivery to clients using Lambda's
response streaming feature (Function URL with RESPONSE_STREAM).
"""

import json
import logging
from typing import Any, Generator, Optional

logger = logging.getLogger("lambdallm")


class StreamingResponse:
    """Wraps a streaming LLM response for Lambda Response Streaming.

    Usage with Lambda Function URL (streaming mode):
        @handler(model=Model.CLAUDE_3_SONNET)
        def lambda_handler(event, context):
            return context.invoke_streaming("Tell me a story about {topic}", topic="AI")

    The framework detects StreamingResponse and writes chunks
    to the Lambda response stream.
    """

    def __init__(self, generator: Generator[str, None, None], metadata: Optional[dict] = None):
        self.generator = generator
        self.metadata = metadata or {}
        self._chunks: list[str] = []
        self._complete = False

    def __iter__(self):
        """Iterate over response chunks."""
        for chunk in self.generator:
            self._chunks.append(chunk)
            yield chunk
        self._complete = True

    @property
    def full_text(self) -> str:
        """Get the complete response text (blocks until stream completes)."""
        if not self._complete:
            # Consume remaining chunks
            for chunk in self.generator:
                self._chunks.append(chunk)
            self._complete = True
        return "".join(self._chunks)

    def to_lambda_stream(self) -> Generator[bytes, None, None]:
        """Convert to Lambda Response Streaming format.

        Yields newline-delimited JSON chunks compatible with
        Lambda Function URL streaming invocation type.
        """
        # Send metadata header
        yield json.dumps({"type": "metadata", **self.metadata}).encode() + b"\n"

        # Stream content chunks
        for chunk in self:
            yield json.dumps({"type": "content", "text": chunk}).encode() + b"\n"

        # Send completion marker
        yield json.dumps({"type": "complete", "full_text_length": len(self.full_text)}).encode() + b"\n"


class StreamingHandler:
    """Handles Lambda Response Streaming integration.

    Detects when a handler returns a StreamingResponse and
    writes it to the Lambda response stream appropriately.
    """

    @staticmethod
    def is_streaming_response(result: Any) -> bool:
        """Check if the handler returned a streaming response."""
        return isinstance(result, StreamingResponse)

    @staticmethod
    def write_to_stream(response_stream: Any, streaming_response: StreamingResponse) -> None:
        """Write a StreamingResponse to a Lambda response stream.

        Args:
            response_stream: The Lambda response stream object.
            streaming_response: The StreamingResponse to write.
        """
        try:
            for chunk_bytes in streaming_response.to_lambda_stream():
                response_stream.write(chunk_bytes)
            response_stream.close()
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            error_chunk = json.dumps({"type": "error", "message": str(e)}).encode() + b"\n"
            response_stream.write(error_chunk)
            response_stream.close()


def stream_handler(
    model=None,
    timeout_buffer: int = 5,
    max_retries: int = 3,
):
    """Decorator for streaming Lambda handlers.

    Similar to @handler but designed for Lambda Function URLs
    with RESPONSE_STREAM invocation type.

    Example:
        @stream_handler(model=Model.CLAUDE_3_SONNET)
        def lambda_handler(event, context):
            return context.invoke_streaming(
                "Write a story about {topic}",
                topic=event["body"]["topic"]
            )
    """
    import functools
    from lambdallm.core.handler import _resolve_model
    from lambdallm.core.context import LambdaLLMContext

    def decorator(func):
        @functools.wraps(func)
        def wrapper(event, response_stream, lambda_context):
            llm_context = LambdaLLMContext(
                model=_resolve_model(model),
                timeout_strategy="fail-fast",
                timeout_buffer=timeout_buffer,
                max_retries=max_retries,
                fallback_model=None,
                lambda_context=lambda_context,
                middleware=[],
            )

            try:
                result = func(event, llm_context)

                if StreamingHandler.is_streaming_response(result):
                    StreamingHandler.write_to_stream(response_stream, result)
                else:
                    # Non-streaming response, write as single chunk
                    response_bytes = json.dumps(result).encode() + b"\n"
                    response_stream.write(response_bytes)
                    response_stream.close()

            except Exception as e:
                error_response = json.dumps({"error": str(e)}).encode() + b"\n"
                response_stream.write(error_response)
                response_stream.close()

        wrapper._lambdallm_streaming = True
        return wrapper

    return decorator
