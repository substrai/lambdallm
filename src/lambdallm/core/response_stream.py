"""Advanced Lambda Response Streaming with Bedrock streaming API integration.

Provides a high-level interface for streaming LLM responses through
AWS Lambda Response Streaming, with support for:
- Bedrock InvokeModelWithResponseStream integration
- Chunked transfer encoding with backpressure handling
- Token-level streaming with metadata enrichment
- awslambdaric integration for Lambda runtime streaming
"""

from __future__ import annotations

import json
import hashlib
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Generator, List, Optional

logger = logging.getLogger("lambdallm.response_stream")


class StreamEvent(Enum):
    """Types of events in a response stream."""
    METADATA = "metadata"
    TOKEN = "token"
    CONTENT_BLOCK_START = "content_block_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_BLOCK_STOP = "content_block_stop"
    MESSAGE_STOP = "message_stop"
    ERROR = "error"
    METRICS = "metrics"


@dataclass
class StreamChunk:
    """A single chunk in the response stream."""
    event: StreamEvent
    data: str = ""
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    def to_bytes(self) -> bytes:
        """Serialize chunk to newline-delimited JSON bytes."""
        payload = {
            "event": self.event.value,
            "data": self.data,
            "index": self.index,
            "timestamp_ms": self.timestamp_ms,
            **self.metadata,
        }
        return json.dumps(payload).encode("utf-8") + b"\n"


@dataclass
class StreamMetrics:
    """Metrics collected during streaming."""
    first_token_ms: Optional[float] = None
    total_tokens: int = 0
    total_bytes: int = 0
    total_duration_ms: float = 0.0
    chunks_sent: int = 0
    errors: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "first_token_ms": self.first_token_ms,
            "total_tokens": self.total_tokens,
            "total_bytes": self.total_bytes,
            "total_duration_ms": self.total_duration_ms,
            "chunks_sent": self.chunks_sent,
            "errors": self.errors,
        }


class BedrockResponseStream:
    """Wraps Bedrock's InvokeModelWithResponseStream for Lambda streaming.

    Handles the Bedrock streaming response format and converts it into
    a normalized stream of StreamChunk objects suitable for Lambda
    Response Streaming.

    Usage:
        stream = BedrockResponseStream(
            bedrock_response=bedrock_client.invoke_model_with_response_stream(...),
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        for chunk in stream.iter_chunks():
            response_stream.write(chunk.to_bytes())
    """

    def __init__(
        self,
        bedrock_response: Any,
        model_id: str,
        include_usage: bool = True,
        on_token: Optional[Callable[[str], None]] = None,
    ):
        self._response = bedrock_response
        self._model_id = model_id
        self._include_usage = include_usage
        self._on_token = on_token
        self._metrics = StreamMetrics()
        self._start_time: Optional[float] = None
        self._accumulated_text: List[str] = []
        self._stream_id = hashlib.md5(
            f"{model_id}:{time.time()}".encode()
        ).hexdigest()[:12]

    @property
    def metrics(self) -> StreamMetrics:
        """Get streaming metrics."""
        return self._metrics

    @property
    def stream_id(self) -> str:
        """Unique identifier for this stream."""
        return self._stream_id

    @property
    def accumulated_text(self) -> str:
        """Get all text accumulated so far."""
        return "".join(self._accumulated_text)

    def iter_chunks(self) -> Generator[StreamChunk, None, None]:
        """Iterate over response chunks from Bedrock streaming API.

        Yields StreamChunk objects normalized from the Bedrock response
        stream format. Handles both Anthropic Messages API and legacy
        text completion formats.
        """
        self._start_time = time.time()

        # Emit metadata chunk first
        yield StreamChunk(
            event=StreamEvent.METADATA,
            metadata={
                "model_id": self._model_id,
                "stream_id": self._stream_id,
                "streaming": True,
            },
        )

        try:
            event_stream = self._get_event_stream()
            chunk_index = 0

            for event in event_stream:
                chunks = self._process_bedrock_event(event, chunk_index)
                for chunk in chunks:
                    self._metrics.chunks_sent += 1
                    self._metrics.total_bytes += len(chunk.data.encode("utf-8"))
                    chunk_index += 1
                    yield chunk

        except Exception as e:
            self._metrics.errors += 1
            logger.error(f"Stream error [{self._stream_id}]: {e}")
            yield StreamChunk(
                event=StreamEvent.ERROR,
                data=str(e),
                metadata={"stream_id": self._stream_id},
            )

        finally:
            end_time = time.time()
            self._metrics.total_duration_ms = (end_time - self._start_time) * 1000

            if self._include_usage:
                yield StreamChunk(
                    event=StreamEvent.METRICS,
                    metadata={
                        "stream_id": self._stream_id,
                        "metrics": self._metrics.to_dict(),
                    },
                )

    def _get_event_stream(self) -> Any:
        """Extract the event stream from the Bedrock response."""
        if hasattr(self._response, "get"):
            body = self._response.get("body", self._response)
            if hasattr(body, "__iter__"):
                return body
        if hasattr(self._response, "__iter__"):
            return self._response
        raise ValueError("Unable to extract event stream from Bedrock response")

    def _process_bedrock_event(self, event: Any, index: int) -> List[StreamChunk]:
        """Process a single Bedrock streaming event into StreamChunks."""
        chunks: List[StreamChunk] = []

        if isinstance(event, dict):
            event_type = event.get("type", "")

            if event_type == "content_block_start":
                chunks.append(StreamChunk(
                    event=StreamEvent.CONTENT_BLOCK_START,
                    index=index,
                    metadata={"block_type": event.get("content_block", {}).get("type", "text")},
                ))

            elif event_type == "content_block_delta":
                delta = event.get("delta", {})
                text = delta.get("text", "")
                if text:
                    self._record_token(text)
                    chunks.append(StreamChunk(
                        event=StreamEvent.CONTENT_BLOCK_DELTA,
                        data=text,
                        index=index,
                    ))

            elif event_type == "content_block_stop":
                chunks.append(StreamChunk(
                    event=StreamEvent.CONTENT_BLOCK_STOP,
                    index=index,
                ))

            elif event_type == "message_stop":
                chunks.append(StreamChunk(
                    event=StreamEvent.MESSAGE_STOP,
                    index=index,
                ))

            elif "chunk" in event:
                chunk_data = event["chunk"]
                if isinstance(chunk_data, dict) and "bytes" in chunk_data:
                    decoded = json.loads(chunk_data["bytes"].decode("utf-8"))
                    text = self._extract_text_from_legacy(decoded)
                    if text:
                        self._record_token(text)
                        chunks.append(StreamChunk(
                            event=StreamEvent.TOKEN,
                            data=text,
                            index=index,
                        ))

        elif isinstance(event, bytes):
            try:
                decoded = json.loads(event.decode("utf-8"))
                text = self._extract_text_from_legacy(decoded)
                if text:
                    self._record_token(text)
                    chunks.append(StreamChunk(
                        event=StreamEvent.TOKEN,
                        data=text,
                        index=index,
                    ))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        return chunks

    def _extract_text_from_legacy(self, decoded: Dict[str, Any]) -> str:
        """Extract text from legacy Bedrock completion format."""
        if "completion" in decoded:
            return decoded["completion"]
        if "outputText" in decoded:
            return decoded["outputText"]
        return decoded.get("text", "")

    def _record_token(self, text: str) -> None:
        """Record a token for metrics and callbacks."""
        self._metrics.total_tokens += 1
        self._accumulated_text.append(text)

        if self._metrics.first_token_ms is None and self._start_time:
            self._metrics.first_token_ms = (time.time() - self._start_time) * 1000

        if self._on_token:
            try:
                self._on_token(text)
            except Exception as e:
                logger.warning(f"Token callback error: {e}")


class LambdaResponseStreamWriter:
    """Writes streaming responses to Lambda's response stream with awslambdaric.

    Integrates with the AWS Lambda Runtime Interface Client (awslambdaric)
    to deliver streaming responses through Lambda Function URLs.

    Usage:
        def handler(event, response_stream, context):
            writer = LambdaResponseStreamWriter(response_stream)
            bedrock_stream = BedrockResponseStream(
                bedrock_response=client.invoke_model_with_response_stream(...),
                model_id="anthropic.claude-3-sonnet-20240229-v1:0",
            )
            writer.stream_response(bedrock_stream)
    """

    CONTENT_TYPE = "application/vnd.awslambda.http-integration-response"

    def __init__(
        self,
        response_stream: Any,
        content_type: str = "text/event-stream",
        buffer_size: int = 0,
        on_complete: Optional[Callable[[StreamMetrics], None]] = None,
    ):
        self._stream = response_stream
        self._content_type = content_type
        self._buffer_size = buffer_size
        self._on_complete = on_complete
        self._buffer: List[bytes] = []
        self._total_buffered: int = 0
        self._closed = False

    @property
    def is_closed(self) -> bool:
        """Check if the stream has been closed."""
        return self._closed

    def write_metadata(self, metadata: Dict[str, Any]) -> None:
        """Write HTTP response metadata for Lambda Function URL integration."""
        if self._closed:
            raise RuntimeError("Cannot write to closed stream")

        prelude = {
            "statusCode": metadata.get("status_code", 200),
            "headers": {
                "Content-Type": self._content_type,
                "Cache-Control": "no-cache",
                "X-Stream-Id": metadata.get("stream_id", ""),
                **metadata.get("headers", {}),
            },
        }
        self._write_raw(json.dumps(prelude).encode("utf-8") + b"\x00\x00\x00\x00\x00\x00\x00\x00")

    def write_chunk(self, chunk: StreamChunk) -> None:
        """Write a single StreamChunk to the response stream."""
        if self._closed:
            raise RuntimeError("Cannot write to closed stream")

        data = chunk.to_bytes()

        if self._buffer_size > 0:
            self._buffer.append(data)
            self._total_buffered += len(data)
            if self._total_buffered >= self._buffer_size:
                self._flush_buffer()
        else:
            self._write_raw(data)

    def stream_response(
        self,
        bedrock_stream: BedrockResponseStream,
        include_http_metadata: bool = True,
    ) -> StreamMetrics:
        """Stream a complete BedrockResponseStream to the Lambda response stream."""
        if self._closed:
            raise RuntimeError("Cannot write to closed stream")

        try:
            if include_http_metadata:
                self.write_metadata({"stream_id": bedrock_stream.stream_id})

            for chunk in bedrock_stream.iter_chunks():
                self.write_chunk(chunk)

            self._flush_buffer()

        except Exception as e:
            logger.error(f"Lambda stream write error: {e}")
            error_chunk = StreamChunk(event=StreamEvent.ERROR, data=str(e))
            try:
                self._write_raw(error_chunk.to_bytes())
            except Exception:
                pass

        finally:
            self._close()

        metrics = bedrock_stream.metrics
        if self._on_complete:
            try:
                self._on_complete(metrics)
            except Exception as e:
                logger.warning(f"on_complete callback error: {e}")

        return metrics

    def _write_raw(self, data: bytes) -> None:
        """Write raw bytes to the underlying stream."""
        if hasattr(self._stream, "write"):
            self._stream.write(data)
        elif hasattr(self._stream, "send"):
            self._stream.send(data)
        else:
            raise TypeError(f"Response stream does not support write/send: {type(self._stream)}")

    def _flush_buffer(self) -> None:
        """Flush the internal buffer to the stream."""
        if self._buffer:
            combined = b"".join(self._buffer)
            self._write_raw(combined)
            self._buffer.clear()
            self._total_buffered = 0

    def _close(self) -> None:
        """Close the response stream."""
        if not self._closed:
            self._flush_buffer()
            if hasattr(self._stream, "close"):
                self._stream.close()
            self._closed = True


def streaming_handler(
    model: Optional[str] = None,
    content_type: str = "text/event-stream",
    buffer_size: int = 0,
    include_usage: bool = True,
):
    """Decorator for Lambda handlers that stream Bedrock responses.

    Simplifies the creation of streaming Lambda handlers by automatically
    managing the BedrockResponseStream and LambdaResponseStreamWriter.

    Example:
        @streaming_handler(model="anthropic.claude-3-sonnet-20240229-v1:0")
        def handler(event, context):
            return {
                "prompt": event["body"]["prompt"],
                "streaming": True,
                "max_tokens": 1024,
            }
    """
    import functools

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(event: Dict[str, Any], response_stream: Any, context: Any) -> None:
            writer = LambdaResponseStreamWriter(
                response_stream=response_stream,
                content_type=content_type,
                buffer_size=buffer_size,
            )

            try:
                result = func(event, context)

                if isinstance(result, BedrockResponseStream):
                    writer.stream_response(result)
                elif isinstance(result, dict) and result.get("streaming"):
                    _handle_config_result(result, writer, model, include_usage)
                else:
                    chunk = StreamChunk(
                        event=StreamEvent.TOKEN,
                        data=json.dumps(result) if isinstance(result, dict) else str(result),
                    )
                    writer.write_chunk(chunk)
                    writer._close()

            except Exception as e:
                logger.error(f"Streaming handler error: {e}")
                error_chunk = StreamChunk(event=StreamEvent.ERROR, data=str(e))
                try:
                    writer.write_chunk(error_chunk)
                except Exception:
                    pass
                finally:
                    writer._close()

        wrapper._lambdallm_response_stream = True
        return wrapper

    return decorator


def _handle_config_result(
    config: Dict[str, Any],
    writer: LambdaResponseStreamWriter,
    model: Optional[str],
    include_usage: bool,
) -> None:
    """Handle a config dict result from a streaming handler."""
    try:
        import boto3

        client = boto3.client("bedrock-runtime")
        model_id = config.get("model", model)

        if not model_id:
            raise ValueError("No model specified in config or decorator")

        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": config.get("max_tokens", 1024),
            "messages": [{"role": "user", "content": config["prompt"]}],
        }

        if "system" in config:
            body["system"] = config["system"]
        if "temperature" in config:
            body["temperature"] = config["temperature"]

        response = client.invoke_model_with_response_stream(
            modelId=model_id,
            body=json.dumps(body),
        )

        bedrock_stream = BedrockResponseStream(
            bedrock_response=response.get("body", []),
            model_id=model_id,
            include_usage=include_usage,
        )

        writer.stream_response(bedrock_stream)

    except ImportError:
        raise RuntimeError("boto3 is required for automatic Bedrock streaming")
    except Exception as e:
        logger.error(f"Config-based streaming error: {e}")
        error_chunk = StreamChunk(event=StreamEvent.ERROR, data=str(e))
        writer.write_chunk(error_chunk)
        writer._close()
