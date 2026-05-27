"""Tests for Lambda Response Streaming with Bedrock integration."""

import json
import time
from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from lambdallm.core.response_stream import (
    BedrockResponseStream,
    LambdaResponseStreamWriter,
    StreamChunk,
    StreamEvent,
    StreamMetrics,
    streaming_handler,
)


class MockResponseStream:
    """Mock Lambda response stream for testing."""

    def __init__(self):
        self.data = BytesIO()
        self.closed = False

    def write(self, data: bytes):
        if self.closed:
            raise RuntimeError("Stream is closed")
        self.data.write(data)

    def close(self):
        self.closed = True

    def get_written_data(self) -> bytes:
        return self.data.getvalue()

    def get_chunks(self) -> list:
        raw = self.data.getvalue().decode("utf-8")
        chunks = []
        for line in raw.split("\n"):
            line = line.strip().strip("\x00")
            if line:
                try:
                    chunks.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return chunks


class TestStreamChunk:
    """Tests for StreamChunk dataclass."""

    def test_to_bytes_basic(self):
        chunk = StreamChunk(
            event=StreamEvent.TOKEN,
            data="Hello",
            index=0,
            timestamp_ms=1000.0,
        )
        result = chunk.to_bytes()
        assert result.endswith(b"\n")
        parsed = json.loads(result.decode("utf-8"))
        assert parsed["event"] == "token"
        assert parsed["data"] == "Hello"
        assert parsed["index"] == 0

    def test_to_bytes_with_metadata(self):
        chunk = StreamChunk(
            event=StreamEvent.METADATA,
            metadata={"model_id": "claude-3", "stream_id": "abc123"},
        )
        result = chunk.to_bytes()
        parsed = json.loads(result.decode("utf-8"))
        assert parsed["event"] == "metadata"
        assert parsed["model_id"] == "claude-3"
        assert parsed["stream_id"] == "abc123"

    def test_stream_event_values(self):
        assert StreamEvent.TOKEN.value == "token"
        assert StreamEvent.ERROR.value == "error"
        assert StreamEvent.METRICS.value == "metrics"
        assert StreamEvent.CONTENT_BLOCK_DELTA.value == "content_block_delta"

    def test_timestamp_auto_generated(self):
        before = time.time() * 1000
        chunk = StreamChunk(event=StreamEvent.TOKEN, data="test")
        after = time.time() * 1000
        assert before <= chunk.timestamp_ms <= after


class TestStreamMetrics:
    """Tests for StreamMetrics."""

    def test_default_values(self):
        metrics = StreamMetrics()
        assert metrics.first_token_ms is None
        assert metrics.total_tokens == 0
        assert metrics.total_bytes == 0
        assert metrics.total_duration_ms == 0.0
        assert metrics.chunks_sent == 0
        assert metrics.errors == 0

    def test_to_dict(self):
        metrics = StreamMetrics(
            first_token_ms=50.0,
            total_tokens=10,
            total_bytes=256,
            total_duration_ms=1000.0,
            chunks_sent=12,
            errors=0,
        )
        d = metrics.to_dict()
        assert d["first_token_ms"] == 50.0
        assert d["total_tokens"] == 10
        assert d["total_bytes"] == 256
        assert d["total_duration_ms"] == 1000.0


class TestBedrockResponseStream:
    """Tests for BedrockResponseStream."""

    def test_anthropic_messages_format(self):
        """Test processing Anthropic Messages API streaming events."""
        events = [
            {"type": "content_block_start", "content_block": {"type": "text"}},
            {"type": "content_block_delta", "delta": {"text": "Hello"}},
            {"type": "content_block_delta", "delta": {"text": " world"}},
            {"type": "content_block_stop"},
            {"type": "message_stop"},
        ]

        stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="anthropic.claude-3-sonnet-20240229-v1:0",
        )

        chunks = list(stream.iter_chunks())
        # First chunk is metadata
        assert chunks[0].event == StreamEvent.METADATA
        # Should have content block events
        token_chunks = [c for c in chunks if c.event == StreamEvent.CONTENT_BLOCK_DELTA]
        assert len(token_chunks) == 2
        assert token_chunks[0].data == "Hello"
        assert token_chunks[1].data == " world"
        assert stream.accumulated_text == "Hello world"

    def test_legacy_chunk_format(self):
        """Test processing legacy Bedrock chunk format."""
        events = [
            {"chunk": {"bytes": json.dumps({"completion": "Once"}).encode()}},
            {"chunk": {"bytes": json.dumps({"completion": " upon"}).encode()}},
            {"chunk": {"bytes": json.dumps({"completion": " a time"}).encode()}},
        ]

        stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="anthropic.claude-v2",
        )

        chunks = list(stream.iter_chunks())
        token_chunks = [c for c in chunks if c.event == StreamEvent.TOKEN]
        assert len(token_chunks) == 3
        assert stream.accumulated_text == "Once upon a time"

    def test_metrics_tracking(self):
        """Test that metrics are properly tracked during streaming."""
        events = [
            {"type": "content_block_delta", "delta": {"text": "Hello"}},
            {"type": "content_block_delta", "delta": {"text": " world"}},
        ]

        stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="test-model",
        )

        chunks = list(stream.iter_chunks())
        metrics = stream.metrics
        assert metrics.total_tokens == 2
        assert metrics.total_bytes > 0
        assert metrics.first_token_ms is not None
        assert metrics.total_duration_ms > 0
        assert metrics.chunks_sent == 2

    def test_on_token_callback(self):
        """Test that on_token callback is invoked for each token."""
        received_tokens = []
        events = [
            {"type": "content_block_delta", "delta": {"text": "A"}},
            {"type": "content_block_delta", "delta": {"text": "B"}},
        ]

        stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="test-model",
            on_token=lambda t: received_tokens.append(t),
        )

        list(stream.iter_chunks())
        assert received_tokens == ["A", "B"]

    def test_error_handling(self):
        """Test graceful error handling during streaming."""
        def failing_stream():
            yield {"type": "content_block_delta", "delta": {"text": "ok"}}
            raise RuntimeError("Connection lost")

        stream = BedrockResponseStream(
            bedrock_response=failing_stream(),
            model_id="test-model",
        )

        chunks = list(stream.iter_chunks())
        error_chunks = [c for c in chunks if c.event == StreamEvent.ERROR]
        assert len(error_chunks) == 1
        assert "Connection lost" in error_chunks[0].data
        assert stream.metrics.errors == 1

    def test_stream_id_generated(self):
        """Test that a unique stream ID is generated."""
        stream = BedrockResponseStream(
            bedrock_response=[],
            model_id="test-model",
        )
        assert stream.stream_id is not None
        assert len(stream.stream_id) == 12

    def test_bytes_event_format(self):
        """Test processing raw bytes events."""
        events = [
            json.dumps({"text": "Hello from bytes"}).encode("utf-8"),
        ]

        stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="test-model",
        )

        chunks = list(stream.iter_chunks())
        token_chunks = [c for c in chunks if c.event == StreamEvent.TOKEN]
        assert len(token_chunks) == 1
        assert token_chunks[0].data == "Hello from bytes"


class TestLambdaResponseStreamWriter:
    """Tests for LambdaResponseStreamWriter."""

    def test_write_chunk(self):
        """Test writing a single chunk to the stream."""
        mock_stream = MockResponseStream()
        writer = LambdaResponseStreamWriter(mock_stream)

        chunk = StreamChunk(event=StreamEvent.TOKEN, data="Hello")
        writer.write_chunk(chunk)

        written = mock_stream.get_written_data()
        assert b"Hello" in written
        assert b"token" in written

    def test_stream_response_full(self):
        """Test streaming a complete Bedrock response."""
        mock_stream = MockResponseStream()
        writer = LambdaResponseStreamWriter(mock_stream)

        events = [
            {"type": "content_block_delta", "delta": {"text": "Hi"}},
            {"type": "content_block_delta", "delta": {"text": " there"}},
        ]

        bedrock_stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="test-model",
        )

        metrics = writer.stream_response(bedrock_stream)
        assert metrics.total_tokens == 2
        assert mock_stream.closed

    def test_buffered_writing(self):
        """Test buffered writing mode."""
        mock_stream = MockResponseStream()
        writer = LambdaResponseStreamWriter(mock_stream, buffer_size=1024)

        for i in range(5):
            chunk = StreamChunk(event=StreamEvent.TOKEN, data=f"chunk{i}")
            writer.write_chunk(chunk)

        # Buffer hasn't been flushed yet (under buffer_size)
        # Force flush
        writer._flush_buffer()
        assert mock_stream.get_written_data() != b""

    def test_closed_stream_raises(self):
        """Test that writing to a closed stream raises an error."""
        mock_stream = MockResponseStream()
        writer = LambdaResponseStreamWriter(mock_stream)
        writer._close()

        with pytest.raises(RuntimeError, match="Cannot write to closed stream"):
            writer.write_chunk(StreamChunk(event=StreamEvent.TOKEN, data="test"))

    def test_on_complete_callback(self):
        """Test on_complete callback is invoked after streaming."""
        mock_stream = MockResponseStream()
        callback_metrics = []

        writer = LambdaResponseStreamWriter(
            mock_stream,
            on_complete=lambda m: callback_metrics.append(m),
        )

        events = [{"type": "content_block_delta", "delta": {"text": "done"}}]
        bedrock_stream = BedrockResponseStream(
            bedrock_response=events,
            model_id="test-model",
        )

        writer.stream_response(bedrock_stream)
        assert len(callback_metrics) == 1
        assert callback_metrics[0].total_tokens == 1


class TestStreamingHandlerDecorator:
    """Tests for the @streaming_handler decorator."""

    def test_decorator_marks_function(self):
        """Test that the decorator marks the function as streaming."""
        @streaming_handler(model="test-model")
        def my_handler(event, context):
            return {"streaming": True, "prompt": "Hello"}

        assert hasattr(my_handler, "_lambdallm_response_stream")
        assert my_handler._lambdallm_response_stream is True

    def test_non_streaming_result(self):
        """Test handler that returns a non-streaming result."""
        @streaming_handler(model="test-model")
        def my_handler(event, context):
            return {"result": "direct response"}

        mock_stream = MockResponseStream()
        mock_context = MagicMock()

        my_handler({}, mock_stream, mock_context)
        assert mock_stream.closed
        written = mock_stream.get_written_data()
        assert b"direct response" in written

    def test_bedrock_stream_result(self):
        """Test handler that returns a BedrockResponseStream directly."""
        events = [
            {"type": "content_block_delta", "delta": {"text": "streamed"}},
        ]

        @streaming_handler(model="test-model")
        def my_handler(event, context):
            return BedrockResponseStream(
                bedrock_response=events,
                model_id="test-model",
            )

        mock_stream = MockResponseStream()
        mock_context = MagicMock()

        my_handler({}, mock_stream, mock_context)
        assert mock_stream.closed
        assert b"streamed" in mock_stream.get_written_data()

    def test_handler_error_handling(self):
        """Test that handler errors are caught and streamed."""
        @streaming_handler(model="test-model")
        def my_handler(event, context):
            raise ValueError("Something went wrong")

        mock_stream = MockResponseStream()
        mock_context = MagicMock()

        my_handler({}, mock_stream, mock_context)
        assert mock_stream.closed
        assert b"Something went wrong" in mock_stream.get_written_data()
