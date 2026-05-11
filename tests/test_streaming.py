"""Tests for streaming support."""

import pytest
from lambdallm.core.streaming import StreamingResponse, StreamingHandler


class TestStreamingResponse:
    def test_basic_streaming(self):
        """StreamingResponse should yield chunks."""
        def gen():
            yield "Hello "
            yield "World"

        stream = StreamingResponse(gen())
        chunks = list(stream)

        assert chunks == ["Hello ", "World"]

    def test_full_text(self):
        """full_text should return complete response."""
        def gen():
            yield "Hello "
            yield "World"

        stream = StreamingResponse(gen())
        assert stream.full_text == "Hello World"

    def test_lambda_stream_format(self):
        """to_lambda_stream should yield JSON chunks."""
        import json

        def gen():
            yield "Hi"

        stream = StreamingResponse(gen(), metadata={"model": "test"})
        chunks = list(stream.to_lambda_stream())

        # First chunk is metadata
        meta = json.loads(chunks[0])
        assert meta["type"] == "metadata"
        assert meta["model"] == "test"

        # Content chunk
        content = json.loads(chunks[1])
        assert content["type"] == "content"
        assert content["text"] == "Hi"

        # Complete marker
        complete = json.loads(chunks[2])
        assert complete["type"] == "complete"


class TestStreamingHandler:
    def test_detects_streaming_response(self):
        def gen():
            yield "test"

        stream = StreamingResponse(gen())
        assert StreamingHandler.is_streaming_response(stream) is True
        assert StreamingHandler.is_streaming_response({"body": "test"}) is False
