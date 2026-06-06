"""Tests for the request/response middleware pipeline."""

import time

import pytest

from lambdallm.middleware.pipeline import (
    LoggingMiddleware,
    Middleware,
    MiddlewarePipeline,
    PipelineContext,
    Request,
    Response,
    TransformationMiddleware,
    ValidationMiddleware,
)


class CounterMiddleware(Middleware):
    """Test middleware that counts invocations."""

    def __init__(self, name: str = "counter", priority: int = 50):
        super().__init__(name=name, priority=priority)
        self.request_count = 0
        self.response_count = 0

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        self.request_count += 1
        context.set(f"{self.name}_request_count", self.request_count)
        return request

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        self.response_count += 1
        context.set(f"{self.name}_response_count", self.response_count)
        return response


class AbortingMiddleware(Middleware):
    """Test middleware that aborts the pipeline."""

    def __init__(self, priority: int = 50):
        super().__init__(name="aborter", priority=priority)

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        context.abort("Aborted by test middleware")
        return request

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        return response


class ErrorMiddleware(Middleware):
    """Test middleware that raises an error."""

    def __init__(self, priority: int = 50):
        super().__init__(name="error_raiser", priority=priority)

    def process_request(self, request: Request, context: PipelineContext) -> Request:
        raise RuntimeError("Intentional error in request processing")

    def process_response(self, response: Response, context: PipelineContext) -> Response:
        raise RuntimeError("Intentional error in response processing")


def make_request(prompt: str = "Hello, world") -> Request:
    return Request(prompt=prompt, model="test-model")


def make_response(content: str = "Hi there") -> Response:
    return Response(content=content, model="test-model", latency_ms=42.0)


def echo_handler(request: Request) -> Response:
    return Response(content=f"Echo: {request.prompt}", model=request.model)


class TestPipelineContext:
    def test_set_and_get(self):
        ctx = PipelineContext()
        ctx.set("key", "value")
        assert ctx.get("key") == "value"

    def test_get_default(self):
        ctx = PipelineContext()
        assert ctx.get("missing", "default") == "default"

    def test_abort(self):
        ctx = PipelineContext()
        assert not ctx.is_aborted
        ctx.abort("test reason")
        assert ctx.is_aborted
        assert ctx.abort_reason == "test reason"

    def test_record_error(self):
        ctx = PipelineContext()
        ctx.record_error("test_mw", ValueError("bad value"))
        assert len(ctx.errors) == 1
        assert ctx.errors[0]["middleware"] == "test_mw"
        assert ctx.errors[0]["error_type"] == "ValueError"
        assert ctx.errors[0]["message"] == "bad value"


class TestMiddlewarePipeline:
    def test_add_and_len(self):
        pipeline = MiddlewarePipeline()
        pipeline.add(CounterMiddleware())
        assert len(pipeline) == 1
        pipeline.add(CounterMiddleware(name="second"))
        assert len(pipeline) == 2

    def test_remove_middleware(self):
        pipeline = MiddlewarePipeline()
        pipeline.add(CounterMiddleware(name="to_remove"))
        assert pipeline.remove("to_remove") is True
        assert len(pipeline) == 0
        assert pipeline.remove("nonexistent") is False

    def test_get_middleware(self):
        pipeline = MiddlewarePipeline()
        mw = CounterMiddleware(name="findme")
        pipeline.add(mw)
        assert pipeline.get("findme") is mw
        assert pipeline.get("nope") is None

    def test_priority_ordering(self):
        pipeline = MiddlewarePipeline()
        mw_high = CounterMiddleware(name="high", priority=100)
        mw_low = CounterMiddleware(name="low", priority=10)
        mw_mid = CounterMiddleware(name="mid", priority=50)
        pipeline.add(mw_high).add(mw_low).add(mw_mid)

        ordered = pipeline.middlewares
        assert ordered[0].name == "low"
        assert ordered[1].name == "mid"
        assert ordered[2].name == "high"

    def test_process_request(self):
        pipeline = MiddlewarePipeline()
        counter = CounterMiddleware()
        pipeline.add(counter)

        request = make_request()
        result, ctx = pipeline.process_request(request)
        assert counter.request_count == 1
        assert ctx.get("counter_request_count") == 1
        assert "request_start" in ctx.timestamps
        assert "request_end" in ctx.timestamps

    def test_process_response(self):
        pipeline = MiddlewarePipeline()
        counter = CounterMiddleware()
        pipeline.add(counter)

        response = make_response()
        result, ctx = pipeline.process_response(response)
        assert counter.response_count == 1
        assert ctx.get("counter_response_count") == 1

    def test_execute_full_pipeline(self):
        pipeline = MiddlewarePipeline()
        counter = CounterMiddleware()
        pipeline.add(counter)

        request = make_request("test prompt")
        response, ctx = pipeline.execute(request, echo_handler)
        assert response.content == "Echo: test prompt"
        assert counter.request_count == 1
        assert counter.response_count == 1

    def test_abort_stops_pipeline(self):
        pipeline = MiddlewarePipeline()
        aborter = AbortingMiddleware(priority=10)
        counter = CounterMiddleware(name="after_abort", priority=50)
        pipeline.add(aborter).add(counter)

        request = make_request()
        response, ctx = pipeline.execute(request, echo_handler)
        assert ctx.is_aborted
        assert counter.request_count == 0
        assert response.metadata.get("aborted") is True

    def test_error_strategy_continue(self):
        pipeline = MiddlewarePipeline(error_strategy="continue")
        pipeline.add(ErrorMiddleware(priority=10))
        counter = CounterMiddleware(name="after_error", priority=50)
        pipeline.add(counter)

        request = make_request()
        result, ctx = pipeline.process_request(request)
        assert counter.request_count == 1
        assert len(ctx.errors) == 1

    def test_error_strategy_abort(self):
        pipeline = MiddlewarePipeline(error_strategy="abort")
        pipeline.add(ErrorMiddleware(priority=10))
        counter = CounterMiddleware(name="after_error", priority=50)
        pipeline.add(counter)

        request = make_request()
        result, ctx = pipeline.process_request(request)
        assert ctx.is_aborted
        assert counter.request_count == 0

    def test_error_strategy_raise(self):
        pipeline = MiddlewarePipeline(error_strategy="raise")
        pipeline.add(ErrorMiddleware(priority=10))

        request = make_request()
        with pytest.raises(RuntimeError, match="Intentional error"):
            pipeline.process_request(request)

    def test_disabled_middleware_skipped(self):
        pipeline = MiddlewarePipeline()
        counter = CounterMiddleware()
        counter.enabled = False
        pipeline.add(counter)

        request = make_request()
        pipeline.process_request(request)
        assert counter.request_count == 0

    def test_clear(self):
        pipeline = MiddlewarePipeline()
        pipeline.add(CounterMiddleware())
        pipeline.add(CounterMiddleware(name="second"))
        pipeline.clear()
        assert len(pipeline) == 0


class TestLoggingMiddleware:
    def test_logs_request(self):
        mw = LoggingMiddleware()
        request = make_request("hello")
        ctx = PipelineContext()
        mw.process_request(request, ctx)
        assert len(mw.logs) == 1
        assert mw.logs[0]["stage"] == "request"
        assert mw.logs[0]["prompt_length"] == 5
        assert ctx.get("request_logged") is True

    def test_logs_response(self):
        mw = LoggingMiddleware()
        response = make_response("world")
        ctx = PipelineContext()
        mw.process_response(response, ctx)
        assert len(mw.logs) == 1
        assert mw.logs[0]["stage"] == "response"
        assert mw.logs[0]["content_length"] == 5
        assert ctx.get("response_logged") is True


class TestValidationMiddleware:
    def test_rejects_empty_prompt(self):
        mw = ValidationMiddleware()
        request = Request(prompt="   ")
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="cannot be empty"):
            mw.process_request(request, ctx)

    def test_rejects_oversized_prompt(self):
        mw = ValidationMiddleware(max_prompt_length=10)
        request = Request(prompt="x" * 20)
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="exceeds maximum length"):
            mw.process_request(request, ctx)

    def test_validates_required_fields(self):
        mw = ValidationMiddleware(required_fields=["temperature"])
        request = Request(prompt="test", parameters={})
        ctx = PipelineContext()
        with pytest.raises(ValueError, match="Required field missing"):
            mw.process_request(request, ctx)

    def test_passes_valid_request(self):
        mw = ValidationMiddleware(required_fields=["temperature"])
        request = Request(prompt="test", parameters={"temperature": 0.7})
        ctx = PipelineContext()
        result = mw.process_request(request, ctx)
        assert ctx.get("request_validated") is True


class TestTransformationMiddleware:
    def test_applies_request_transforms(self):
        def upper_prompt(req: Request) -> Request:
            req.prompt = req.prompt.upper()
            return req

        mw = TransformationMiddleware(request_transforms=[upper_prompt])
        request = Request(prompt="hello")
        ctx = PipelineContext()
        result = mw.process_request(request, ctx)
        assert result.prompt == "HELLO"

    def test_applies_response_transforms(self):
        def strip_content(resp: Response) -> Response:
            resp.content = resp.content.strip()
            return resp

        mw = TransformationMiddleware(response_transforms=[strip_content])
        response = Response(content="  padded  ")
        ctx = PipelineContext()
        result = mw.process_response(response, ctx)
        assert result.content == "padded"

    def test_multiple_transforms_chained(self):
        def add_prefix(req: Request) -> Request:
            req.prompt = "PREFIX: " + req.prompt
            return req

        def add_suffix(req: Request) -> Request:
            req.prompt = req.prompt + " :SUFFIX"
            return req

        mw = TransformationMiddleware(request_transforms=[add_prefix, add_suffix])
        request = Request(prompt="body")
        ctx = PipelineContext()
        result = mw.process_request(request, ctx)
        assert result.prompt == "PREFIX: body :SUFFIX"
