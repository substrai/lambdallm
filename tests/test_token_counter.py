"""Tests for token counting utility with tiktoken-compatible interface."""

from __future__ import annotations

import pytest

from lambdallm.core.token_counter import (
    TokenCount,
    TokenCounter,
    count_message_tokens,
    count_tokens,
    _MODEL_TOKEN_LIMITS,
)


class TestTokenCounterInit:
    def test_default_model(self):
        counter = TokenCounter()
        assert counter.model == "claude-3-haiku"

    def test_custom_model(self):
        counter = TokenCounter(model="claude-3-opus")
        assert counter.model == "claude-3-opus"

    def test_method_is_heuristic_or_tiktoken(self):
        counter = TokenCounter()
        assert counter.method in ("heuristic", "tiktoken")


class TestTokenCounting:
    def test_empty_string(self):
        counter = TokenCounter()
        assert counter.count("") == 0

    def test_short_text(self):
        counter = TokenCounter()
        n = counter.count("Hello, world!")
        assert n >= 3 and n <= 8

    def test_longer_text_more_tokens(self):
        counter = TokenCounter()
        short = counter.count("Hello")
        long = counter.count("Hello world this is a longer sentence with many words in it for testing purposes")
        assert long > short

    def test_whitespace_only(self):
        counter = TokenCounter()
        n = counter.count("   ")
        assert n >= 0

    def test_consistent_results(self):
        counter = TokenCounter()
        text = "The quick brown fox jumps over the lazy dog"
        assert counter.count(text) == counter.count(text)

    def test_unicode_text(self):
        counter = TokenCounter()
        n = counter.count("こんにちは世界")
        assert n >= 1


class TestMessageCounting:
    def test_single_user_message(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "What is machine learning?"}]
        result = counter.count_messages(messages)
        assert result.total > 0
        assert result.prompt_tokens == result.total

    def test_conversation_more_tokens(self):
        counter = TokenCounter()
        single = counter.count_messages([
            {"role": "user", "content": "Hi"}
        ])
        convo = counter.count_messages([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is machine learning?"},
            {"role": "assistant", "content": "Machine learning is a field of AI."},
            {"role": "user", "content": "Can you give an example?"},
        ])
        assert convo.total > single.total

    def test_within_limit(self):
        counter = TokenCounter(model="claude-3-haiku")
        messages = [{"role": "user", "content": "Short message"}]
        result = counter.count_messages(messages)
        assert result.within_limit is True
        assert result.limit == 200000

    def test_utilization_percent(self):
        counter = TokenCounter()
        messages = [{"role": "user", "content": "Hello"}]
        result = counter.count_messages(messages)
        assert 0.0 <= result.utilization_percent <= 1.0

    def test_model_stored_in_result(self):
        counter = TokenCounter(model="claude-3-opus")
        result = counter.count_messages([{"role": "user", "content": "test"}])
        assert result.model == "claude-3-opus"

    def test_multimodal_content_list(self):
        counter = TokenCounter()
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image:"},
                {"type": "image", "source": {"type": "url", "url": "https://example.com/img.jpg"}},
            ]
        }]
        result = counter.count_messages(messages)
        assert result.total > 1000  # Image adds ~1000 tokens


class TestCountWithLimit:
    def test_small_text_within_limit(self):
        counter = TokenCounter(model="claude-3-haiku")
        result = counter.count_with_limit("Hello world")
        assert result.within_limit is True

    def test_returns_token_count(self):
        counter = TokenCounter()
        result = counter.count_with_limit("The quick brown fox")
        assert result.total > 0

    def test_limit_set_correctly(self):
        counter = TokenCounter(model="claude-3-opus")
        result = counter.count_with_limit("test")
        assert result.limit == 200000


class TestTruncation:
    def test_short_text_not_truncated(self):
        counter = TokenCounter()
        text = "Hello"
        result = counter.truncate(text, max_tokens=100)
        assert result == text

    def test_long_text_truncated(self):
        counter = TokenCounter(fallback_chars_per_token=4.0)
        text = "word " * 500  # ~625 tokens at 4 chars/token
        result = counter.truncate(text, max_tokens=50)
        assert counter.count(result) <= 55  # Allow small margin

    def test_truncated_has_suffix(self):
        counter = TokenCounter(fallback_chars_per_token=4.0)
        text = "a " * 1000
        result = counter.truncate(text, max_tokens=20, suffix="[TRUNCATED]")
        assert "[TRUNCATED]" in result


class TestCostEstimation:
    def test_estimate_cost_positive(self):
        counter = TokenCounter()
        cost = counter.estimate_cost("Hello world", output_tokens=100)
        assert cost > 0

    def test_longer_input_higher_cost(self):
        counter = TokenCounter()
        cost_short = counter.estimate_cost("Hi", output_tokens=50)
        cost_long = counter.estimate_cost("Hello world " * 100, output_tokens=50)
        assert cost_long > cost_short

    def test_custom_prices(self):
        counter = TokenCounter()
        cost = counter.estimate_cost(
            "test", output_tokens=0,
            input_price_per_1k=0.01,
            output_price_per_1k=0.03,
        )
        assert cost > 0


class TestConvenienceFunctions:
    def test_count_tokens(self):
        n = count_tokens("Hello world")
        assert n > 0

    def test_count_tokens_custom_model(self):
        n = count_tokens("test", model="claude-3-opus")
        assert n > 0

    def test_count_message_tokens(self):
        result = count_message_tokens([{"role": "user", "content": "Hi"}])
        assert isinstance(result, TokenCount)
        assert result.total > 0


class TestModelLimits:
    def test_known_models_have_limits(self):
        assert "claude-3-haiku" in _MODEL_TOKEN_LIMITS
        assert "claude-3-opus" in _MODEL_TOKEN_LIMITS
        assert "gpt-4o" in _MODEL_TOKEN_LIMITS

    def test_limits_are_positive(self):
        for model, limit in _MODEL_TOKEN_LIMITS.items():
            assert limit > 0
