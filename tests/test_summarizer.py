"""Tests for conversation summarization for context window management."""

from __future__ import annotations

import pytest

from lambdallm.state.summarizer import ConversationSummarizer, SummaryResult


def _make_messages(count: int, role_cycle: bool = True) -> list:
    messages = []
    for i in range(count):
        role = "user" if (i % 2 == 0) else "assistant"
        messages.append({
            "role": role,
            "content": f"This is message number {i}. It contains some important information about the topic. The user needs to know this fact for future reference.",
        })
    return messages


class TestSummarizerInit:
    def test_defaults(self):
        s = ConversationSummarizer()
        assert s.max_tokens == 200000
        assert s.summarization_count == 0

    def test_threshold_tokens(self):
        s = ConversationSummarizer(max_tokens=100000, summary_threshold=0.8)
        assert s.threshold_tokens == 80000

    def test_invalid_threshold(self):
        with pytest.raises(ValueError):
            ConversationSummarizer(summary_threshold=1.5)

    def test_invalid_preserve_n(self):
        with pytest.raises(ValueError):
            ConversationSummarizer(preserve_last_n=1)


class TestTokenEstimation:
    def test_empty_messages(self):
        s = ConversationSummarizer()
        assert s.estimate_tokens([]) == 3

    def test_increases_with_content(self):
        s = ConversationSummarizer()
        short = s.estimate_tokens([{"role": "user", "content": "Hi"}])
        long = s.estimate_tokens([{"role": "user", "content": "Hi " * 100}])
        assert long > short

    def test_multiple_messages(self):
        s = ConversationSummarizer()
        msgs = _make_messages(10)
        tokens = s.estimate_tokens(msgs)
        assert tokens > 50


class TestShouldSummarize:
    def test_short_conversation_no_summary(self):
        s = ConversationSummarizer(max_tokens=200000, summary_threshold=0.8)
        msgs = _make_messages(5)
        assert s.should_summarize(msgs) is False

    def test_long_conversation_triggers_summary(self):
        s = ConversationSummarizer(max_tokens=100, summary_threshold=0.1)
        msgs = _make_messages(20)
        assert s.should_summarize(msgs) is True


class TestMaybeSummarize:
    def test_no_summarize_when_short(self):
        s = ConversationSummarizer(max_tokens=200000)
        msgs = _make_messages(5)
        result_msgs, result = s.maybe_summarize(msgs)
        assert result.was_summarized is False
        assert result.tokens_saved == 0
        assert len(result_msgs) == len(msgs)

    def test_summarize_when_long(self):
        s = ConversationSummarizer(max_tokens=100, summary_threshold=0.1, preserve_last_n=2)
        msgs = _make_messages(20)
        result_msgs, result = s.maybe_summarize(msgs)
        assert result.was_summarized is True
        assert len(result_msgs) < len(msgs)

    def test_tokens_saved_positive(self):
        s = ConversationSummarizer(max_tokens=100, summary_threshold=0.1, preserve_last_n=2)
        msgs = _make_messages(20)
        _, result = s.maybe_summarize(msgs)
        assert result.tokens_saved > 0

    def test_summarization_count_increments(self):
        s = ConversationSummarizer(max_tokens=100, summary_threshold=0.1, preserve_last_n=2)
        msgs = _make_messages(20)
        s.maybe_summarize(msgs)
        assert s.summarization_count == 1


class TestSummarize:
    def test_preserves_last_n_messages(self):
        s = ConversationSummarizer(preserve_last_n=4)
        msgs = _make_messages(20)
        result_msgs, result = s.summarize(msgs)
        assert result.was_summarized is True
        # Last 4 non-system messages should be preserved verbatim
        convo = [m for m in result_msgs if "[CONVERSATION SUMMARY" not in m.get("content", "")]
        assert len([m for m in convo if m["role"] != "system"]) == 4

    def test_system_messages_preserved(self):
        s = ConversationSummarizer(preserve_last_n=4)
        msgs = [
            {"role": "system", "content": "You are helpful."},
        ] + _make_messages(15)
        result_msgs, result = s.summarize(msgs)
        system_msgs = [m for m in result_msgs if m["role"] == "system" and "CONVERSATION SUMMARY" not in m.get("content", "")]
        assert len(system_msgs) >= 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_summary_message_injected(self):
        s = ConversationSummarizer(preserve_last_n=4)
        msgs = _make_messages(20)
        result_msgs, result = s.summarize(msgs)
        summary_msgs = [m for m in result_msgs if "CONVERSATION SUMMARY" in m.get("content", "")]
        assert len(summary_msgs) == 1

    def test_result_has_strategy(self):
        s = ConversationSummarizer(preserve_last_n=4)
        msgs = _make_messages(20)
        _, result = s.summarize(msgs)
        assert result.strategy in ("extractive", "llm", "none")

    def test_few_messages_not_summarized(self):
        s = ConversationSummarizer(preserve_last_n=6)
        msgs = _make_messages(4)
        result_msgs, result = s.summarize(msgs)
        assert result.was_summarized is False

    def test_original_and_final_counts(self):
        s = ConversationSummarizer(preserve_last_n=4)
        msgs = _make_messages(20)
        _, result = s.summarize(msgs)
        assert result.original_message_count == 20
        assert result.final_message_count < 20


class TestLLMSummarizer:
    def test_llm_invoker_called(self):
        called_with = []

        def mock_invoker(prompt: str) -> str:
            called_with.append(prompt)
            return "LLM-generated summary of the conversation."

        s = ConversationSummarizer(
            preserve_last_n=2,
            summary_invoker=mock_invoker,
        )
        msgs = _make_messages(10)
        _, result = s.summarize(msgs)
        assert len(called_with) == 1
        assert result.strategy == "llm"
        assert "LLM-generated summary" in result.summary_text

    def test_fallback_on_invoker_error(self):
        def failing_invoker(prompt: str) -> str:
            raise RuntimeError("LLM down")

        s = ConversationSummarizer(
            preserve_last_n=2,
            summary_invoker=failing_invoker,
        )
        msgs = _make_messages(10)
        result_msgs, result = s.summarize(msgs)
        # Should fall back to extractive
        assert result.was_summarized is True
        assert result.summary_text != ""


class TestSummaryResult:
    def test_result_fields(self):
        result = SummaryResult(
            original_message_count=20,
            final_message_count=8,
            original_token_estimate=5000,
            final_token_estimate=1200,
            tokens_saved=3800,
            summary_text="Key facts from prior conversation.",
            was_summarized=True,
            strategy="extractive",
        )
        assert result.tokens_saved == 3800
        assert result.was_summarized is True
        assert result.strategy == "extractive"
