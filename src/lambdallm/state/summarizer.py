"""Conversation summarization for context window management.

Auto-summarizes old messages when approaching the context limit,
preserving key facts while reducing token count. Enables long-running
conversations to stay within Lambda's memory and model context constraints.

Usage:
    from lambdallm.state.summarizer import ConversationSummarizer

    summarizer = ConversationSummarizer(
        max_tokens=150000,
        summary_threshold=0.80,  # Summarize at 80% capacity
        preserve_last_n=6,       # Always keep last 6 messages intact
    )

    messages, was_summarized = summarizer.maybe_summarize(messages)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class SummaryResult:
    """Result of a conversation summarization."""

    original_message_count: int
    final_message_count: int
    original_token_estimate: int
    final_token_estimate: int
    tokens_saved: int
    summary_text: str
    was_summarized: bool
    strategy: str  # "extractive" or "llm"


class ConversationSummarizer:
    """Summarizes conversation history to fit within context windows.

    When a conversation approaches the token limit, older messages are
    summarized into a compact representation while preserving the most
    recent messages for coherence.

    Args:
        max_tokens: Model context window size.
        summary_threshold: Fraction of max_tokens that triggers summarization.
        preserve_last_n: Number of recent messages to always keep verbatim.
        chars_per_token: Chars-per-token estimate for token counting.
        summary_invoker: Optional async LLM function for abstractive summaries.
    """

    def __init__(
        self,
        max_tokens: int = 200000,
        summary_threshold: float = 0.80,
        preserve_last_n: int = 6,
        chars_per_token: float = 4.0,
        summary_invoker: Optional[Callable[[str], str]] = None,
    ):
        if not 0.1 <= summary_threshold <= 0.99:
            raise ValueError("summary_threshold must be between 0.1 and 0.99")
        if preserve_last_n < 2:
            raise ValueError("preserve_last_n must be at least 2")

        self._max_tokens = max_tokens
        self._threshold = summary_threshold
        self._preserve_n = preserve_last_n
        self._chars_per_token = chars_per_token
        self._invoker = summary_invoker
        self._summarization_count = 0

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def threshold_tokens(self) -> int:
        """Token count at which summarization is triggered."""
        return int(self._max_tokens * self._threshold)

    @property
    def summarization_count(self) -> int:
        """How many times summarization has been applied."""
        return self._summarization_count

    def estimate_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate token count for a message list."""
        total = 3  # Base overhead
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += max(1, int(len(content) / self._chars_per_token)) + 3
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text", "")
                        total += max(1, int(len(text) / self._chars_per_token))
        return total

    def should_summarize(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if summarization is needed.

        Args:
            messages: Current conversation messages.

        Returns:
            True if token count exceeds the threshold.
        """
        return self.estimate_tokens(messages) > self.threshold_tokens

    def maybe_summarize(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], SummaryResult]:
        """Summarize messages only if threshold is exceeded.

        Args:
            messages: Current conversation messages.

        Returns:
            Tuple of (possibly summarized messages, SummaryResult).
        """
        original_tokens = self.estimate_tokens(messages)
        original_count = len(messages)

        if not self.should_summarize(messages):
            return messages, SummaryResult(
                original_message_count=original_count,
                final_message_count=original_count,
                original_token_estimate=original_tokens,
                final_token_estimate=original_tokens,
                tokens_saved=0,
                summary_text="",
                was_summarized=False,
                strategy="none",
            )

        return self.summarize(messages)

    def summarize(
        self,
        messages: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], SummaryResult]:
        """Force summarization of old messages.

        Splits messages into:
        - Head: System message (preserved verbatim)
        - Middle: Older messages to summarize
        - Tail: Last N messages (preserved verbatim)

        Args:
            messages: Conversation messages to summarize.

        Returns:
            Tuple of (compressed messages, SummaryResult).
        """
        original_tokens = self.estimate_tokens(messages)
        original_count = len(messages)

        # Separate system message(s) from conversation
        system_messages = [m for m in messages if m.get("role") == "system"]
        convo_messages = [m for m in messages if m.get("role") != "system"]

        if len(convo_messages) <= self._preserve_n:
            # Not enough messages to summarize
            return messages, SummaryResult(
                original_message_count=original_count,
                final_message_count=original_count,
                original_token_estimate=original_tokens,
                final_token_estimate=original_tokens,
                tokens_saved=0,
                summary_text="",
                was_summarized=False,
                strategy="none",
            )

        # Split: old messages go into summary, recent kept verbatim
        to_summarize = convo_messages[:-self._preserve_n]
        to_keep = convo_messages[-self._preserve_n:]

        # Build summary
        if self._invoker:
            summary_text = self._llm_summarize(to_summarize)
            strategy = "llm"
        else:
            summary_text = self._extractive_summarize(to_summarize)
            strategy = "extractive"

        # Build compressed message list
        summary_message: Dict[str, Any] = {
            "role": "system",
            "content": f"[CONVERSATION SUMMARY — {len(to_summarize)} earlier messages]\n{summary_text}",
        }

        compressed = system_messages + [summary_message] + to_keep

        final_tokens = self.estimate_tokens(compressed)
        self._summarization_count += 1

        return compressed, SummaryResult(
            original_message_count=original_count,
            final_message_count=len(compressed),
            original_token_estimate=original_tokens,
            final_token_estimate=final_tokens,
            tokens_saved=original_tokens - final_tokens,
            summary_text=summary_text,
            was_summarized=True,
            strategy=strategy,
        )

    def _extractive_summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Extractive summarization — preserve key facts from messages."""
        lines: List[str] = []
        key_facts: List[str] = []

        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if not isinstance(content, str):
                continue

            # Extract sentences with key information patterns
            sentences = re.split(r'[.!?]\s+', content)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 10:
                    continue
                # Score sentence importance
                score = self._sentence_importance(sentence)
                if score > 0:
                    key_facts.append(f"[{role}]: {sentence}")

        if not key_facts:
            # Fallback: take first sentence of each message
            for msg in messages:
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    first = content.split(".")[0].strip()
                    if first:
                        role = msg.get("role", "?")
                        key_facts.append(f"[{role}]: {first}")

        # Limit summary length
        summary = " | ".join(key_facts[:15])
        return summary if summary else f"Prior conversation with {len(messages)} messages."

    def _sentence_importance(self, sentence: str) -> int:
        """Score sentence importance for extractive summarization."""
        score = 0
        text = sentence.lower()

        # Key information markers
        importance_patterns = [
            r'\b(is|are|was|were|will be|can be|should be)\b',
            r'\b(requirement|must|need|important|key|critical|note)\b',
            r'\b(because|therefore|however|but|although|since)\b',
            r'\b(first|second|finally|in summary|to summarize)\b',
            r'\b(error|problem|issue|bug|fix|solution|answer)\b',
            r'\b(user|customer|system|api|service|database)\b',
        ]

        for pattern in importance_patterns:
            if re.search(pattern, text):
                score += 1

        # Prefer longer, more informative sentences
        if len(sentence.split()) > 8:
            score += 1

        return score

    def _llm_summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Use LLM invoker for abstractive summarization."""
        conversation_text = "\n".join(
            f"{m.get('role', '?').upper()}: {m.get('content', '')}"
            for m in messages
            if isinstance(m.get("content"), str)
        )

        prompt = (
            "Summarize the following conversation, preserving all key facts, "
            "decisions, user preferences, and important context. "
            "Be concise but complete.\n\n"
            f"CONVERSATION:\n{conversation_text}\n\n"
            "SUMMARY:"
        )

        try:
            return self._invoker(prompt)  # type: ignore
        except Exception:
            return self._extractive_summarize(messages)
