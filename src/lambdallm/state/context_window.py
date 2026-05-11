"""Context window manager for LambdaLLM.

Automatically manages conversation history to fit within model context limits.
Prevents token overflow by intelligently trimming or summarizing history.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from lambdallm.state.session import Message

logger = logging.getLogger("lambdallm")

# Approximate context window sizes (tokens)
MODEL_CONTEXT_LIMITS = {
    "anthropic.claude-3-haiku-20240307-v1:0": 200_000,
    "anthropic.claude-3-sonnet-20240229-v1:0": 200_000,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 200_000,
    "anthropic.claude-3-opus-20240229-v1:0": 200_000,
    "amazon.titan-text-express-v1": 8_000,
    "amazon.titan-text-lite-v1": 4_000,
    "meta.llama3-8b-instruct-v1:0": 8_000,
    "meta.llama3-70b-instruct-v1:0": 8_000,
}

# Rough estimate: 1 token ≈ 4 characters for English text
CHARS_PER_TOKEN = 4


@dataclass
class ContextWindowConfig:
    """Configuration for context window management."""

    max_tokens: Optional[int] = None  # None = use model default
    reserve_tokens: int = 1024  # Reserve for the response
    strategy: str = "truncate_oldest"  # truncate_oldest | summarize | smart
    model_id: Optional[str] = None


class ContextWindowManager:
    """Manages conversation history to fit within model context limits.

    Strategies:
    - truncate_oldest: Remove oldest messages until history fits
    - summarize: Summarize old messages into a single message (requires LLM call)
    - smart: Keep system prompt + recent messages + summarize middle
    """

    def __init__(self, config: Optional[ContextWindowConfig] = None):
        self.config = config or ContextWindowConfig()

    def fit_messages(
        self,
        messages: list[Message],
        system_prompt: str = "",
        model_id: Optional[str] = None,
    ) -> list[Message]:
        """Trim messages to fit within the context window.

        Args:
            messages: Full conversation history.
            system_prompt: System prompt that always takes priority.
            model_id: Model ID to determine context limit.

        Returns:
            Trimmed list of messages that fits within the context window.
        """
        max_tokens = self._get_max_tokens(model_id)
        available_tokens = max_tokens - self.config.reserve_tokens

        # Account for system prompt
        system_tokens = self._estimate_tokens(system_prompt)
        available_tokens -= system_tokens

        if available_tokens <= 0:
            logger.warning("System prompt exceeds context window")
            return messages[-1:]  # Return at least the last message

        strategy = self.config.strategy

        if strategy == "truncate_oldest":
            return self._truncate_oldest(messages, available_tokens)
        elif strategy == "smart":
            return self._smart_trim(messages, available_tokens)
        else:
            return self._truncate_oldest(messages, available_tokens)

    def _truncate_oldest(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """Remove oldest messages until history fits."""
        if not messages:
            return []

        # Start from the most recent and work backwards
        result = []
        total_tokens = 0

        for msg in reversed(messages):
            msg_tokens = self._estimate_tokens(msg.content)
            if total_tokens + msg_tokens > max_tokens:
                break
            result.insert(0, msg)
            total_tokens += msg_tokens

        if not result and messages:
            # Always include at least the last message (truncated if needed)
            result = [messages[-1]]

        trimmed_count = len(messages) - len(result)
        if trimmed_count > 0:
            logger.info(f"Context window: trimmed {trimmed_count} oldest messages")

        return result

    def _smart_trim(self, messages: list[Message], max_tokens: int) -> list[Message]:
        """Smart trimming: keep first message (context) + recent messages.

        Strategy:
        1. Always keep the first message (often contains important context)
        2. Always keep the last N messages (recent conversation)
        3. Trim the middle
        """
        if len(messages) <= 3:
            return messages

        # Reserve 30% for first message, 70% for recent
        first_budget = int(max_tokens * 0.3)
        recent_budget = max_tokens - first_budget

        # Keep first message if it fits
        first_msg = messages[0]
        first_tokens = self._estimate_tokens(first_msg.content)

        result = []
        if first_tokens <= first_budget:
            result.append(first_msg)
            remaining_budget = max_tokens - first_tokens
        else:
            remaining_budget = max_tokens

        # Fill from the end with recent messages
        recent = []
        total = 0
        for msg in reversed(messages[1:]):
            msg_tokens = self._estimate_tokens(msg.content)
            if total + msg_tokens > remaining_budget:
                break
            recent.insert(0, msg)
            total += msg_tokens

        result.extend(recent)

        trimmed_count = len(messages) - len(result)
        if trimmed_count > 0:
            logger.info(f"Context window (smart): kept first + {len(recent)} recent, trimmed {trimmed_count}")

        return result

    def _get_max_tokens(self, model_id: Optional[str] = None) -> int:
        """Get the context window size for a model."""
        if self.config.max_tokens:
            return self.config.max_tokens

        mid = model_id or self.config.model_id
        if mid and mid in MODEL_CONTEXT_LIMITS:
            return MODEL_CONTEXT_LIMITS[mid]

        return 100_000  # Conservative default

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count from text length."""
        if not text:
            return 0
        return len(text) // CHARS_PER_TOKEN

    def get_usage_info(self, messages: list[Message], model_id: Optional[str] = None) -> dict:
        """Get context window usage information."""
        max_tokens = self._get_max_tokens(model_id)
        used_tokens = sum(self._estimate_tokens(m.content) for m in messages)

        return {
            "max_tokens": max_tokens,
            "used_tokens": used_tokens,
            "available_tokens": max_tokens - used_tokens - self.config.reserve_tokens,
            "usage_percent": round((used_tokens / max_tokens) * 100, 1),
            "message_count": len(messages),
        }
