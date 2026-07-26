"""Token counting utility with tiktoken-compatible interface.

Provides accurate pre-call token estimation for cost prediction and
limit enforcement. Uses character-based heuristics when tiktoken is
unavailable, with the same API surface for drop-in compatibility.

Usage:
    from lambdallm.core.token_counter import TokenCounter, count_tokens

    counter = TokenCounter(model="claude-3-haiku")
    n = counter.count("Hello, world!")
    print(f"Tokens: {n}")

    # Count a conversation
    messages = [
        {"role": "user", "content": "Summarize this document: ..."},
        {"role": "assistant", "content": "Here is the summary..."},
    ]
    n = counter.count_messages(messages)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


# Approximate chars-per-token for English text by model family
_MODEL_CHARS_PER_TOKEN: Dict[str, float] = {
    "claude": 3.8,
    "titan": 4.0,
    "llama": 3.9,
    "mistral": 3.8,
    "gpt": 4.0,
    "default": 4.0,
}

# Overhead tokens per message role in conversation format
_ROLE_OVERHEAD: Dict[str, int] = {
    "system": 4,
    "user": 3,
    "assistant": 3,
}

# Token limits per model (input context window)
_MODEL_TOKEN_LIMITS: Dict[str, int] = {
    "claude-3-opus": 200000,
    "claude-3.5-sonnet": 200000,
    "claude-3-sonnet": 200000,
    "claude-3-haiku": 200000,
    "claude-3.5-haiku": 200000,
    "claude-2": 100000,
    "amazon.titan-text-lite-v1": 4096,
    "amazon.titan-text-express-v1": 8192,
    "llama-3-70b": 8192,
    "llama-3-8b": 8192,
    "gpt-4o": 128000,
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
}


@dataclass
class TokenCount:
    """Detailed token count result."""

    total: int
    prompt_tokens: int
    completion_tokens: int = 0
    model: str = ""
    method: str = "heuristic"  # "tiktoken" or "heuristic"
    within_limit: bool = True
    limit: int = 0
    utilization_percent: float = 0.0


class TokenCounter:
    """Counts tokens with tiktoken-compatible interface.

    Attempts to use tiktoken for exact counts when available.
    Falls back to character-based heuristics when not installed.
    The API is compatible with tiktoken's encoding interface.

    Args:
        model: Target model identifier.
        fallback_chars_per_token: Characters-per-token ratio for fallback.
    """

    def __init__(
        self,
        model: str = "claude-3-haiku",
        fallback_chars_per_token: Optional[float] = None,
    ):
        self._model = model
        self._chars_per_token = fallback_chars_per_token or self._get_chars_per_token(model)
        self._tiktoken_enc = self._load_tiktoken(model)
        self._method = "tiktoken" if self._tiktoken_enc else "heuristic"

    @property
    def model(self) -> str:
        """The configured model name."""
        return self._model

    @property
    def method(self) -> str:
        """Counting method: 'tiktoken' or 'heuristic'."""
        return self._method

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs (tiktoken-compatible interface).

        Args:
            text: Text to encode.

        Returns:
            List of token IDs. Uses heuristic IDs when tiktoken unavailable.
        """
        if self._tiktoken_enc:
            return self._tiktoken_enc.encode(text)

        # Heuristic: generate pseudo token IDs based on word splits
        tokens = self._heuristic_tokenize(text)
        return list(range(len(tokens)))

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text (tiktoken-compatible).

        Args:
            tokens: Token IDs to decode.

        Returns:
            Decoded text string.
        """
        if self._tiktoken_enc:
            return self._tiktoken_enc.decode(tokens)
        return f"[{len(tokens)} tokens]"

    def count(self, text: str) -> int:
        """Count tokens in a text string.

        Args:
            text: Text to count tokens for.

        Returns:
            Estimated token count.
        """
        if not text:
            return 0

        if self._tiktoken_enc:
            return len(self._tiktoken_enc.encode(text))

        return self._heuristic_count(text)

    def count_messages(self, messages: List[Dict[str, Any]]) -> TokenCount:
        """Count tokens in a conversation message list.

        Accounts for role overhead and conversation structure.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            TokenCount with detailed breakdown.
        """
        total = 3  # Base conversation overhead

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            # Add role overhead
            total += _ROLE_OVERHEAD.get(role, 3)

            # Count content tokens
            if isinstance(content, str):
                total += self.count(content)
            elif isinstance(content, list):
                # Handle multi-modal content blocks
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self.count(block.get("text", ""))
                        elif block.get("type") == "image":
                            total += 1000  # Approximate image token cost
                    elif isinstance(block, str):
                        total += self.count(block)

        limit = _MODEL_TOKEN_LIMITS.get(self._model, 200000)
        utilization = total / limit * 100 if limit > 0 else 0.0

        return TokenCount(
            total=total,
            prompt_tokens=total,
            completion_tokens=0,
            model=self._model,
            method=self._method,
            within_limit=total <= limit,
            limit=limit,
            utilization_percent=utilization,
        )

    def count_with_limit(self, text: str) -> TokenCount:
        """Count tokens and check against model context limit.

        Args:
            text: Text to count.

        Returns:
            TokenCount with limit check results.
        """
        n = self.count(text)
        limit = _MODEL_TOKEN_LIMITS.get(self._model, 200000)
        utilization = n / limit * 100 if limit > 0 else 0.0

        return TokenCount(
            total=n,
            prompt_tokens=n,
            model=self._model,
            method=self._method,
            within_limit=n <= limit,
            limit=limit,
            utilization_percent=utilization,
        )

    def truncate(self, text: str, max_tokens: int, suffix: str = "...") -> str:
        """Truncate text to fit within a token limit.

        Args:
            text: Text to truncate.
            max_tokens: Maximum allowed tokens.
            suffix: String to append when truncated.

        Returns:
            Truncated text within token limit.
        """
        if self.count(text) <= max_tokens:
            return text

        suffix_tokens = self.count(suffix)
        target = max_tokens - suffix_tokens

        if self._tiktoken_enc:
            tokens = self._tiktoken_enc.encode(text)
            truncated = self._tiktoken_enc.decode(tokens[:target])
            return truncated + suffix

        # Heuristic truncation
        target_chars = int(target * self._chars_per_token)
        return text[:target_chars] + suffix

    def estimate_cost(
        self,
        input_text: str,
        output_tokens: int = 0,
        input_price_per_1k: float = 0.00025,
        output_price_per_1k: float = 0.00125,
    ) -> float:
        """Estimate cost for a call based on token counts.

        Args:
            input_text: The prompt text.
            output_tokens: Expected output tokens.
            input_price_per_1k: Input price per 1K tokens.
            output_price_per_1k: Output price per 1K tokens.

        Returns:
            Estimated cost in USD.
        """
        input_tokens = self.count(input_text)
        input_cost = (input_tokens / 1000) * input_price_per_1k
        output_cost = (output_tokens / 1000) * output_price_per_1k
        return input_cost + output_cost

    def _heuristic_count(self, text: str) -> int:
        """Heuristic token count using character ratio."""
        if not text:
            return 0

        # More accurate heuristic: count words + punctuation
        # Average English word is ~1.3 tokens
        words = len(re.findall(r'\S+', text))
        # Count special chars that typically become their own tokens
        special = len(re.findall(r'[^\w\s]', text))

        # Estimate: words * 1.3 + some overhead for punctuation
        estimated = int(words * 1.3) + max(0, special - words // 4)
        # Cross-check with char ratio
        char_based = max(1, int(len(text) / self._chars_per_token))

        # Take the average of both estimates
        return max(1, (estimated + char_based) // 2)

    def _heuristic_tokenize(self, text: str) -> List[str]:
        """Simple whitespace/punctuation tokenizer for heuristic counting."""
        # Split on whitespace and punctuation boundaries
        return re.findall(r'\w+|[^\w\s]', text)

    def _get_chars_per_token(self, model: str) -> float:
        """Get chars-per-token ratio for a model family."""
        model_lower = model.lower()
        for family, ratio in _MODEL_CHARS_PER_TOKEN.items():
            if family in model_lower:
                return ratio
        return _MODEL_CHARS_PER_TOKEN["default"]

    def _load_tiktoken(self, model: str):
        """Attempt to load tiktoken encoder for the model."""
        try:
            import tiktoken  # type: ignore

            # Map model to tiktoken encoding
            if "claude" in model.lower() or "titan" in model.lower():
                # Claude uses cl100k_base approximation
                return tiktoken.get_encoding("cl100k_base")
            elif "gpt-4" in model.lower():
                return tiktoken.encoding_for_model("gpt-4")
            elif "gpt-3.5" in model.lower():
                return tiktoken.encoding_for_model("gpt-3.5-turbo")
            else:
                return tiktoken.get_encoding("cl100k_base")
        except (ImportError, KeyError, Exception):
            return None


def count_tokens(text: str, model: str = "claude-3-haiku") -> int:
    """Convenience function to count tokens for a text string.

    Args:
        text: Text to count.
        model: Target model for accurate estimation.

    Returns:
        Estimated token count.
    """
    return TokenCounter(model=model).count(text)


def count_message_tokens(
    messages: List[Dict[str, Any]],
    model: str = "claude-3-haiku",
) -> TokenCount:
    """Convenience function to count tokens in a message list.

    Args:
        messages: Conversation messages.
        model: Target model.

    Returns:
        TokenCount with full breakdown.
    """
    return TokenCounter(model=model).count_messages(messages)
