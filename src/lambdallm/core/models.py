"""Model definitions and registry for LambdaLLM."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Model(str, Enum):
    """Supported model identifiers with sensible defaults."""

    CLAUDE_3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_3_5_SONNET = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    TITAN_TEXT_EXPRESS = "amazon.titan-text-express-v1"
    TITAN_TEXT_LITE = "amazon.titan-text-lite-v1"
    LLAMA3_8B = "meta.llama3-8b-instruct-v1:0"
    LLAMA3_70B = "meta.llama3-70b-instruct-v1:0"


@dataclass
class ModelConfig:
    """Configuration for a specific model invocation."""

    model_id: str
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: list[str] = field(default_factory=list)
    region: Optional[str] = None

    @classmethod
    def from_model(cls, model: Model, **kwargs) -> "ModelConfig":
        """Create config from a Model enum with optional overrides."""
        return cls(model_id=model.value, **kwargs)


@dataclass
class ModelResponse:
    """Standardized response from any model provider."""

    content: str
    model_id: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    cost_usd: float = 0.0
    raw_response: Optional[dict] = None

    @property
    def total_tokens(self) -> int:
        return self.tokens_in + self.tokens_out
