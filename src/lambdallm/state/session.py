"""Session management for multi-turn conversations.

Solves Lambda's statelessness by transparently persisting
conversation state to DynamoDB between invocations.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


class MemoryStrategy(Enum):
    """Strategies for managing conversation history within context limits."""

    FULL_HISTORY = "full_history"
    SLIDING_WINDOW = "sliding_window"
    SUMMARY = "summary"
    SEMANTIC = "semantic"

    @classmethod
    def SLIDING_WINDOW_N(cls, max_messages: int = 20):
        """Create a sliding window strategy with custom size."""
        return _SlidingWindowConfig(max_messages=max_messages)


@dataclass
class _SlidingWindowConfig:
    """Configuration for sliding window memory strategy."""
    max_messages: int = 20
    strategy: str = "sliding_window"


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user", "assistant", "system"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Message":
        return cls(
            role=data["role"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Session:
    """Conversation session with persistent state.

    Usage:
        @handler(session=Session(store="dynamodb", ttl_hours=24))
        def my_handler(event, context):
            session = context.session
            session.add_message("user", event["body"]["message"])
            response = context.invoke(session.format_history())
            session.add_message("assistant", response)
            return {"reply": response}
    """

    store: str = "dynamodb"
    ttl_hours: int = 24
    memory: Any = field(default_factory=lambda: MemoryStrategy.SLIDING_WINDOW)
    max_messages: int = 20

    # Runtime state (populated when loaded)
    session_id: Optional[str] = None
    messages: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    _dirty: bool = field(default=False, repr=False)
    _store_instance: Any = field(default=None, repr=False)

    def load(self, session_id: str) -> "Session":
        """Load session from store."""
        self.session_id = session_id

        if self._store_instance is None:
            self._store_instance = self._create_store()

        data = self._store_instance.get(session_id)
        if data:
            self.messages = [Message.from_dict(m) for m in data.get("messages", [])]
            self.metadata = data.get("metadata", {})

        return self

    def save(self) -> None:
        """Save session to store (only if modified)."""
        if not self._dirty:
            return

        if self._store_instance is None:
            self._store_instance = self._create_store()

        self._store_instance.put(
            session_id=self.session_id,
            data={
                "messages": [m.to_dict() for m in self.messages],
                "metadata": self.metadata,
            },
            ttl_seconds=self.ttl_hours * 3600,
        )
        self._dirty = False

    def add_message(self, role: str, content: str, **metadata) -> None:
        """Add a message to the conversation history."""
        self.messages.append(Message(role=role, content=content, metadata=metadata))
        self._apply_memory_strategy()
        self._dirty = True

    def get_history(self) -> list[dict]:
        """Get conversation history as list of dicts (for LLM context)."""
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def format_history(self) -> str:
        """Format conversation history as a string for prompt injection."""
        lines = []
        for msg in self.messages:
            lines.append(f"{msg.role}: {msg.content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all messages from the session."""
        self.messages = []
        self._dirty = True

    @property
    def message_count(self) -> int:
        return len(self.messages)

    @property
    def last_message(self) -> Optional[Message]:
        return self.messages[-1] if self.messages else None

    def _apply_memory_strategy(self) -> None:
        """Apply the configured memory strategy to trim history."""
        if isinstance(self.memory, _SlidingWindowConfig):
            max_msgs = self.memory.max_messages
        elif self.memory == MemoryStrategy.SLIDING_WINDOW:
            max_msgs = self.max_messages
        elif self.memory == MemoryStrategy.FULL_HISTORY:
            return  # Keep everything
        else:
            max_msgs = self.max_messages

        if len(self.messages) > max_msgs:
            self.messages = self.messages[-max_msgs:]

    def _create_store(self):
        """Create the backing store instance."""
        if self.store == "dynamodb":
            from lambdallm.state.dynamodb import DynamoDBStateStore
            return DynamoDBStateStore()
        elif self.store == "memory":
            from lambdallm.state.memory import InMemoryStateStore
            return InMemoryStateStore()
        else:
            raise ValueError(f"Unknown store: {self.store}. Use 'dynamodb' or 'memory'.")
