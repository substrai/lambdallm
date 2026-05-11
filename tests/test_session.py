"""Tests for session state management."""

import pytest
from lambdallm.state.session import Session, MemoryStrategy, Message
from lambdallm.state.memory import InMemoryStateStore


class TestSession:
    """Test session management."""

    def test_session_creation(self):
        """Session should be created with defaults."""
        session = Session(store="memory")
        assert session.store == "memory"
        assert session.ttl_hours == 24
        assert session.message_count == 0

    def test_add_message(self):
        """Should add messages to history."""
        session = Session(store="memory")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi there!")

        assert session.message_count == 2
        assert session.last_message.content == "Hi there!"
        assert session.last_message.role == "assistant"

    def test_sliding_window_trims(self):
        """Sliding window should trim old messages."""
        session = Session(store="memory", max_messages=3)

        for i in range(5):
            session.add_message("user", f"Message {i}")

        assert session.message_count == 3
        assert session.messages[0].content == "Message 2"
        assert session.messages[-1].content == "Message 4"

    def test_get_history(self):
        """get_history should return list of dicts."""
        session = Session(store="memory")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")

        history = session.get_history()
        assert len(history) == 2
        assert history[0] == {"role": "user", "content": "Hello"}
        assert history[1] == {"role": "assistant", "content": "Hi"}

    def test_format_history(self):
        """format_history should return formatted string."""
        session = Session(store="memory")
        session.add_message("user", "Hello")
        session.add_message("assistant", "Hi")

        formatted = session.format_history()
        assert "user: Hello" in formatted
        assert "assistant: Hi" in formatted

    def test_clear_session(self):
        """clear should remove all messages."""
        session = Session(store="memory")
        session.add_message("user", "Hello")
        session.clear()
        assert session.message_count == 0

    def test_save_and_load_with_memory_store(self):
        """Session should persist and reload from memory store."""
        store = InMemoryStateStore()

        # Save
        session1 = Session(store="memory")
        session1._store_instance = store
        session1.session_id = "test-123"
        session1.add_message("user", "Hello")
        session1.add_message("assistant", "Hi")
        session1.save()

        # Load
        session2 = Session(store="memory")
        session2._store_instance = store
        session2.load("test-123")

        assert session2.message_count == 2
        assert session2.messages[0].content == "Hello"


class TestMessage:
    """Test Message dataclass."""

    def test_message_creation(self):
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp > 0

    def test_message_serialization(self):
        msg = Message(role="user", content="Hello")
        data = msg.to_dict()
        assert data["role"] == "user"
        assert data["content"] == "Hello"

        msg2 = Message.from_dict(data)
        assert msg2.role == msg.role
        assert msg2.content == msg.content
