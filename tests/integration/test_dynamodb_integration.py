"""Integration tests for DynamoDB state store using moto mock.

Tests full lifecycle: save/load/delete sessions, TTL behavior,
concurrent access patterns, and error handling edge cases.
"""

import json
import time
import threading
import pytest
from unittest.mock import patch
from moto import mock_aws
import boto3

from lambdallm.state.dynamodb import DynamoDBStateStore


@pytest.fixture(autouse=True)
def reset_global_client():
    """Reset module-level client between tests."""
    import lambdallm.state.dynamodb as mod
    mod._dynamodb_client = None
    mod._table_name = None
    yield
    mod._dynamodb_client = None
    mod._table_name = None


def _create_table():
    """Helper to create the mocked DynamoDB table."""
    client = boto3.resource("dynamodb", region_name="us-east-1")
    client.create_table(
        TableName="lambdallm-sessions",
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[
            {"AttributeName": "session_id", "AttributeType": "S"}
        ],
        BillingMode="PAY_PER_REQUEST",
    )


class TestSessionLifecycle:
    """Test basic CRUD operations for session state."""

    @mock_aws
    def test_put_and_get_session(self):
        """Test saving and retrieving a session."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        session_data = {"messages": [{"role": "user", "content": "hello"}], "model": "gpt-4"}
        store.put("session-001", session_data)

        result = store.get("session-001")
        assert result == session_data
        assert result["messages"][0]["content"] == "hello"

    @mock_aws
    def test_get_nonexistent_session_returns_none(self):
        """Test that getting a non-existent session returns None."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        result = store.get("nonexistent-session")
        assert result is None

    @mock_aws
    def test_delete_session(self):
        """Test deleting a session removes it from the store."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("session-to-delete", {"data": "temporary"})
        assert store.exists("session-to-delete") is True

        store.delete("session-to-delete")
        assert store.exists("session-to-delete") is False

    @mock_aws
    def test_delete_nonexistent_session_no_error(self):
        """Test that deleting a non-existent session does not raise."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        # Should not raise
        store.delete("never-existed")

    @mock_aws
    def test_exists_returns_correct_boolean(self):
        """Test exists() returns True for existing sessions, False otherwise."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        assert store.exists("session-x") is False
        store.put("session-x", {"active": True})
        assert store.exists("session-x") is True

    @mock_aws
    def test_overwrite_existing_session(self):
        """Test that putting data to an existing session overwrites it."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("session-overwrite", {"version": 1})
        store.put("session-overwrite", {"version": 2, "extra": "field"})

        result = store.get("session-overwrite")
        assert result == {"version": 2, "extra": "field"}


class TestTTLBehavior:
    """Test time-to-live expiration logic."""

    @mock_aws
    def test_expired_session_returns_none(self):
        """Test that an expired session is treated as non-existent."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("expiring-session", {"data": "temp"}, ttl_seconds=1)

        # Simulate time passing beyond TTL
        with patch("lambdallm.state.dynamodb.time.time", return_value=time.time() + 10):
            result = store.get("expiring-session")
            assert result is None

    @mock_aws
    def test_session_within_ttl_is_accessible(self):
        """Test that a session within its TTL window is still accessible."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("valid-session", {"data": "still here"}, ttl_seconds=3600)

        result = store.get("valid-session")
        assert result == {"data": "still here"}

    @mock_aws
    def test_custom_ttl_seconds(self):
        """Test that custom TTL values are stored correctly."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        now = time.time()
        with patch("lambdallm.state.dynamodb.time.time", return_value=now):
            store.put("custom-ttl", {"data": "test"}, ttl_seconds=7200)

        # Verify the item has correct expires_at
        table = boto3.resource("dynamodb", region_name="us-east-1").Table("lambdallm-sessions")
        response = table.get_item(Key={"session_id": "custom-ttl"})
        item = response["Item"]
        assert int(item["expires_at"]) == int(now + 7200)


class TestConcurrentAccess:
    """Test concurrent access patterns simulating Lambda parallelism."""

    @mock_aws
    def test_concurrent_writes_to_different_sessions(self):
        """Test that concurrent writes to different sessions succeed."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        results = {}
        errors = []

        def write_session(session_id, data):
            try:
                store.put(session_id, data)
                results[session_id] = store.get(session_id)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            t = threading.Thread(
                target=write_session,
                args=(f"concurrent-{i}", {"thread": i})
            )
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 5
        for i in range(5):
            assert results[f"concurrent-{i}"] == {"thread": i}

    @mock_aws
    def test_last_write_wins_same_session(self):
        """Test that concurrent writes to the same session use last-write-wins."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("contested-session", {"writer": "A", "seq": 1})
        store.put("contested-session", {"writer": "B", "seq": 2})

        result = store.get("contested-session")
        assert result["writer"] == "B"
        assert result["seq"] == 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    @mock_aws
    def test_large_session_data(self):
        """Test storing large session data."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        large_data = {
            "messages": [
                {"role": "assistant", "content": "x" * 1000}
                for _ in range(300)
            ]
        }
        store.put("large-session", large_data)
        result = store.get("large-session")
        assert len(result["messages"]) == 300

    @mock_aws
    def test_special_characters_in_session_id(self):
        """Test session IDs with special characters."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        special_ids = [
            "session/with/slashes",
            "session:with:colons",
            "session.with.dots",
            "session_with_underscores",
        ]
        for sid in special_ids:
            store.put(sid, {"id": sid})
            result = store.get(sid)
            assert result == {"id": sid}

    @mock_aws
    def test_empty_data_dict(self):
        """Test storing an empty dictionary."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        store.put("empty-session", {})
        result = store.get("empty-session")
        assert result == {}

    @mock_aws
    def test_nested_complex_data(self):
        """Test storing deeply nested and complex data structures."""
        _create_table()
        store = DynamoDBStateStore(table_name="lambdallm-sessions", region="us-east-1")

        complex_data = {
            "messages": [{"role": "user", "content": "test"}],
            "metadata": {
                "nested": {
                    "deep": {"value": [1, 2, 3]},
                    "list_of_dicts": [{"a": 1}, {"b": 2}],
                }
            },
            "flags": [True, False, None],
            "count": 42,
            "ratio": 3.14,
        }
        store.put("complex-session", complex_data)
        result = store.get("complex-session")
        assert result == complex_data

    @mock_aws
    def test_get_handles_table_error_gracefully(self):
        """Test that get returns None when table does not exist."""
        store = DynamoDBStateStore(table_name="nonexistent-table", region="us-east-1")
        result = store.get("any-session")
        assert result is None
