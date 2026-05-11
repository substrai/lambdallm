"""In-memory state store for LambdaLLM.

Used for local development and testing. Data does not persist
across Lambda invocations (only useful for single-invocation testing).
"""

import time
from typing import Optional


class InMemoryStateStore:
    """In-memory state store for development/testing.

    WARNING: Data is lost between Lambda invocations.
    Use DynamoDB for production.
    """

    _store: dict = {}

    def get(self, session_id: str) -> Optional[dict]:
        """Retrieve session data."""
        entry = self._store.get(session_id)
        if not entry:
            return None

        # Check TTL
        if entry.get("expires_at", 0) and time.time() > entry["expires_at"]:
            del self._store[session_id]
            return None

        return entry.get("data")

    def put(self, session_id: str, data: dict, ttl_seconds: int = 86400) -> None:
        """Store session data."""
        self._store[session_id] = {
            "data": data,
            "expires_at": time.time() + ttl_seconds,
            "updated_at": time.time(),
        }

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        self._store.pop(session_id, None)

    def exists(self, session_id: str) -> bool:
        """Check if session exists."""
        return self.get(session_id) is not None

    def clear(self) -> None:
        """Clear all sessions (useful in tests)."""
        self._store.clear()
