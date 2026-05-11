"""DynamoDB state store for LambdaLLM.

Provides serverless, pay-per-use session persistence.
Optimized for Lambda with lazy client initialization.
"""

import json
import time
import logging
from typing import Any, Optional

logger = logging.getLogger("lambdallm")

# Module-level client for Lambda container reuse
_dynamodb_client = None
_table_name = None


def _get_client(region: str = "us-east-1"):
    """Lazy-load DynamoDB client (cold-start optimization)."""
    global _dynamodb_client
    if _dynamodb_client is None:
        import boto3
        _dynamodb_client = boto3.resource("dynamodb", region_name=region)
    return _dynamodb_client


class DynamoDBStateStore:
    """DynamoDB-backed state store for session persistence.

    Table schema (auto-created by lambdallm deploy):
        - Partition key: session_id (S)
        - TTL attribute: expires_at (N)

    Attributes:
        table_name: DynamoDB table name (default: lambdallm-sessions)
        region: AWS region (default: us-east-1)
    """

    def __init__(self, table_name: str = "lambdallm-sessions", region: str = "us-east-1"):
        self.table_name = table_name
        self.region = region
        self._table = None

    @property
    def table(self):
        """Lazy-load table reference."""
        if self._table is None:
            client = _get_client(self.region)
            self._table = client.Table(self.table_name)
        return self._table

    def get(self, session_id: str) -> Optional[dict]:
        """Retrieve session data by ID.

        Returns None if session doesn't exist or has expired.
        """
        try:
            response = self.table.get_item(Key={"session_id": session_id})
            item = response.get("Item")

            if not item:
                return None

            # Check TTL (DynamoDB TTL deletion is eventually consistent)
            expires_at = item.get("expires_at", 0)
            if expires_at and time.time() > expires_at:
                return None

            # Deserialize data
            data_str = item.get("data", "{}")
            return json.loads(data_str) if isinstance(data_str, str) else data_str

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def put(self, session_id: str, data: dict, ttl_seconds: int = 86400) -> None:
        """Store session data.

        Args:
            session_id: Unique session identifier.
            data: Session data to persist.
            ttl_seconds: Time-to-live in seconds (default: 24 hours).
        """
        try:
            expires_at = int(time.time() + ttl_seconds)

            self.table.put_item(
                Item={
                    "session_id": session_id,
                    "data": json.dumps(data),
                    "expires_at": expires_at,
                    "updated_at": int(time.time()),
                }
            )
            logger.debug(f"Saved session {session_id} (TTL: {ttl_seconds}s)")

        except Exception as e:
            logger.error(f"Failed to save session {session_id}: {e}")
            raise

    def delete(self, session_id: str) -> None:
        """Delete a session."""
        try:
            self.table.delete_item(Key={"session_id": session_id})
            logger.debug(f"Deleted session {session_id}")
        except Exception as e:
            logger.error(f"Failed to delete session {session_id}: {e}")

    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        return self.get(session_id) is not None
