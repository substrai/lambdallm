"""Auto-session integration for the @handler decorator.

Automatically loads and saves session state around handler execution,
making multi-turn conversations transparent to the user.
"""

import logging
from typing import Any, Optional

from lambdallm.state.session import Session
from lambdallm.state.context_window import ContextWindowManager, ContextWindowConfig

logger = logging.getLogger("lambdallm")


class AutoSession:
    """Manages automatic session lifecycle within a handler invocation.

    Integrates with @handler to:
    1. Extract session_id from the event
    2. Load session from DynamoDB before handler executes
    3. Provide session to the handler via context
    4. Save session after handler completes
    5. Apply context window management to history
    """

    def __init__(self, session_config: Session, model_id: Optional[str] = None):
        self.session_config = session_config
        self.context_manager = ContextWindowManager(
            ContextWindowConfig(model_id=model_id)
        )

    def load_from_event(self, event: dict) -> Session:
        """Extract session_id from event and load session.

        Looks for session_id in:
        1. event["body"]["session_id"]
        2. event["headers"]["x-session-id"]
        3. event["requestContext"]["connectionId"] (WebSocket)
        4. Generates a new session_id if none found
        """
        session_id = self._extract_session_id(event)

        session = Session(
            store=self.session_config.store,
            ttl_hours=self.session_config.ttl_hours,
            memory=self.session_config.memory,
            max_messages=self.session_config.max_messages,
        )

        session.load(session_id)
        logger.debug(f"Loaded session '{session_id}' with {session.message_count} messages")

        return session

    def save_session(self, session: Session) -> None:
        """Save session state back to store."""
        if session and session._dirty:
            session.save()
            logger.debug(f"Saved session '{session.session_id}'")

    def apply_context_window(self, session: Session, system_prompt: str = "") -> list[dict]:
        """Get conversation history trimmed to fit context window.

        Args:
            session: The loaded session.
            system_prompt: System prompt to account for in token budget.

        Returns:
            List of message dicts that fit within the context window.
        """
        if not session or not session.messages:
            return []

        trimmed = self.context_manager.fit_messages(
            session.messages,
            system_prompt=system_prompt,
        )

        return [{"role": m.role, "content": m.content} for m in trimmed]

    def _extract_session_id(self, event: dict) -> str:
        """Extract session_id from various event formats."""
        import json as json_module
        import uuid

        # Try body
        body = event.get("body", {})
        if isinstance(body, str):
            try:
                body = json_module.loads(body)
            except (json_module.JSONDecodeError, TypeError):
                body = {}

        if isinstance(body, dict) and "session_id" in body:
            return body["session_id"]

        # Try headers
        headers = event.get("headers", {})
        if headers and "x-session-id" in headers:
            return headers["x-session-id"]

        # Try WebSocket connectionId
        request_context = event.get("requestContext", {})
        if "connectionId" in request_context:
            return request_context["connectionId"]

        # Generate new session ID
        new_id = str(uuid.uuid4())
        logger.debug(f"No session_id found in event, generated: {new_id}")
        return new_id
