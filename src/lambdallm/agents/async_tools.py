"""Async tool execution for LambdaLLM agents.

Dispatches long-running tool calls to SQS for background processing.
The agent can resume when the tool result arrives via callback.
"""

import json
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Optional

from lambdallm.core.exceptions import LambdaLLMError

logger = logging.getLogger("lambdallm")


@dataclass
class AsyncToolRequest:
    """A request to execute a tool asynchronously."""

    request_id: str
    tool_name: str
    tool_input: dict
    callback_url: Optional[str] = None
    session_id: Optional[str] = None
    timeout_seconds: int = 300


@dataclass
class AsyncToolResult:
    """Result from an async tool execution."""

    request_id: str
    tool_name: str
    output: Any = None
    error: Optional[str] = None
    status: str = "completed"  # completed | failed | timeout


class AsyncToolDispatcher:
    """Dispatches tool calls to SQS for async execution.

    Used when a tool would take too long for synchronous execution
    within Lambda's timeout. The agent checkpoints its state and
    resumes when the tool result arrives.

    Flow:
    1. Agent decides to call a long-running tool
    2. Dispatcher sends request to SQS queue
    3. Agent returns 202 (Accepted) with request_id
    4. Background worker processes the tool call
    5. Worker sends result to callback (or stores in DynamoDB)
    6. Next invocation resumes the agent with the tool result

    Example:
        dispatcher = AsyncToolDispatcher(queue_url="https://sqs.../my-queue")
        request = dispatcher.dispatch("long_search", {"query": "complex query"})
        # Returns immediately with request_id
    """

    def __init__(
        self,
        queue_url: Optional[str] = None,
        region: str = "us-east-1",
        result_store: Optional[Any] = None,
    ):
        self.queue_url = queue_url
        self.region = region
        self.result_store = result_store
        self._client = None

    @property
    def sqs_client(self):
        """Lazy-load SQS client."""
        if self._client is None:
            import boto3
            self._client = boto3.client("sqs", region_name=self.region)
        return self._client

    def dispatch(
        self,
        tool_name: str,
        tool_input: dict,
        session_id: Optional[str] = None,
        callback_url: Optional[str] = None,
        timeout_seconds: int = 300,
    ) -> AsyncToolRequest:
        """Dispatch a tool call for async execution.

        Args:
            tool_name: Name of the tool to execute.
            tool_input: Input arguments for the tool.
            session_id: Session ID for result correlation.
            callback_url: URL to POST result to when complete.
            timeout_seconds: Max time for async execution.

        Returns:
            AsyncToolRequest with request_id for tracking.
        """
        request_id = str(uuid.uuid4())

        request = AsyncToolRequest(
            request_id=request_id,
            tool_name=tool_name,
            tool_input=tool_input,
            callback_url=callback_url,
            session_id=session_id,
            timeout_seconds=timeout_seconds,
        )

        if self.queue_url:
            self._send_to_sqs(request)
        else:
            logger.warning("No SQS queue configured. Async dispatch is a no-op.")

        logger.info(f"Dispatched async tool '{tool_name}' (request_id: {request_id})")
        return request

    def get_result(self, request_id: str) -> Optional[AsyncToolResult]:
        """Check if an async tool result is available.

        Args:
            request_id: The request ID from dispatch().

        Returns:
            AsyncToolResult if available, None if still pending.
        """
        if self.result_store:
            data = self.result_store.get(f"async:{request_id}")
            if data:
                return AsyncToolResult(
                    request_id=request_id,
                    tool_name=data.get("tool_name", ""),
                    output=data.get("output"),
                    error=data.get("error"),
                    status=data.get("status", "completed"),
                )
        return None

    def store_result(self, result: AsyncToolResult) -> None:
        """Store an async tool result (called by the background worker)."""
        if self.result_store:
            self.result_store.put(
                f"async:{result.request_id}",
                {
                    "tool_name": result.tool_name,
                    "output": result.output,
                    "error": result.error,
                    "status": result.status,
                },
                ttl_seconds=3600,  # Results expire after 1 hour
            )

    def _send_to_sqs(self, request: AsyncToolRequest) -> None:
        """Send tool request to SQS queue."""
        try:
            self.sqs_client.send_message(
                QueueUrl=self.queue_url,
                MessageBody=json.dumps({
                    "request_id": request.request_id,
                    "tool_name": request.tool_name,
                    "tool_input": request.tool_input,
                    "callback_url": request.callback_url,
                    "session_id": request.session_id,
                    "timeout_seconds": request.timeout_seconds,
                }),
                MessageAttributes={
                    "tool_name": {"DataType": "String", "StringValue": request.tool_name},
                    "request_id": {"DataType": "String", "StringValue": request.request_id},
                },
            )
        except Exception as e:
            raise LambdaLLMError(f"Failed to dispatch async tool: {e}")


class HumanInTheLoop:
    """Pauses agent execution to request human approval.

    Sends a notification (SNS/SQS) and waits for human response.
    The agent checkpoints and resumes when approval arrives.

    Example:
        hitl = HumanInTheLoop(topic_arn="arn:aws:sns:...")

        # In agent tool:
        @Tool(description="Request human approval for high-value actions")
        def request_approval(action: str, reason: str) -> str:
            request = hitl.request(action=action, reason=reason)
            return f"Approval requested (id: {request.request_id}). Waiting for human."
    """

    def __init__(self, topic_arn: Optional[str] = None, region: str = "us-east-1"):
        self.topic_arn = topic_arn
        self.region = region
        self._sns_client = None

    def request(self, action: str, reason: str, metadata: Optional[dict] = None) -> AsyncToolRequest:
        """Send a human approval request.

        Args:
            action: What action needs approval.
            reason: Why approval is needed.
            metadata: Additional context for the reviewer.

        Returns:
            AsyncToolRequest for tracking the approval.
        """
        request_id = str(uuid.uuid4())

        message = {
            "type": "approval_request",
            "request_id": request_id,
            "action": action,
            "reason": reason,
            "metadata": metadata or {},
        }

        if self.topic_arn:
            self._send_notification(message)

        logger.info(f"Human approval requested for: {action} (id: {request_id})")

        return AsyncToolRequest(
            request_id=request_id,
            tool_name="human_approval",
            tool_input={"action": action, "reason": reason},
        )

    def _send_notification(self, message: dict) -> None:
        """Send notification via SNS."""
        if self._sns_client is None:
            import boto3
            self._sns_client = boto3.client("sns", region_name=self.region)

        try:
            self._sns_client.publish(
                TopicArn=self.topic_arn,
                Subject=f"Approval Required: {message.get('action', 'Unknown')}",
                Message=json.dumps(message, indent=2),
            )
        except Exception as e:
            logger.error(f"Failed to send approval notification: {e}")
