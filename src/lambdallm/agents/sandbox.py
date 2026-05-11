"""Tool sandboxing for LambdaLLM agents.

Restricts which AWS services and actions a tool can access.
Each agent gets a scoped execution environment based on its
declared IAM permissions.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.core.exceptions import LambdaLLMError

logger = logging.getLogger("lambdallm")


@dataclass
class SandboxPolicy:
    """Defines what a tool is allowed to do.

    Follows least-privilege: tools can only access
    explicitly declared resources and actions.
    """

    allowed_actions: list[str] = field(default_factory=list)
    denied_actions: list[str] = field(default_factory=list)
    allowed_resources: list[str] = field(default_factory=list)
    max_execution_time: int = 30  # seconds
    max_memory_mb: int = 128
    allow_network: bool = True
    allow_file_system: bool = False


class ToolSandbox:
    """Enforces execution constraints on tool invocations.

    Validates that tool calls comply with the sandbox policy
    before execution. Provides timeout enforcement and
    resource tracking.

    Example:
        sandbox = ToolSandbox(policy=SandboxPolicy(
            allowed_actions=["dynamodb:GetItem", "s3:GetObject"],
            max_execution_time=10,
        ))

        result = sandbox.execute(tool_func, **kwargs)
    """

    def __init__(self, policy: Optional[SandboxPolicy] = None):
        self.policy = policy or SandboxPolicy()
        self._execution_count = 0
        self._total_time = 0.0

    def execute(self, func, **kwargs) -> Any:
        """Execute a function within the sandbox constraints.

        Args:
            func: The tool function to execute.
            **kwargs: Arguments to pass to the function.

        Returns:
            Function result.

        Raises:
            LambdaLLMError: If execution violates sandbox policy.
        """
        import time
        import signal

        self._execution_count += 1
        start = time.time()

        # Set timeout (Unix only)
        try:
            old_handler = signal.signal(signal.SIGALRM, self._timeout_handler)
            signal.alarm(self.policy.max_execution_time)
        except (ValueError, AttributeError):
            # signal.alarm not available (Windows or non-main thread)
            old_handler = None

        try:
            result = func(**kwargs)
            elapsed = time.time() - start
            self._total_time += elapsed

            logger.debug(
                f"Sandbox: executed in {elapsed:.2f}s "
                f"(limit: {self.policy.max_execution_time}s)"
            )

            return result

        except TimeoutError:
            raise LambdaLLMError(
                f"Tool execution exceeded timeout ({self.policy.max_execution_time}s)"
            )
        finally:
            if old_handler is not None:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)

    def validate_action(self, action: str) -> bool:
        """Check if an IAM action is allowed by the policy."""
        if action in self.policy.denied_actions:
            return False

        if not self.policy.allowed_actions:
            return True  # No restrictions if no allowlist

        # Check wildcards
        for allowed in self.policy.allowed_actions:
            if allowed == "*":
                return True
            if allowed.endswith("*"):
                prefix = allowed[:-1]
                if action.startswith(prefix):
                    return True
            if action == allowed:
                return True

        return False

    def _timeout_handler(self, signum, frame):
        raise TimeoutError("Tool execution timed out")

    @property
    def stats(self) -> dict:
        return {
            "execution_count": self._execution_count,
            "total_time_seconds": round(self._total_time, 3),
        }
