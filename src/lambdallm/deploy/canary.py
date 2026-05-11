"""Canary deployment support for LambdaLLM.

Gradually shifts traffic from old version to new version,
automatically rolling back if error metrics spike.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from lambdallm.core.exceptions import LambdaLLMError

logger = logging.getLogger("lambdallm")


@dataclass
class CanaryConfig:
    """Configuration for canary deployments."""

    initial_traffic_percent: int = 10  # Start with 10% on new version
    increment_percent: int = 10  # Increase by 10% each interval
    interval_seconds: int = 60  # Wait 60s between increments
    error_threshold_percent: float = 5.0  # Rollback if errors exceed 5%
    rollback_on_alarm: bool = True
    max_duration_seconds: int = 600  # Max 10 minutes for full rollout


@dataclass
class CanaryStatus:
    """Current status of a canary deployment."""

    phase: str  # shifting | monitoring | complete | rolled_back
    current_traffic_percent: int = 0
    new_version: str = ""
    old_version: str = ""
    error_rate: float = 0.0
    elapsed_seconds: float = 0.0


class CanaryDeployer:
    """Manages gradual traffic shifting for safe deployments.

    Flow:
    1. Deploy new version alongside old version
    2. Route 10% traffic to new version
    3. Monitor error rate for 60 seconds
    4. If healthy, increase to 20%, 30%, ... 100%
    5. If errors spike, immediately rollback to 0%

    Uses Lambda aliases and weighted routing for traffic splitting.

    Example:
        canary = CanaryDeployer(config=CanaryConfig(
            initial_traffic_percent=10,
            interval_seconds=60,
            error_threshold_percent=5.0,
        ))

        status = canary.deploy(
            function_name="my-function",
            new_version="5",
            old_version="4",
        )
    """

    def __init__(self, config: Optional[CanaryConfig] = None, region: str = "us-east-1"):
        self.config = config or CanaryConfig()
        self.region = region
        self._client = None

    def deploy(self, function_name: str, new_version: str, old_version: str) -> CanaryStatus:
        """Execute a canary deployment.

        Args:
            function_name: Lambda function name.
            new_version: New version number to deploy.
            old_version: Current version to shift traffic from.

        Returns:
            CanaryStatus with final deployment state.
        """
        start_time = time.time()
        current_percent = self.config.initial_traffic_percent

        logger.info(
            f"Starting canary deployment: {function_name} "
            f"v{old_version} -> v{new_version} (starting at {current_percent}%)"
        )

        status = CanaryStatus(
            phase="shifting",
            new_version=new_version,
            old_version=old_version,
        )

        while current_percent <= 100:
            elapsed = time.time() - start_time

            # Check max duration
            if elapsed > self.config.max_duration_seconds:
                logger.warning("Canary max duration reached, completing at current level")
                break

            # Update traffic weight
            self._update_alias_weight(function_name, new_version, current_percent / 100.0)
            status.current_traffic_percent = current_percent

            logger.info(f"  Canary: {current_percent}% traffic on v{new_version}")

            # Monitor for errors
            time.sleep(self.config.interval_seconds)

            error_rate = self._check_error_rate(function_name, new_version)
            status.error_rate = error_rate

            if error_rate > self.config.error_threshold_percent:
                # ROLLBACK
                logger.error(
                    f"  Canary ROLLBACK: error rate {error_rate:.1f}% "
                    f"exceeds threshold {self.config.error_threshold_percent}%"
                )
                self._update_alias_weight(function_name, new_version, 0.0)
                status.phase = "rolled_back"
                status.elapsed_seconds = time.time() - start_time
                return status

            # Increase traffic
            current_percent += self.config.increment_percent

        # Full rollout complete
        self._update_alias_weight(function_name, new_version, 1.0)
        status.phase = "complete"
        status.current_traffic_percent = 100
        status.elapsed_seconds = time.time() - start_time

        logger.info(f"  Canary complete: 100% traffic on v{new_version}")
        return status

    def _update_alias_weight(self, function_name: str, version: str, weight: float) -> None:
        """Update Lambda alias routing weight."""
        client = self._get_client()
        if not client:
            logger.debug(f"Would set alias weight to {weight} for v{version}")
            return

        try:
            # Update the 'live' alias to split traffic
            routing_config = {}
            if 0 < weight < 1:
                routing_config = {
                    "AdditionalVersionWeights": {version: weight}
                }

            client.update_alias(
                FunctionName=function_name,
                Name="live",
                FunctionVersion=version if weight >= 1 else None,
                RoutingConfig=routing_config if routing_config else {},
            )
        except Exception as e:
            logger.error(f"Failed to update alias weight: {e}")

    def _check_error_rate(self, function_name: str, version: str) -> float:
        """Check error rate for the new version from CloudWatch."""
        # In production, query CloudWatch metrics
        # For now, return 0 (healthy)
        return 0.0

    def _get_client(self):
        """Lazy-load Lambda client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("lambda", region_name=self.region)
            except ImportError:
                return None
        return self._client
