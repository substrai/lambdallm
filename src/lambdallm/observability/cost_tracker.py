"""Persistent cost tracking for LambdaLLM.

Tracks cumulative spending across invocations using DynamoDB.
Provides budget enforcement, forecasting, and chargeback reporting.
"""

import time
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from lambdallm.core.exceptions import BudgetExceededError

logger = logging.getLogger("lambdallm")


@dataclass
class CostEntry:
    """A single cost record."""

    timestamp: float
    model_id: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    handler_name: str = ""
    request_id: str = ""
    user_id: str = ""
    metadata: dict = field(default_factory=dict)


@dataclass
class CostReport:
    """Aggregated cost report."""

    period: str  # "daily" | "monthly"
    start_date: str
    end_date: str
    total_cost_usd: float
    total_requests: int
    total_tokens_in: int
    total_tokens_out: int
    by_model: dict = field(default_factory=dict)
    by_handler: dict = field(default_factory=dict)
    by_user: dict = field(default_factory=dict)
    budget_usd: float = 0.0
    utilization_percent: float = 0.0


class CostTracker:
    """Persistent cost tracking with budget enforcement.

    Stores cost data in DynamoDB for cross-invocation tracking.
    Provides real-time budget checks and historical reporting.

    Features:
    - Per-request cost recording
    - Daily/monthly budget enforcement
    - Cost breakdown by model, handler, user
    - Spend forecasting based on current trajectory
    - Chargeback reports per team/project

    Example:
        tracker = CostTracker(
            daily_budget=50.0,
            monthly_budget=1000.0,
            on_exceeded="downgrade",
        )

        # Check budget before invocation
        tracker.check_budget()  # Raises BudgetExceededError if over

        # Record cost after invocation
        tracker.record(CostEntry(
            timestamp=time.time(),
            model_id="claude-3-haiku",
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.0003,
        ))
    """

    def __init__(
        self,
        daily_budget: float = 50.0,
        monthly_budget: float = 1000.0,
        on_exceeded: str = "block",  # block | downgrade | alert
        table_name: str = "lambdallm-costs",
        region: str = "us-east-1",
    ):
        self.daily_budget = daily_budget
        self.monthly_budget = monthly_budget
        self.on_exceeded = on_exceeded
        self.table_name = table_name
        self.region = region
        self._client = None
        self._daily_cache: Optional[float] = None
        self._cache_time: float = 0

    def check_budget(self) -> dict:
        """Check current budget status.

        Returns:
            Dict with daily/monthly spend and utilization.

        Raises:
            BudgetExceededError: If on_exceeded="block" and budget is exceeded.
        """
        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        status = {
            "daily_spend": daily_spend,
            "daily_budget": self.daily_budget,
            "daily_utilization": daily_spend / self.daily_budget if self.daily_budget > 0 else 0,
            "monthly_spend": monthly_spend,
            "monthly_budget": self.monthly_budget,
            "monthly_utilization": monthly_spend / self.monthly_budget if self.monthly_budget > 0 else 0,
            "exceeded": daily_spend >= self.daily_budget or monthly_spend >= self.monthly_budget,
        }

        if status["exceeded"] and self.on_exceeded == "block":
            raise BudgetExceededError(
                f"Budget exceeded: daily ${daily_spend:.4f}/${self.daily_budget:.2f}, "
                f"monthly ${monthly_spend:.4f}/${self.monthly_budget:.2f}"
            )

        if status["daily_utilization"] >= 0.8:
            logger.warning(
                f"Budget alert: {status['daily_utilization']*100:.1f}% of daily budget consumed"
            )

        return status

    def record(self, entry: CostEntry) -> None:
        """Record a cost entry."""
        try:
            self._write_entry(entry)
            # Invalidate cache
            self._daily_cache = None
        except Exception as e:
            logger.error(f"Failed to record cost: {e}")

    def get_daily_spend(self) -> float:
        """Get total spend for today (with caching)."""
        # Cache for 60 seconds to reduce DynamoDB reads
        if self._daily_cache is not None and (time.time() - self._cache_time) < 60:
            return self._daily_cache

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        spend = self._query_spend(today)
        self._daily_cache = spend
        self._cache_time = time.time()
        return spend

    def get_monthly_spend(self) -> float:
        """Get total spend for current month."""
        month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
        return self._query_spend(month_prefix)

    def get_report(self, period: str = "daily") -> CostReport:
        """Generate a cost report.

        Args:
            period: "daily" or "monthly"
        """
        now = datetime.now(timezone.utc)

        if period == "daily":
            date_key = now.strftime("%Y-%m-%d")
            budget = self.daily_budget
        else:
            date_key = now.strftime("%Y-%m")
            budget = self.monthly_budget

        entries = self._query_entries(date_key)

        # Aggregate
        total_cost = sum(e.get("cost_usd", 0) for e in entries)
        total_requests = len(entries)
        total_tokens_in = sum(e.get("tokens_in", 0) for e in entries)
        total_tokens_out = sum(e.get("tokens_out", 0) for e in entries)

        # Group by model
        by_model = {}
        for e in entries:
            model = e.get("model_id", "unknown")
            by_model.setdefault(model, 0)
            by_model[model] += e.get("cost_usd", 0)

        # Group by handler
        by_handler = {}
        for e in entries:
            handler_name = e.get("handler_name", "unknown")
            by_handler.setdefault(handler_name, 0)
            by_handler[handler_name] += e.get("cost_usd", 0)

        return CostReport(
            period=period,
            start_date=date_key,
            end_date=date_key,
            total_cost_usd=total_cost,
            total_requests=total_requests,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            by_model=by_model,
            by_handler=by_handler,
            budget_usd=budget,
            utilization_percent=(total_cost / budget * 100) if budget > 0 else 0,
        )

    def forecast_monthly(self) -> dict:
        """Forecast monthly spend based on current daily rate."""
        now = datetime.now(timezone.utc)
        day_of_month = now.day
        daily_spend = self.get_daily_spend()
        monthly_spend = self.get_monthly_spend()

        # Simple linear projection
        if day_of_month > 0:
            daily_avg = monthly_spend / day_of_month
            days_in_month = 30  # Approximation
            projected = daily_avg * days_in_month
        else:
            projected = 0

        return {
            "current_monthly_spend": monthly_spend,
            "daily_average": monthly_spend / max(day_of_month, 1),
            "projected_monthly": projected,
            "monthly_budget": self.monthly_budget,
            "projected_utilization": (projected / self.monthly_budget * 100) if self.monthly_budget > 0 else 0,
            "on_track": projected <= self.monthly_budget,
        }

    def _write_entry(self, entry: CostEntry) -> None:
        """Write cost entry to DynamoDB."""
        client = self._get_client()
        if not client:
            return

        date_key = datetime.fromtimestamp(entry.timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

        try:
            client.put_item(
                TableName=self.table_name,
                Item={
                    "date_key": {"S": date_key},
                    "timestamp": {"N": str(entry.timestamp)},
                    "model_id": {"S": entry.model_id},
                    "tokens_in": {"N": str(entry.tokens_in)},
                    "tokens_out": {"N": str(entry.tokens_out)},
                    "cost_usd": {"N": str(entry.cost_usd)},
                    "handler_name": {"S": entry.handler_name},
                    "request_id": {"S": entry.request_id},
                    "user_id": {"S": entry.user_id},
                },
            )
        except Exception as e:
            logger.error(f"DynamoDB write failed: {e}")

    def _query_spend(self, date_prefix: str) -> float:
        """Query total spend for a date prefix."""
        client = self._get_client()
        if not client:
            return 0.0

        try:
            response = client.query(
                TableName=self.table_name,
                KeyConditionExpression="date_key = :dk",
                ExpressionAttributeValues={":dk": {"S": date_prefix}},
                ProjectionExpression="cost_usd",
            )
            items = response.get("Items", [])
            return sum(float(item.get("cost_usd", {}).get("N", "0")) for item in items)
        except Exception as e:
            logger.error(f"Cost query failed: {e}")
            return 0.0

    def _query_entries(self, date_prefix: str) -> list[dict]:
        """Query all entries for a date prefix."""
        client = self._get_client()
        if not client:
            return []

        try:
            response = client.query(
                TableName=self.table_name,
                KeyConditionExpression="date_key = :dk",
                ExpressionAttributeValues={":dk": {"S": date_prefix}},
            )
            return [
                {k: list(v.values())[0] for k, v in item.items()}
                for item in response.get("Items", [])
            ]
        except Exception:
            return []

    def _get_client(self):
        """Lazy-load DynamoDB client."""
        if self._client is None:
            try:
                import boto3
                self._client = boto3.client("dynamodb", region_name=self.region)
            except ImportError:
                logger.debug("boto3 not available, cost tracking disabled")
                return None
        return self._client
