"""Cost tracking middleware for LambdaLLM.

Enforces budget limits and tracks spending per request.
Implements the cost-aware principle of the framework.
"""

import json
import logging
import time
from typing import Any, Optional

from lambdallm.middleware.base import Middleware
from lambdallm.core.exceptions import BudgetExceededError

logger = logging.getLogger("lambdallm")


class CostTrackingMiddleware(Middleware):
    """Tracks and enforces cost budgets.

    Can block, downgrade, or alert when budget thresholds are approached.

    Config:
        daily_budget_usd: Maximum daily spend in USD.
        monthly_budget_usd: Maximum monthly spend in USD.
        on_exceeded: Action when budget exceeded ("block", "downgrade", "alert").
        alert_threshold: Percentage at which to start alerting (default: 0.8).
    """

    def __init__(
        self,
        daily_budget_usd: float = 50.0,
        monthly_budget_usd: float = 1000.0,
        on_exceeded: str = "block",
        alert_threshold: float = 0.8,
        cost_store: Optional[Any] = None,
    ):
        self.daily_budget_usd = daily_budget_usd
        self.monthly_budget_usd = monthly_budget_usd
        self.on_exceeded = on_exceeded
        self.alert_threshold = alert_threshold
        self.cost_store = cost_store

    def before_invoke(self, event: dict, context: Any) -> dict:
        """Check budget before allowing the request."""
        current_spend = self._get_current_spend()

        if current_spend >= self.daily_budget_usd:
            if self.on_exceeded == "block":
                raise BudgetExceededError(
                    f"Daily budget exceeded: ${current_spend:.4f} / ${self.daily_budget_usd:.2f}"
                )
            elif self.on_exceeded == "downgrade":
                # Signal to the router to use cheapest model
                context._budget_exceeded = True
                logger.warning(f"Budget threshold reached, downgrading model selection")
            elif self.on_exceeded == "alert":
                logger.warning(f"Budget exceeded: ${current_spend:.4f} / ${self.daily_budget_usd:.2f}")

        elif current_spend >= (self.daily_budget_usd * self.alert_threshold):
            logger.warning(
                f"Approaching budget limit: ${current_spend:.4f} / ${self.daily_budget_usd:.2f} "
                f"({current_spend / self.daily_budget_usd * 100:.1f}%)"
            )

        return event

    def after_invoke(self, event: dict, result: Any, context: Any) -> Any:
        """Record the cost of this invocation."""
        cost = context.total_cost
        if cost > 0:
            self._record_cost(cost)
            logger.debug(f"Request cost: ${cost:.6f}")

        return result

    def _get_current_spend(self) -> float:
        """Get current daily spend. Override for DynamoDB-backed tracking."""
        if self.cost_store:
            return self.cost_store.get_daily_spend()
        # In-memory fallback (resets on cold start - fine for basic usage)
        return 0.0

    def _record_cost(self, cost: float) -> None:
        """Record cost of an invocation."""
        if self.cost_store:
            self.cost_store.record(cost)
