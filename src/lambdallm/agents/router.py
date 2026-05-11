"""Multi-agent router for LambdaLLM.

Routes incoming requests to specialized agents based on
intent classification. Enables building multi-agent systems
where each agent handles a specific domain.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.agents.agent import Agent, AgentResult

logger = logging.getLogger("lambdallm")


@dataclass
class RouteConfig:
    """Configuration for a single route."""

    agent: Agent
    description: str
    keywords: list[str] = field(default_factory=list)
    priority: int = 0  # Higher = checked first


class AgentRouter:
    """Routes requests to specialized agents based on intent.

    Supports two routing strategies:
    - keyword: Fast keyword matching (no LLM call)
    - llm: Use LLM to classify intent (more accurate, costs tokens)

    Example:
        router = AgentRouter(
            routes=[
                RouteConfig(agent=finance_agent, description="Financial questions", keywords=["revenue", "profit"]),
                RouteConfig(agent=tech_agent, description="Technical questions", keywords=["code", "api", "bug"]),
            ],
            fallback=general_agent,
            strategy="keyword",
        )

        result = router.route(query="What was Q3 revenue?", context=context)
    """

    def __init__(
        self,
        routes: list[RouteConfig],
        fallback: Optional[Agent] = None,
        strategy: str = "keyword",  # keyword | llm
    ):
        self.routes = sorted(routes, key=lambda r: r.priority, reverse=True)
        self.fallback = fallback
        self.strategy = strategy

    def route(self, query: str, context: Any, **kwargs) -> AgentResult:
        """Route a query to the appropriate agent.

        Args:
            query: The user's question or task.
            context: LambdaLLMContext for model invocations.
            **kwargs: Additional arguments passed to the agent.

        Returns:
            AgentResult from the selected agent.
        """
        if self.strategy == "keyword":
            agent = self._route_by_keyword(query)
        elif self.strategy == "llm":
            agent = self._route_by_llm(query, context)
        else:
            agent = self.fallback

        if agent is None:
            if self.fallback:
                agent = self.fallback
            else:
                raise ValueError("No agent matched and no fallback configured")

        logger.info(f"Routing to agent: {agent.name}")
        return agent.run(query=query, context=context, **kwargs)

    def _route_by_keyword(self, query: str) -> Optional[Agent]:
        """Route based on keyword matching (fast, no LLM call)."""
        query_lower = query.lower()

        for route in self.routes:
            for keyword in route.keywords:
                if keyword.lower() in query_lower:
                    logger.debug(f"Keyword match: '{keyword}' -> agent '{route.agent.name}'")
                    return route.agent

        return None

    def _route_by_llm(self, query: str, context: Any) -> Optional[Agent]:
        """Route using LLM intent classification (accurate, costs tokens)."""
        route_descriptions = "\n".join(
            f"- {route.agent.name}: {route.description}"
            for route in self.routes
        )

        classification_prompt = f"""Classify this query into one of these categories:
{route_descriptions}

Query: {query}

Respond with ONLY the agent name, nothing else."""

        try:
            response = context.invoke(classification_prompt)
            agent_name = response.strip().lower()

            for route in self.routes:
                if route.agent.name.lower() == agent_name:
                    return route.agent

        except Exception as e:
            logger.warning(f"LLM routing failed: {e}, using fallback")

        return None

    def add_route(self, route: RouteConfig) -> None:
        """Add a new route dynamically."""
        self.routes.append(route)
        self.routes.sort(key=lambda r: r.priority, reverse=True)

    @property
    def agent_names(self) -> list[str]:
        """List all registered agent names."""
        return [r.agent.name for r in self.routes]
