"""Agent implementation for LambdaLLM.

ReAct-style agent that uses tools, makes decisions, and operates
within Lambda's timeout constraints.
"""

import json
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from lambdallm.agents.tool import ToolRegistry, ToolDefinition
from lambdallm.core.exceptions import LambdaLLMError, TimeoutError

logger = logging.getLogger("lambdallm")


@dataclass
class AgentConfig:
    """Configuration for an agent."""

    max_iterations: int = 10
    timeout_buffer: int = 30  # seconds reserved before Lambda timeout
    max_tool_calls: int = 20
    max_cost_usd: Optional[float] = None
    allow_parallel_tools: bool = False
    verbose: bool = False


@dataclass
class AgentStep:
    """A single step in the agent's reasoning loop."""

    iteration: int
    thought: str = ""
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    is_final: bool = False
    final_answer: Optional[str] = None
    cost_usd: float = 0.0
    latency_ms: float = 0.0


@dataclass
class AgentResult:
    """Result from an agent execution."""

    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    status: str = "completed"  # completed | max_iterations | timeout | cost_limit
    metadata: dict = field(default_factory=dict)


class Agent:
    """ReAct-style AI agent with tool usage.

    Implements the Reason-Act-Observe loop:
    1. Reason: LLM decides what to do next
    2. Act: Execute a tool or provide final answer
    3. Observe: Feed tool result back to LLM

    Respects Lambda timeout by checking remaining time before each iteration.

    Example:
        @Tool(description="Search documents")
        def search(query: str) -> list[dict]:
            pass

        agent = Agent(
            name="researcher",
            system_prompt="You are a research assistant.",
            tools=[search],
            max_iterations=5,
        )

        result = agent.run(query="What is LambdaLLM?", context=context)
    """

    def __init__(
        self,
        name: str,
        system_prompt: str,
        tools: list,
        max_iterations: int = 10,
        timeout_buffer: int = 30,
        max_tool_calls: int = 20,
        max_cost_usd: Optional[float] = None,
        verbose: bool = False,
    ):
        self.name = name
        self.system_prompt = system_prompt
        self.config = AgentConfig(
            max_iterations=max_iterations,
            timeout_buffer=timeout_buffer,
            max_tool_calls=max_tool_calls,
            max_cost_usd=max_cost_usd,
            verbose=verbose,
        )
        self.registry = ToolRegistry(tools)

    def run(self, query: str, context: Any, session_history: Optional[list] = None) -> AgentResult:
        """Execute the agent's reasoning loop.

        Args:
            query: The user's question or task.
            context: LambdaLLMContext for model invocations.
            session_history: Optional previous conversation messages.

        Returns:
            AgentResult with the final answer and execution trace.
        """
        start_time = time.time()
        steps: list[AgentStep] = []
        tool_call_count = 0
        total_cost = 0.0

        # Build the initial messages
        messages = self._build_initial_messages(query, session_history)

        for iteration in range(self.config.max_iterations):
            # Check timeout
            if self._should_stop_for_timeout(context):
                logger.warning(f"Agent '{self.name}' stopping: approaching Lambda timeout")
                return self._build_result(
                    steps, "timeout", start_time, total_cost, tool_call_count,
                    fallback_answer="I need more time to complete this task. Please try again."
                )

            # Check cost limit
            if self.config.max_cost_usd and total_cost >= self.config.max_cost_usd:
                logger.warning(f"Agent '{self.name}' stopping: cost limit reached (${total_cost:.4f})")
                return self._build_result(
                    steps, "cost_limit", start_time, total_cost, tool_call_count,
                    fallback_answer="I've reached my cost limit for this request."
                )

            # Reason: Ask LLM what to do next
            step_start = time.time()
            llm_response = self._invoke_llm(messages, context)
            step_cost = context.total_cost - total_cost
            total_cost = context.total_cost

            # Parse the LLM's decision
            decision = self._parse_decision(llm_response)

            step = AgentStep(
                iteration=iteration + 1,
                thought=decision.get("thought", ""),
                cost_usd=step_cost,
                latency_ms=(time.time() - step_start) * 1000,
            )

            if self.config.verbose:
                logger.info(f"[Agent:{self.name}] Iteration {iteration + 1}: {decision.get('thought', '')[:100]}")

            # Check if agent wants to give final answer
            if decision.get("final_answer"):
                step.is_final = True
                step.final_answer = decision["final_answer"]
                steps.append(step)
                return self._build_result(
                    steps, "completed", start_time, total_cost, tool_call_count,
                    fallback_answer=decision["final_answer"]
                )

            # Act: Execute the tool
            if decision.get("tool_name"):
                tool_name = decision["tool_name"]
                tool_input = decision.get("tool_input", {})

                step.tool_name = tool_name
                step.tool_input = tool_input

                # Execute tool
                try:
                    tool_output = self.registry.invoke(tool_name, **tool_input)
                    step.tool_output = str(tool_output)
                    tool_call_count += 1
                except LambdaLLMError as e:
                    step.tool_output = f"Error: {e}"

                if self.config.verbose:
                    logger.info(f"[Agent:{self.name}] Tool '{tool_name}': {str(step.tool_output)[:100]}")

                # Observe: Add tool result to messages
                messages.append({"role": "assistant", "content": llm_response})
                messages.append({
                    "role": "user",
                    "content": f"Tool '{tool_name}' returned:\n{step.tool_output}\n\nContinue reasoning.",
                })

                # Check tool call limit
                if tool_call_count >= self.config.max_tool_calls:
                    steps.append(step)
                    return self._build_result(
                        steps, "max_iterations", start_time, total_cost, tool_call_count,
                        fallback_answer="I've used all available tool calls. Based on what I found so far: " + (step.tool_output or "")
                    )

            steps.append(step)

        # Max iterations reached
        return self._build_result(
            steps, "max_iterations", start_time, total_cost, tool_call_count,
            fallback_answer="I've reached my iteration limit. Based on my analysis so far, here's what I found."
        )

    def _build_initial_messages(self, query: str, history: Optional[list]) -> list[dict]:
        """Build the initial message list for the agent."""
        tool_descriptions = "\n".join(
            f"- {t['name']}: {t['description']}" for t in self.registry.get_schemas()
        )

        system_content = f"""{self.system_prompt}

You have access to the following tools:
{tool_descriptions}

To use a tool, respond with JSON:
{{"thought": "your reasoning", "tool_name": "tool_name", "tool_input": {{"param": "value"}}}}

To give a final answer, respond with JSON:
{{"thought": "your reasoning", "final_answer": "your answer to the user"}}

Always respond with valid JSON. Think step by step."""

        messages = [{"role": "system", "content": system_content}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": query})

        return messages

    def _invoke_llm(self, messages: list[dict], context: Any) -> str:
        """Invoke the LLM with the current message history."""
        # Format messages into a single prompt for the context.invoke() interface
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append("Assistant:")
        full_prompt = "\n\n".join(prompt_parts)

        return context.invoke(full_prompt)

    def _parse_decision(self, response: str) -> dict:
        """Parse the LLM's JSON decision."""
        try:
            # Try direct JSON parse
            if response.strip().startswith("{"):
                return json.loads(response.strip())

            # Try extracting JSON from markdown code block
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
                return json.loads(json_str.strip())

            if "```" in response:
                json_str = response.split("```")[1].split("```")[0]
                return json.loads(json_str.strip())

            # Fallback: treat entire response as final answer
            return {"thought": "Direct response", "final_answer": response}

        except (json.JSONDecodeError, IndexError):
            # If we can't parse JSON, treat as final answer
            return {"thought": "Could not parse structured response", "final_answer": response}

    def _should_stop_for_timeout(self, context: Any) -> bool:
        """Check if we should stop due to approaching Lambda timeout."""
        if context and hasattr(context, "remaining_time_ms"):
            return context.remaining_time_ms < (self.config.timeout_buffer * 1000)
        return False

    def _build_result(
        self,
        steps: list[AgentStep],
        status: str,
        start_time: float,
        total_cost: float,
        tool_calls: int,
        fallback_answer: str = "",
    ) -> AgentResult:
        """Build the final AgentResult."""
        # Find the final answer from steps
        answer = fallback_answer
        for step in reversed(steps):
            if step.is_final and step.final_answer:
                answer = step.final_answer
                break

        return AgentResult(
            answer=answer,
            steps=steps,
            total_iterations=len(steps),
            total_tool_calls=tool_calls,
            total_cost_usd=total_cost,
            total_latency_ms=(time.time() - start_time) * 1000,
            status=status,
        )
