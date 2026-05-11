/**
 * LambdaLLM - Serverless-native LLM orchestration framework for AWS Lambda.
 * TypeScript implementation - full feature parity with Python version.
 *
 * @packageDocumentation
 * @author Gaurav Kumar Sinha <gaurav@substrai.dev>
 * @see https://github.com/substrai/lambdallm
 */

// Core
export { handler, HandlerOptions } from './core/handler';
export { Prompt, PromptOptions } from './core/prompt';
export { Model, ModelConfig, ModelResponse, calculateCost, MODEL_COSTS } from './core/models';
export { LambdaLLMContext } from './core/context';
export { LambdaLLMError, ModelInvocationError, TimeoutError, BudgetExceededError, ConfigurationError } from './core/exceptions';

// Chains
export { Chain, Step, ChainResult, StepOptions, ChainOptions } from './chains/chain';

// State
export { Session, MemoryStrategy, Message } from './state/session';
export { DynamoDBStateStore, InMemoryStateStore, StateStore } from './state/store';
export { ContextWindowManager } from './state/contextWindow';

// Middleware
export { Middleware } from './middleware/base';
export { LoggingMiddleware, CostTrackingMiddleware } from './middleware/logging';

// Agents
export { Tool, Agent, AgentResult, ToolDefinition, AgentOptions } from './agents/agent';
export { AgentRouter, RouteConfig, ToolSandbox, SandboxPolicy, AsyncToolDispatcher, StreamingResponse } from './agents/extras';

// Observability
export { Tracer, Span, MetricsEmitter, CostTracker, CostEntry, CostReport, Experiment, Variant, PromptAnalytics, PromptMetrics, CostAwareRouter, RoutingRule } from './observability/index';

// Testing
export { MockLambdaContext, MockProvider, GoldenDatasetRunner, GoldenCase, GoldenResult } from './testing/index';

export const VERSION = '1.2.0';
