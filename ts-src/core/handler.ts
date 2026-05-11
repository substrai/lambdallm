/**
 * The @handler decorator equivalent for TypeScript Lambda handlers.
 */

import { Model, ModelConfig } from './models';
import { LambdaLLMContext } from './context';
import { LambdaLLMError, BudgetExceededError, TimeoutError, ModelInvocationError } from './exceptions';

export interface HandlerOptions {
  model?: Model | string;
  timeoutStrategy?: 'fail-fast' | 'truncate' | 'checkpoint';
  timeoutBuffer?: number;
  maxRetries?: number;
  fallbackModel?: Model | string;
}

type LambdaEvent = Record<string, any>;
type LambdaContext = any;
type LambdaResponse = { statusCode: number; headers?: Record<string, string>; body: string };

type HandlerFunction = (event: LambdaEvent, context: LambdaLLMContext) => Promise<any>;

/**
 * Wraps a Lambda handler with LLM orchestration.
 *
 * @example
 * ```typescript
 * export const lambdaHandler = handler(
 *   { model: Model.CLAUDE_3_HAIKU },
 *   async (event, context) => {
 *     const result = await context.invoke('Summarize: {text}', { text: event.body.text });
 *     return { statusCode: 200, body: { result } };
 *   }
 * );
 * ```
 */
export function handler(
  options: HandlerOptions,
  fn: HandlerFunction
): (event: LambdaEvent, lambdaContext: LambdaContext) => Promise<LambdaResponse> {

  const resolvedModel: ModelConfig = {
    modelId: options.model || Model.CLAUDE_3_HAIKU,
    maxTokens: 1024,
    temperature: 0.7,
    region: 'us-east-1',
  };

  const fallback = options.fallbackModel
    ? { modelId: options.fallbackModel as string, maxTokens: 1024, temperature: 0.7 }
    : undefined;

  return async (event: LambdaEvent, lambdaContext: LambdaContext): Promise<LambdaResponse> => {
    const startTime = Date.now();

    const context = new LambdaLLMContext({
      model: resolvedModel,
      timeoutBuffer: options.timeoutBuffer || 5,
      maxRetries: options.maxRetries || 3,
      fallbackModel: fallback,
      lambdaContext,
    });

    try {
      const result = await fn(event, context);
      return formatResponse(result);
    } catch (e: any) {
      if (e instanceof BudgetExceededError) {
        return errorResponse(429, e.message);
      }
      if (e instanceof TimeoutError) {
        return errorResponse(408, e.message);
      }
      if (e instanceof ModelInvocationError) {
        return errorResponse(502, e.message);
      }
      if (e instanceof LambdaLLMError) {
        return errorResponse(500, e.message);
      }
      return errorResponse(500, 'Internal server error');
    }
  };
}

function formatResponse(result: any): LambdaResponse {
  if (result && typeof result === 'object' && 'statusCode' in result) {
    const body = typeof result.body === 'object' ? JSON.stringify(result.body) : result.body || '';
    return { statusCode: result.statusCode, headers: { 'Content-Type': 'application/json' }, body };
  }
  return { statusCode: 200, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(result) };
}

function errorResponse(statusCode: number, message: string): LambdaResponse {
  return { statusCode, headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ error: message }) };
}
