/**
 * Logging and Cost middleware for LambdaLLM.
 */

import { Middleware } from './base';
import { LambdaLLMContext } from '../core/context';

export class LoggingMiddleware extends Middleware {
  private startTime = 0;
  private logPrompts: boolean;

  constructor(options: { logPrompts?: boolean } = {}) {
    super();
    this.logPrompts = options.logPrompts || false;
  }

  beforeInvoke(event: Record<string, any>, context: LambdaLLMContext): Record<string, any> {
    this.startTime = Date.now();
    const logData: any = {
      event: 'request.start',
      path: event.path || event.rawPath || '/',
      method: event.httpMethod || 'INVOKE',
    };
    if (this.logPrompts && event.body) {
      logData.body_preview = typeof event.body === 'string' ? event.body.slice(0, 200) : event.body;
    }
    console.log(JSON.stringify(logData));
    return event;
  }

  afterInvoke(event: Record<string, any>, result: any, context: LambdaLLMContext): any {
    const latencyMs = Date.now() - this.startTime;
    console.log(JSON.stringify({
      event: 'request.complete',
      latency_ms: latencyMs,
      status_code: result?.statusCode || 200,
      total_cost_usd: context.totalCost,
      invocations: context.invocationCount,
    }));
    return result;
  }
}

export class CostTrackingMiddleware extends Middleware {
  private dailyBudget: number;
  private onExceeded: 'block' | 'downgrade' | 'alert';

  constructor(options: { dailyBudget?: number; onExceeded?: 'block' | 'downgrade' | 'alert' } = {}) {
    super();
    this.dailyBudget = options.dailyBudget || 50.0;
    this.onExceeded = options.onExceeded || 'block';
  }

  beforeInvoke(event: Record<string, any>, context: LambdaLLMContext): Record<string, any> {
    // Budget check would query DynamoDB in production
    return event;
  }

  afterInvoke(event: Record<string, any>, result: any, context: LambdaLLMContext): any {
    if (context.totalCost > 0) {
      console.log(JSON.stringify({
        event: 'cost.recorded',
        cost_usd: context.totalCost,
        budget_daily: this.dailyBudget,
      }));
    }
    return result;
  }
}
