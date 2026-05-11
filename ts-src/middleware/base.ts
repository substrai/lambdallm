/**
 * Middleware base class for LambdaLLM.
 */

import { LambdaLLMContext } from '../core/context';

export abstract class Middleware {
  beforeInvoke(event: Record<string, any>, context: LambdaLLMContext): Record<string, any> {
    return event;
  }

  afterInvoke(event: Record<string, any>, result: any, context: LambdaLLMContext): any {
    return result;
  }

  onError(event: Record<string, any>, error: Error, context: LambdaLLMContext): void {}
}
