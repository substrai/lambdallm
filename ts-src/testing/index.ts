/**
 * Testing utilities for LambdaLLM TypeScript.
 */

import { ModelResponse } from '../core/models';
import { LambdaLLMContext } from '../core/context';

export class MockLambdaContext {
  functionName = 'test-function';
  functionVersion = '$LATEST';
  memoryLimitInMB = 256;
  private timeoutMs: number;

  constructor(timeoutMs = 30000) { this.timeoutMs = timeoutMs; }
  getRemainingTimeInMillis(): number { return this.timeoutMs; }
}

export class MockProvider {
  private responses: string[];
  private callCount = 0;
  public calls: Array<{ prompt: string }> = [];

  constructor(responses: string[] = ['Mock response']) {
    this.responses = responses;
  }

  async invoke(prompt: string): Promise<ModelResponse> {
    this.calls.push({ prompt });
    const content = this.responses[this.callCount % this.responses.length];
    this.callCount++;
    return {
      content,
      modelId: 'mock-model',
      tokensIn: Math.ceil(prompt.length / 4),
      tokensOut: Math.ceil(content.length / 4),
      latencyMs: 50,
      costUsd: 0.0001,
    };
  }

  get totalCalls(): number { return this.callCount; }
  get lastPrompt(): string | undefined { return this.calls[this.calls.length - 1]?.prompt; }
}

export interface GoldenCase {
  name: string;
  input: Record<string, any>;
  expectedContains?: string[];
  expectedNotContains?: string[];
  maxLatencyMs?: number;
}

export interface GoldenResult {
  datasetName: string;
  totalCases: number;
  passed: number;
  failed: number;
  passRate: number;
  avgLatencyMs: number;
  results: Array<{ caseName: string; passed: boolean; failureReason?: string }>;
}

export class GoldenDatasetRunner {
  async run(cases: GoldenCase[], handler: (event: any, ctx: any) => Promise<any>): Promise<GoldenResult> {
    const results: GoldenResult['results'] = [];
    let totalLatency = 0;

    for (const testCase of cases) {
      const start = Date.now();
      try {
        const event = { body: JSON.stringify(testCase.input) };
        const result = await handler(event, new MockLambdaContext());
        const latency = Date.now() - start;
        totalLatency += latency;

        const body = typeof result.body === 'string' ? result.body : JSON.stringify(result.body);
        const { passed, reason } = this.validate(testCase, body, latency);
        results.push({ caseName: testCase.name, passed, failureReason: reason });
      } catch (e: any) {
        totalLatency += Date.now() - start;
        results.push({ caseName: testCase.name, passed: false, failureReason: `Exception: ${e.message}` });
      }
    }

    const passed = results.filter(r => r.passed).length;
    return {
      datasetName: 'golden',
      totalCases: cases.length,
      passed,
      failed: cases.length - passed,
      passRate: passed / Math.max(cases.length, 1),
      avgLatencyMs: totalLatency / Math.max(cases.length, 1),
      results,
    };
  }

  private validate(testCase: GoldenCase, output: string, latencyMs: number): { passed: boolean; reason?: string } {
    const lower = output.toLowerCase();
    if (testCase.expectedContains) {
      for (const expected of testCase.expectedContains) {
        if (!lower.includes(expected.toLowerCase())) return { passed: false, reason: `Expected to contain '${expected}'` };
      }
    }
    if (testCase.expectedNotContains) {
      for (const notExpected of testCase.expectedNotContains) {
        if (lower.includes(notExpected.toLowerCase())) return { passed: false, reason: `Should not contain '${notExpected}'` };
      }
    }
    if (testCase.maxLatencyMs && latencyMs > testCase.maxLatencyMs) {
      return { passed: false, reason: `Latency ${latencyMs}ms exceeds max ${testCase.maxLatencyMs}ms` };
    }
    return { passed: true };
  }
}
