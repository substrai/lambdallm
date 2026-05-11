import { Prompt } from '../src/core/prompt';
import { Model, ModelConfig, calculateCost, MODEL_COSTS } from '../src/core/models';
import { LambdaLLMError, ModelInvocationError, TimeoutError, BudgetExceededError, ConfigurationError } from '../src/core/exceptions';
import { Session, MemoryStrategy } from '../src/state/session';
import { Chain, Step } from '../src/chains/chain';
import { ContextWindowManager } from '../src/state/contextWindow';
import { InMemoryStateStore } from '../src/state/store';
import { Tracer, MetricsEmitter, CostTracker, Experiment, PromptAnalytics, CostAwareRouter } from '../src/observability/index';
import { ToolSandbox, AgentRouter } from '../src/agents/extras';
import { Tool, Agent } from '../src/agents/agent';
import { MockLambdaContext, MockProvider, GoldenDatasetRunner } from '../src/testing/index';
import { LoggingMiddleware, CostTrackingMiddleware } from '../src/middleware/logging';

// ===== PROMPT TESTS =====

describe('Prompt', () => {
  test('creates with valid template', () => {
    const p = new Prompt({ template: 'Hello {name}' });
    expect(p.template).toBe('Hello {name}');
  });

  test('rejects empty template', () => {
    expect(() => new Prompt({ template: '' })).toThrow(ConfigurationError);
  });

  test('formats variables', () => {
    const p = new Prompt({ template: 'Hello {name}, age {age}' });
    expect(p.format({ name: 'Gaurav', age: '30' })).toBe('Hello Gaurav, age 30');
  });

  test('validates missing variables', () => {
    const p = new Prompt({ template: '{name} {age}' });
    expect(() => p.format({ name: 'test' })).toThrow(ConfigurationError);
  });
});

// ===== MODEL TESTS =====

describe('Model', () => {
  test('enum has correct values', () => {
    expect(Model.CLAUDE_3_HAIKU).toContain('anthropic');
    expect(Model.TITAN_TEXT_EXPRESS).toContain('amazon');
    expect(Model.LLAMA3_8B).toContain('meta');
  });

  test('calculateCost returns number', () => {
    const cost = calculateCost(Model.CLAUDE_3_HAIKU, 100, 50);
    expect(cost).toBeGreaterThan(0);
    expect(typeof cost).toBe('number');
  });

  test('MODEL_COSTS has entries for all models', () => {
    expect(MODEL_COSTS[Model.CLAUDE_3_HAIKU]).toBeDefined();
    expect(MODEL_COSTS[Model.CLAUDE_3_SONNET]).toBeDefined();
    expect(MODEL_COSTS[Model.TITAN_TEXT_EXPRESS]).toBeDefined();
  });
});

// ===== EXCEPTIONS TESTS =====

describe('Exceptions', () => {
  test('LambdaLLMError is an Error', () => {
    const e = new LambdaLLMError('test');
    expect(e).toBeInstanceOf(Error);
    expect(e.name).toBe('LambdaLLMError');
  });

  test('ModelInvocationError has retryable flag', () => {
    const e = new ModelInvocationError('throttled', true);
    expect(e.retryable).toBe(true);
    const e2 = new ModelInvocationError('invalid', false);
    expect(e2.retryable).toBe(false);
  });
});

// ===== SESSION TESTS =====

describe('Session', () => {
  test('creates with defaults', () => {
    const s = new Session();
    expect(s.messageCount).toBe(0);
    expect(s.store).toBe('memory');
  });

  test('adds messages', () => {
    const s = new Session();
    s.addMessage('user', 'Hello');
    s.addMessage('assistant', 'Hi');
    expect(s.messageCount).toBe(2);
    expect(s.lastMessage?.content).toBe('Hi');
  });

  test('sliding window trims', () => {
    const s = new Session({ maxMessages: 3 });
    for (let i = 0; i < 5; i++) s.addMessage('user', `Msg ${i}`);
    expect(s.messageCount).toBe(3);
    expect(s.messages[0].content).toBe('Msg 2');
  });

  test('getHistory returns dicts', () => {
    const s = new Session();
    s.addMessage('user', 'Hello');
    const history = s.getHistory();
    expect(history).toEqual([{ role: 'user', content: 'Hello' }]);
  });

  test('formatHistory returns string', () => {
    const s = new Session();
    s.addMessage('user', 'Hello');
    s.addMessage('assistant', 'Hi');
    expect(s.formatHistory()).toContain('user: Hello');
    expect(s.formatHistory()).toContain('assistant: Hi');
  });

  test('clear removes all', () => {
    const s = new Session();
    s.addMessage('user', 'Hello');
    s.clear();
    expect(s.messageCount).toBe(0);
  });
});

// ===== CHAIN TESTS =====

describe('Chain', () => {
  test('creates with steps', () => {
    const chain = new Chain({
      name: 'test',
      steps: [new Step({ name: 's1', prompt: 'Do: {input}' })],
    });
    expect(chain.name).toBe('test');
    expect(chain.steps.length).toBe(1);
  });

  test('rejects empty steps', () => {
    expect(() => new Chain({ name: 'empty', steps: [] })).toThrow();
  });

  test('Step requires prompt or func', () => {
    expect(() => new Step({ name: 'bad' })).toThrow();
  });

  test('Step cannot have both', () => {
    expect(() => new Step({ name: 'both', prompt: 'x', func: () => 'y' })).toThrow();
  });

  test('Step identifies type', () => {
    const llm = new Step({ name: 'a', prompt: 'test' });
    const transform = new Step({ name: 'b', func: () => 'x' });
    expect(llm.isLlmStep).toBe(true);
    expect(llm.isTransformStep).toBe(false);
    expect(transform.isTransformStep).toBe(true);
    expect(transform.isLlmStep).toBe(false);
  });
});

// ===== CONTEXT WINDOW TESTS =====

describe('ContextWindowManager', () => {
  test('trims old messages', () => {
    const mgr = new ContextWindowManager({ maxTokens: 100, reserveTokens: 20 });
    const messages = Array.from({ length: 50 }, (_, i) => ({
      role: 'user' as const, content: `Message ${i} with some content`, timestamp: Date.now(),
    }));
    const trimmed = mgr.fitMessages(messages);
    expect(trimmed.length).toBeLessThan(messages.length);
    expect(trimmed.length).toBeGreaterThan(0);
  });

  test('getUsageInfo returns stats', () => {
    const mgr = new ContextWindowManager({ maxTokens: 1000 });
    const messages = [{ role: 'user' as const, content: 'Hello world', timestamp: Date.now() }];
    const info = mgr.getUsageInfo(messages);
    expect(info.maxTokens).toBe(1000);
    expect(info.usedTokens).toBeGreaterThan(0);
    expect(info.usagePercent).toBeGreaterThanOrEqual(0);
  });
});

// ===== STATE STORE TESTS =====

describe('InMemoryStateStore', () => {
  test('put and get', async () => {
    const store = new InMemoryStateStore();
    await store.put('s1', { messages: ['hello'] });
    const data = await store.get('s1');
    expect(data).toEqual({ messages: ['hello'] });
  });

  test('returns null for missing', async () => {
    const store = new InMemoryStateStore();
    expect(await store.get('nonexistent')).toBeNull();
  });

  test('delete removes entry', async () => {
    const store = new InMemoryStateStore();
    await store.put('s1', { x: 1 });
    await store.delete('s1');
    expect(await store.get('s1')).toBeNull();
  });
});

// ===== OBSERVABILITY TESTS =====

describe('Tracer', () => {
  test('records spans', async () => {
    const tracer = new Tracer();
    await tracer.span('test.op', async (span) => { span.setAttribute('key', 'value'); });
    const summary = tracer.getTraceSummary();
    expect(summary.spanCount).toBe(1);
    expect(summary.errors).toBe(0);
  });

  test('records errors', async () => {
    const tracer = new Tracer();
    try { await tracer.span('fail', async () => { throw new Error('boom'); }); } catch {}
    expect(tracer.getTraceSummary().errors).toBe(1);
  });
});

describe('MetricsEmitter', () => {
  test('records metrics', () => {
    const emitter = new MetricsEmitter();
    emitter.record('test.metric', 42);
    expect(emitter.pendingCount).toBe(1);
  });

  test('disabled emitter does nothing', () => {
    const emitter = new MetricsEmitter({ enabled: false });
    emitter.record('test', 1);
    expect(emitter.pendingCount).toBe(0);
  });
});

describe('CostTracker', () => {
  test('records and reports', () => {
    const tracker = new CostTracker({ dailyBudget: 10 });
    tracker.record({ timestamp: Date.now(), modelId: 'test', tokensIn: 100, tokensOut: 50, costUsd: 0.01 });
    const report = tracker.getReport();
    expect(report.totalRequests).toBe(1);
    expect(report.totalCostUsd).toBe(0.01);
  });

  test('checkBudget detects exceeded', () => {
    const tracker = new CostTracker({ dailyBudget: 0.001 });
    tracker.record({ timestamp: Date.now(), modelId: 'test', tokensIn: 100, tokensOut: 50, costUsd: 0.01 });
    expect(tracker.checkBudget().exceeded).toBe(true);
  });
});

describe('Experiment (A/B)', () => {
  test('creates with valid weights', () => {
    const exp = new Experiment({ name: 'test', variants: [{ name: 'a', weight: 0.5 }, { name: 'b', weight: 0.5 }] });
    expect(exp.variants.length).toBe(2);
  });

  test('rejects invalid weights', () => {
    expect(() => new Experiment({ name: 'bad', variants: [{ name: 'a', weight: 0.3 }] })).toThrow();
  });

  test('sticky sessions return same variant', () => {
    const exp = new Experiment({ name: 'test', variants: [{ name: 'a', weight: 0.5 }, { name: 'b', weight: 0.5 }] });
    const v1 = exp.selectVariant('user-123');
    const v2 = exp.selectVariant('user-123');
    expect(v1.name).toBe(v2.name);
  });
});

describe('PromptAnalytics', () => {
  test('records and reports', () => {
    const analytics = new PromptAnalytics();
    analytics.record('summarize', { costUsd: 0.001, latencyMs: 200 });
    analytics.record('summarize', { costUsd: 0.002, latencyMs: 300 });
    const report = analytics.getReport('summarize');
    expect(report?.invocations).toBe(2);
    expect(report?.avgCost).toBeCloseTo(0.0015);
  });
});

describe('CostAwareRouter', () => {
  test('selects cheap model when budget high', () => {
    const router = new CostAwareRouter({ strategy: 'cost-optimized' });
    const model = router.select('short prompt', 0.9);
    expect(model).toContain('haiku');
  });

  test('rule-based routing', () => {
    const router = new CostAwareRouter({
      rules: [{ condition: 'input_tokens < 50', model: 'fast', priority: 1 }],
    });
    const model = router.select('hi', 0);
    expect(model).toContain('haiku');
  });
});

// ===== TOOL SANDBOX TESTS =====

describe('ToolSandbox', () => {
  test('executes function', async () => {
    const sandbox = new ToolSandbox({ maxExecutionTimeMs: 5000 });
    const result = await sandbox.execute(() => 2 + 2);
    expect(result).toBe(4);
  });

  test('validates allowed actions', () => {
    const sandbox = new ToolSandbox({ allowedActions: ['dynamodb:GetItem', 's3:*'], deniedActions: ['s3:DeleteBucket'] });
    expect(sandbox.validateAction('dynamodb:GetItem')).toBe(true);
    expect(sandbox.validateAction('s3:GetObject')).toBe(true);
    expect(sandbox.validateAction('s3:DeleteBucket')).toBe(false);
    expect(sandbox.validateAction('ec2:RunInstances')).toBe(false);
  });
});

// ===== MIDDLEWARE TESTS =====

describe('LoggingMiddleware', () => {
  test('passes event through', () => {
    const mw = new LoggingMiddleware();
    const event = { path: '/test' };
    const result = mw.beforeInvoke(event, {} as any);
    expect(result).toEqual(event);
  });
});

// ===== MOCK PROVIDER TESTS =====

describe('MockProvider', () => {
  test('returns configured responses', async () => {
    const mock = new MockProvider(['Hello', 'World']);
    const r1 = await mock.invoke('test1');
    const r2 = await mock.invoke('test2');
    expect(r1.content).toBe('Hello');
    expect(r2.content).toBe('World');
    expect(mock.totalCalls).toBe(2);
  });
});

describe('MockLambdaContext', () => {
  test('returns remaining time', () => {
    const ctx = new MockLambdaContext(15000);
    expect(ctx.getRemainingTimeInMillis()).toBe(15000);
  });
});

// ===== GOLDEN DATASET TESTS =====

describe('GoldenDatasetRunner', () => {
  test('validates expected_contains', async () => {
    const runner = new GoldenDatasetRunner();
    const handler = async (event: any) => ({ statusCode: 200, body: JSON.stringify({ result: 'Hello world test' }) });

    const result = await runner.run(
      [{ name: 'test1', input: { text: 'hi' }, expectedContains: ['hello'] }],
      handler
    );

    expect(result.passed).toBe(1);
    expect(result.passRate).toBe(1);
  });

  test('fails on missing content', async () => {
    const runner = new GoldenDatasetRunner();
    const handler = async () => ({ statusCode: 200, body: JSON.stringify({ result: 'no match' }) });

    const result = await runner.run(
      [{ name: 'test1', input: {}, expectedContains: ['xyz123'] }],
      handler
    );

    expect(result.failed).toBe(1);
  });
});
