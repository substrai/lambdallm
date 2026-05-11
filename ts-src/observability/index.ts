/**
 * Observability system for LambdaLLM - Tracing, Metrics, Cost, A/B, Analytics.
 */

// ===== TRACER (X-Ray) =====

export interface Span {
  name: string;
  startTime: number;
  endTime?: number;
  status: 'ok' | 'error';
  attributes: Record<string, any>;
  durationMs: number;
}

export class Tracer {
  private spans: Span[] = [];
  private enabled: boolean;

  constructor(options: { enabled?: boolean } = {}) {
    this.enabled = options.enabled !== false;
  }

  async span<T>(name: string, fn: (span: { setAttribute: (k: string, v: any) => void }) => Promise<T>): Promise<T> {
    if (!this.enabled) return fn({ setAttribute: () => {} });

    const start = Date.now();
    const attributes: Record<string, any> = {};
    const spanObj = { setAttribute: (k: string, v: any) => { attributes[k] = v; } };

    try {
      const result = await fn(spanObj);
      this.spans.push({ name, startTime: start, endTime: Date.now(), status: 'ok', attributes, durationMs: Date.now() - start });
      return result;
    } catch (e: any) {
      this.spans.push({ name, startTime: start, endTime: Date.now(), status: 'error', attributes: { ...attributes, error: e.message }, durationMs: Date.now() - start });
      throw e;
    }
  }

  getTraceSummary(): { spanCount: number; totalDurationMs: number; errors: number; spans: Span[] } {
    return {
      spanCount: this.spans.length,
      totalDurationMs: this.spans.reduce((sum, s) => sum + s.durationMs, 0),
      errors: this.spans.filter(s => s.status === 'error').length,
      spans: this.spans,
    };
  }
}

// ===== METRICS EMITTER (CloudWatch) =====

interface MetricPoint { name: string; value: number; unit: string; dimensions: Record<string, string>; timestamp: number; }

export class MetricsEmitter {
  private buffer: MetricPoint[] = [];
  private namespace: string;
  private enabled: boolean;

  constructor(options: { namespace?: string; enabled?: boolean } = {}) {
    this.namespace = options.namespace || 'LambdaLLM';
    this.enabled = options.enabled !== false;
  }

  record(name: string, value: number, unit = 'None', dimensions: Record<string, string> = {}): void {
    if (!this.enabled) return;
    this.buffer.push({ name, value, unit, dimensions, timestamp: Date.now() });
  }

  recordModelInvocation(modelId: string, tokensIn: number, tokensOut: number, latencyMs: number, costUsd: number): void {
    const dims = { ModelId: modelId };
    this.record('model.invocations', 1, 'Count', dims);
    this.record('model.tokens_in', tokensIn, 'Count', dims);
    this.record('model.tokens_out', tokensOut, 'Count', dims);
    this.record('model.latency', latencyMs, 'Milliseconds', dims);
    this.record('model.cost_usd', costUsd, 'None', dims);
  }

  async flush(): Promise<void> {
    if (!this.buffer.length) return;
    // EMF format log (CloudWatch picks these up automatically)
    for (const point of this.buffer) {
      console.log(JSON.stringify({
        _aws: { Timestamp: point.timestamp, CloudWatchMetrics: [{ Namespace: this.namespace, Dimensions: [Object.keys(point.dimensions)], Metrics: [{ Name: point.name, Unit: point.unit }] }] },
        [point.name]: point.value,
        ...point.dimensions,
      }));
    }
    this.buffer = [];
  }

  get pendingCount(): number { return this.buffer.length; }
}

// ===== COST TRACKER =====

export interface CostEntry { timestamp: number; modelId: string; tokensIn: number; tokensOut: number; costUsd: number; handlerName?: string; }

export interface CostReport { period: string; totalCostUsd: number; totalRequests: number; byModel: Record<string, number>; budgetUsd: number; utilizationPercent: number; }

export class CostTracker {
  private dailyBudget: number;
  private monthlyBudget: number;
  private onExceeded: 'block' | 'downgrade' | 'alert';
  private entries: CostEntry[] = [];

  constructor(options: { dailyBudget?: number; monthlyBudget?: number; onExceeded?: 'block' | 'downgrade' | 'alert' } = {}) {
    this.dailyBudget = options.dailyBudget || 50;
    this.monthlyBudget = options.monthlyBudget || 1000;
    this.onExceeded = options.onExceeded || 'block';
  }

  record(entry: CostEntry): void { this.entries.push(entry); }

  getDailySpend(): number {
    const today = new Date().toISOString().split('T')[0];
    return this.entries.filter(e => new Date(e.timestamp).toISOString().startsWith(today)).reduce((sum, e) => sum + e.costUsd, 0);
  }

  checkBudget(): { exceeded: boolean; dailySpend: number; utilization: number } {
    const spend = this.getDailySpend();
    return { exceeded: spend >= this.dailyBudget, dailySpend: spend, utilization: spend / this.dailyBudget };
  }

  getReport(): CostReport {
    const byModel: Record<string, number> = {};
    let total = 0;
    for (const e of this.entries) { byModel[e.modelId] = (byModel[e.modelId] || 0) + e.costUsd; total += e.costUsd; }
    return { period: 'daily', totalCostUsd: total, totalRequests: this.entries.length, byModel, budgetUsd: this.dailyBudget, utilizationPercent: (total / this.dailyBudget) * 100 };
  }

  forecastMonthly(): { projected: number; onTrack: boolean } {
    const daily = this.getDailySpend();
    const projected = daily * 30;
    return { projected, onTrack: projected <= this.monthlyBudget };
  }
}

// ===== A/B TESTING =====

export interface Variant { name: string; weight: number; promptTemplate?: string; modelId?: string; }

export class Experiment {
  public name: string;
  public variants: Variant[];
  public status: 'active' | 'paused' | 'completed' = 'active';

  constructor(options: { name: string; variants: Variant[] }) {
    this.name = options.name;
    this.variants = options.variants;
    const totalWeight = this.variants.reduce((sum, v) => sum + v.weight, 0);
    if (Math.abs(totalWeight - 1.0) > 0.01) throw new Error('Variant weights must sum to 1.0');
  }

  selectVariant(userId?: string): Variant {
    const hash = userId ? Math.abs(this.hashCode(`${this.name}:${userId}`)) % 1000 / 1000 : Math.random();
    let cumulative = 0;
    for (const v of this.variants) { cumulative += v.weight; if (hash <= cumulative) return v; }
    return this.variants[this.variants.length - 1];
  }

  recordResult(variantName: string, metrics: { latencyMs?: number; costUsd?: number; error?: boolean }): void {
    console.log(JSON.stringify({ event: 'ab_test.result', experiment: this.name, variant: variantName, ...metrics }));
  }

  private hashCode(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) { hash = ((hash << 5) - hash) + str.charCodeAt(i); hash |= 0; }
    return hash;
  }
}

// ===== PROMPT ANALYTICS =====

export interface PromptMetrics { promptName: string; invocations: number; totalCost: number; totalLatency: number; errors: number; avgCost: number; avgLatency: number; successRate: number; }

export class PromptAnalytics {
  private metrics: Map<string, { invocations: number; totalCost: number; totalLatency: number; errors: number }> = new Map();

  record(promptName: string, data: { costUsd: number; latencyMs: number; error?: boolean }): void {
    const existing = this.metrics.get(promptName) || { invocations: 0, totalCost: 0, totalLatency: 0, errors: 0 };
    existing.invocations++;
    existing.totalCost += data.costUsd;
    existing.totalLatency += data.latencyMs;
    if (data.error) existing.errors++;
    this.metrics.set(promptName, existing);
    console.log(JSON.stringify({ event: 'prompt.invocation', prompt_name: promptName, ...data }));
  }

  getReport(promptName: string): PromptMetrics | null {
    const m = this.metrics.get(promptName);
    if (!m) return null;
    return {
      promptName, invocations: m.invocations, totalCost: m.totalCost, totalLatency: m.totalLatency, errors: m.errors,
      avgCost: m.totalCost / m.invocations, avgLatency: m.totalLatency / m.invocations,
      successRate: (m.invocations - m.errors) / m.invocations,
    };
  }

  getSuggestions(promptName: string): string[] {
    const report = this.getReport(promptName);
    if (!report) return [];
    const suggestions: string[] = [];
    if (report.avgCost > 0.01) suggestions.push('High cost per call. Consider a cheaper model.');
    if (report.avgLatency > 5000) suggestions.push('High latency. Consider a faster model or reduce max_tokens.');
    if (report.successRate < 0.95) suggestions.push('Success rate below 95%. Check error logs.');
    return suggestions;
  }
}

// ===== COST-AWARE ROUTER =====

export interface RoutingRule { condition: string; model: string; priority?: number; }

export class CostAwareRouter {
  private strategy: 'cost-optimized' | 'quality-first' | 'balanced';
  private rules: RoutingRule[];

  constructor(options: { strategy?: 'cost-optimized' | 'quality-first' | 'balanced'; rules?: RoutingRule[] } = {}) {
    this.strategy = options.strategy || 'cost-optimized';
    this.rules = (options.rules || []).sort((a, b) => (b.priority || 0) - (a.priority || 0));
  }

  select(prompt: string, budgetUtilization: number): string {
    const inputTokens = Math.ceil(prompt.length / 4);
    const ctx = { input_tokens: inputTokens, budget_consumed: budgetUtilization };

    for (const rule of this.rules) {
      if (this.evaluateCondition(rule.condition, ctx)) {
        return this.resolveAlias(rule.model);
      }
    }

    if (this.strategy === 'cost-optimized' || budgetUtilization > 0.8) {
      return 'anthropic.claude-3-haiku-20240307-v1:0';
    }
    return 'anthropic.claude-3-sonnet-20240229-v1:0';
  }

  private evaluateCondition(condition: string, ctx: Record<string, number>): boolean {
    const parts = condition.split(' ');
    if (parts.length !== 3) return false;
    const [varName, op, value] = parts;
    const varValue = ctx[varName] || 0;
    const threshold = parseFloat(value);
    if (op === '<') return varValue < threshold;
    if (op === '>') return varValue > threshold;
    if (op === '>=') return varValue >= threshold;
    if (op === '<=') return varValue <= threshold;
    return false;
  }

  private resolveAlias(alias: string): string {
    const aliases: Record<string, string> = {
      fast: 'anthropic.claude-3-haiku-20240307-v1:0',
      smart: 'anthropic.claude-3-sonnet-20240229-v1:0',
      powerful: 'anthropic.claude-3-5-sonnet-20241022-v2:0',
    };
    return aliases[alias] || alias;
  }
}
