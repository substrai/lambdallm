/**
 * Context window manager - trims history to fit model limits.
 */

import { Message } from './session';

const MODEL_CONTEXT_LIMITS: Record<string, number> = {
  'anthropic.claude-3-haiku-20240307-v1:0': 200000,
  'anthropic.claude-3-sonnet-20240229-v1:0': 200000,
  'anthropic.claude-3-5-sonnet-20241022-v2:0': 200000,
  'amazon.titan-text-express-v1': 8000,
  'amazon.titan-text-lite-v1': 4000,
  'meta.llama3-8b-instruct-v1:0': 8000,
  'meta.llama3-70b-instruct-v1:0': 8000,
};

export class ContextWindowManager {
  private maxTokens?: number;
  private reserveTokens: number;
  private strategy: 'truncate_oldest' | 'smart';

  constructor(options: { maxTokens?: number; reserveTokens?: number; strategy?: 'truncate_oldest' | 'smart' } = {}) {
    this.maxTokens = options.maxTokens;
    this.reserveTokens = options.reserveTokens || 1024;
    this.strategy = options.strategy || 'truncate_oldest';
  }

  fitMessages(messages: Message[], modelId?: string, systemPrompt = ''): Message[] {
    const limit = this.maxTokens || MODEL_CONTEXT_LIMITS[modelId || ''] || 100000;
    const available = limit - this.reserveTokens - this.estimateTokens(systemPrompt);

    if (available <= 0) return messages.slice(-1);

    if (this.strategy === 'smart') return this.smartTrim(messages, available);
    return this.truncateOldest(messages, available);
  }

  private truncateOldest(messages: Message[], maxTokens: number): Message[] {
    const result: Message[] = [];
    let total = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
      const tokens = this.estimateTokens(messages[i].content);
      if (total + tokens > maxTokens) break;
      result.unshift(messages[i]);
      total += tokens;
    }
    return result.length > 0 ? result : [messages[messages.length - 1]];
  }

  private smartTrim(messages: Message[], maxTokens: number): Message[] {
    if (messages.length <= 3) return messages;
    const first = messages[0];
    const firstTokens = this.estimateTokens(first.content);
    const remaining = maxTokens - firstTokens;

    const recent: Message[] = [];
    let total = 0;
    for (let i = messages.length - 1; i >= 1; i--) {
      const tokens = this.estimateTokens(messages[i].content);
      if (total + tokens > remaining) break;
      recent.unshift(messages[i]);
      total += tokens;
    }
    return [first, ...recent];
  }

  private estimateTokens(text: string): number {
    return Math.ceil(text.length / 4);
  }

  getUsageInfo(messages: Message[], modelId?: string): { maxTokens: number; usedTokens: number; usagePercent: number } {
    const max = this.maxTokens || MODEL_CONTEXT_LIMITS[modelId || ''] || 100000;
    const used = messages.reduce((sum, m) => sum + this.estimateTokens(m.content), 0);
    return { maxTokens: max, usedTokens: used, usagePercent: Math.round((used / max) * 100) };
  }
}
