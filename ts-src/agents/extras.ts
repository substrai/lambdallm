/**
 * Multi-agent router, tool sandboxing, and async tool dispatch.
 */

import { Agent, AgentResult, ToolDefinition } from './agent';
import { LambdaLLMContext } from '../core/context';

// ===== MULTI-AGENT ROUTER =====

export interface RouteConfig {
  agent: Agent;
  description: string;
  keywords: string[];
  priority?: number;
}

export class AgentRouter {
  private routes: RouteConfig[];
  private fallback?: Agent;
  private strategy: 'keyword' | 'llm';

  constructor(options: { routes: RouteConfig[]; fallback?: Agent; strategy?: 'keyword' | 'llm' }) {
    this.routes = options.routes.sort((a, b) => (b.priority || 0) - (a.priority || 0));
    this.fallback = options.fallback;
    this.strategy = options.strategy || 'keyword';
  }

  async route(query: string, context: LambdaLLMContext): Promise<AgentResult> {
    let agent: Agent | undefined;

    if (this.strategy === 'keyword') {
      agent = this.routeByKeyword(query);
    } else {
      agent = await this.routeByLLM(query, context);
    }

    if (!agent) agent = this.fallback;
    if (!agent) throw new Error('No agent matched and no fallback configured');

    return agent.run(query, context);
  }

  private routeByKeyword(query: string): Agent | undefined {
    const lower = query.toLowerCase();
    for (const route of this.routes) {
      for (const keyword of route.keywords) {
        if (lower.includes(keyword.toLowerCase())) return route.agent;
      }
    }
    return undefined;
  }

  private async routeByLLM(query: string, context: LambdaLLMContext): Promise<Agent | undefined> {
    const descriptions = this.routes.map(r => `- ${r.agent.name}: ${r.description}`).join('\n');
    const prompt = `Classify this query into one of these categories:\n${descriptions}\n\nQuery: ${query}\n\nRespond with ONLY the agent name.`;
    try {
      const response = await context.invoke(prompt);
      const name = response.trim().toLowerCase();
      const match = this.routes.find(r => r.agent.name.toLowerCase() === name);
      return match?.agent;
    } catch { return undefined; }
  }
}

// ===== TOOL SANDBOX =====

export interface SandboxPolicy {
  allowedActions: string[];
  deniedActions: string[];
  maxExecutionTimeMs: number;
}

export class ToolSandbox {
  private policy: SandboxPolicy;

  constructor(policy?: Partial<SandboxPolicy>) {
    this.policy = {
      allowedActions: policy?.allowedActions || [],
      deniedActions: policy?.deniedActions || [],
      maxExecutionTimeMs: policy?.maxExecutionTimeMs || 30000,
    };
  }

  async execute<T>(fn: () => T | Promise<T>): Promise<T> {
    const start = Date.now();
    const result = await Promise.race([
      Promise.resolve(fn()),
      new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error(`Tool execution exceeded timeout (${this.policy.maxExecutionTimeMs}ms)`)), this.policy.maxExecutionTimeMs)
      ),
    ]);
    return result;
  }

  validateAction(action: string): boolean {
    if (this.policy.deniedActions.includes(action)) return false;
    if (this.policy.allowedActions.length === 0) return true;
    for (const allowed of this.policy.allowedActions) {
      if (allowed === '*') return true;
      if (allowed.endsWith('*') && action.startsWith(allowed.slice(0, -1))) return true;
      if (action === allowed) return true;
    }
    return false;
  }
}

// ===== ASYNC TOOL DISPATCH =====

export interface AsyncToolRequest { requestId: string; toolName: string; toolInput: Record<string, any>; }

export class AsyncToolDispatcher {
  private queueUrl?: string;
  private client: any = null;

  constructor(options: { queueUrl?: string } = {}) {
    this.queueUrl = options.queueUrl;
  }

  async dispatch(toolName: string, toolInput: Record<string, any>): Promise<AsyncToolRequest> {
    const requestId = this.generateId();
    const request: AsyncToolRequest = { requestId, toolName, toolInput };

    if (this.queueUrl) {
      await this.sendToSQS(request);
    }

    console.log(JSON.stringify({ event: 'async_tool.dispatched', ...request }));
    return request;
  }

  private async sendToSQS(request: AsyncToolRequest): Promise<void> {
    try {
      const { SQSClient, SendMessageCommand } = require('@aws-sdk/client-sqs');
      if (!this.client) this.client = new SQSClient({});
      await this.client.send(new SendMessageCommand({
        QueueUrl: this.queueUrl,
        MessageBody: JSON.stringify(request),
      }));
    } catch (e: any) {
      throw new Error(`Failed to dispatch async tool: ${e.message}`);
    }
  }

  private generateId(): string {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = Math.random() * 16 | 0;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }
}

// ===== STREAMING =====

export class StreamingResponse {
  private chunks: string[] = [];
  private generator: AsyncGenerator<string>;

  constructor(generator: AsyncGenerator<string>) {
    this.generator = generator;
  }

  async *[Symbol.asyncIterator](): AsyncGenerator<string> {
    for await (const chunk of this.generator) {
      this.chunks.push(chunk);
      yield chunk;
    }
  }

  async getFullText(): Promise<string> {
    for await (const chunk of this.generator) { this.chunks.push(chunk); }
    return this.chunks.join('');
  }

  toLambdaStream(): AsyncGenerator<Buffer> {
    const self = this;
    return (async function* () {
      yield Buffer.from(JSON.stringify({ type: 'metadata' }) + '\n');
      for await (const chunk of self.generator) {
        self.chunks.push(chunk);
        yield Buffer.from(JSON.stringify({ type: 'content', text: chunk }) + '\n');
      }
      yield Buffer.from(JSON.stringify({ type: 'complete' }) + '\n');
    })();
  }
}
