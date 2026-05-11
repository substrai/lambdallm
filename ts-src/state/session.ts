/**
 * Session state management for LambdaLLM.
 */

export enum MemoryStrategy {
  FULL_HISTORY = 'full_history',
  SLIDING_WINDOW = 'sliding_window',
  SUMMARY = 'summary',
}

export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: number;
}

export interface SessionOptions {
  store?: 'memory' | 'dynamodb';
  ttlHours?: number;
  maxMessages?: number;
  memory?: MemoryStrategy;
}

export class Session {
  public sessionId?: string;
  public messages: Message[] = [];
  public store: string;
  public ttlHours: number;
  public maxMessages: number;
  public memory: MemoryStrategy;
  private dirty = false;

  constructor(options: SessionOptions = {}) {
    this.store = options.store || 'memory';
    this.ttlHours = options.ttlHours || 24;
    this.maxMessages = options.maxMessages || 20;
    this.memory = options.memory || MemoryStrategy.SLIDING_WINDOW;
  }

  addMessage(role: Message['role'], content: string): void {
    this.messages.push({ role, content, timestamp: Date.now() });
    this.applyMemoryStrategy();
    this.dirty = true;
  }

  getHistory(): Array<{ role: string; content: string }> {
    return this.messages.map(m => ({ role: m.role, content: m.content }));
  }

  formatHistory(): string {
    return this.messages.map(m => `${m.role}: ${m.content}`).join('\n');
  }

  clear(): void {
    this.messages = [];
    this.dirty = true;
  }

  get messageCount(): number {
    return this.messages.length;
  }

  get lastMessage(): Message | undefined {
    return this.messages[this.messages.length - 1];
  }

  private applyMemoryStrategy(): void {
    if (this.memory === MemoryStrategy.FULL_HISTORY) return;
    if (this.messages.length > this.maxMessages) {
      this.messages = this.messages.slice(-this.maxMessages);
    }
  }
}
