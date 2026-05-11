/**
 * DynamoDB state store for session persistence.
 */

export interface StateStore {
  get(sessionId: string): Promise<any | null>;
  put(sessionId: string, data: any, ttlSeconds?: number): Promise<void>;
  delete(sessionId: string): Promise<void>;
}

export class DynamoDBStateStore implements StateStore {
  private tableName: string;
  private client: any = null;

  constructor(options: { tableName?: string; region?: string } = {}) {
    this.tableName = options.tableName || 'lambdallm-sessions';
  }

  private getClient() {
    if (!this.client) {
      const { DynamoDBClient } = require('@aws-sdk/client-dynamodb');
      const { DynamoDBDocumentClient, GetCommand, PutCommand, DeleteCommand } = require('@aws-sdk/lib-dynamodb');
      const raw = new DynamoDBClient({});
      this.client = DynamoDBDocumentClient.from(raw);
    }
    return this.client;
  }

  async get(sessionId: string): Promise<any | null> {
    try {
      const { GetCommand } = require('@aws-sdk/lib-dynamodb');
      const client = this.getClient();
      const result = await client.send(new GetCommand({
        TableName: this.tableName,
        Key: { session_id: sessionId },
      }));
      if (!result.Item) return null;
      if (result.Item.expires_at && Date.now() / 1000 > result.Item.expires_at) return null;
      return JSON.parse(result.Item.data || '{}');
    } catch (e) {
      console.error(`Failed to get session ${sessionId}:`, e);
      return null;
    }
  }

  async put(sessionId: string, data: any, ttlSeconds = 86400): Promise<void> {
    try {
      const { PutCommand } = require('@aws-sdk/lib-dynamodb');
      const client = this.getClient();
      await client.send(new PutCommand({
        TableName: this.tableName,
        Item: {
          session_id: sessionId,
          data: JSON.stringify(data),
          expires_at: Math.floor(Date.now() / 1000) + ttlSeconds,
          updated_at: Math.floor(Date.now() / 1000),
        },
      }));
    } catch (e) {
      console.error(`Failed to save session ${sessionId}:`, e);
      throw e;
    }
  }

  async delete(sessionId: string): Promise<void> {
    try {
      const { DeleteCommand } = require('@aws-sdk/lib-dynamodb');
      const client = this.getClient();
      await client.send(new DeleteCommand({
        TableName: this.tableName,
        Key: { session_id: sessionId },
      }));
    } catch (e) {
      console.error(`Failed to delete session ${sessionId}:`, e);
    }
  }
}

export class InMemoryStateStore implements StateStore {
  private store: Map<string, { data: any; expiresAt: number }> = new Map();

  async get(sessionId: string): Promise<any | null> {
    const entry = this.store.get(sessionId);
    if (!entry) return null;
    if (Date.now() / 1000 > entry.expiresAt) { this.store.delete(sessionId); return null; }
    return entry.data;
  }

  async put(sessionId: string, data: any, ttlSeconds = 86400): Promise<void> {
    this.store.set(sessionId, { data, expiresAt: Date.now() / 1000 + ttlSeconds });
  }

  async delete(sessionId: string): Promise<void> {
    this.store.delete(sessionId);
  }

  clear(): void { this.store.clear(); }
}
