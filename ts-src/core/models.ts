/**
 * Model definitions and registry for LambdaLLM.
 */

export enum Model {
  CLAUDE_3_HAIKU = 'anthropic.claude-3-haiku-20240307-v1:0',
  CLAUDE_3_SONNET = 'anthropic.claude-3-sonnet-20240229-v1:0',
  CLAUDE_3_5_SONNET = 'anthropic.claude-3-5-sonnet-20241022-v2:0',
  CLAUDE_3_OPUS = 'anthropic.claude-3-opus-20240229-v1:0',
  TITAN_TEXT_EXPRESS = 'amazon.titan-text-express-v1',
  TITAN_TEXT_LITE = 'amazon.titan-text-lite-v1',
  LLAMA3_8B = 'meta.llama3-8b-instruct-v1:0',
  LLAMA3_70B = 'meta.llama3-70b-instruct-v1:0',
}

export interface ModelConfig {
  modelId: string;
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  region?: string;
}

export interface ModelResponse {
  content: string;
  modelId: string;
  tokensIn: number;
  tokensOut: number;
  latencyMs: number;
  costUsd: number;
}

export const MODEL_COSTS: Record<string, { input: number; output: number }> = {
  [Model.CLAUDE_3_HAIKU]: { input: 0.00025, output: 0.00125 },
  [Model.CLAUDE_3_SONNET]: { input: 0.003, output: 0.015 },
  [Model.CLAUDE_3_5_SONNET]: { input: 0.003, output: 0.015 },
  [Model.CLAUDE_3_OPUS]: { input: 0.015, output: 0.075 },
  [Model.TITAN_TEXT_EXPRESS]: { input: 0.0002, output: 0.0006 },
  [Model.TITAN_TEXT_LITE]: { input: 0.00015, output: 0.0002 },
  [Model.LLAMA3_8B]: { input: 0.0003, output: 0.0006 },
  [Model.LLAMA3_70B]: { input: 0.00265, output: 0.0035 },
};

export function calculateCost(modelId: string, tokensIn: number, tokensOut: number): number {
  const costs = MODEL_COSTS[modelId] || { input: 0.001, output: 0.002 };
  return (tokensIn / 1000) * costs.input + (tokensOut / 1000) * costs.output;
}
