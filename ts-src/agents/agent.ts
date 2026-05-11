/**
 * Agent and Tool system for LambdaLLM.
 */

import { LambdaLLMContext } from '../core/context';
import { LambdaLLMError } from '../core/exceptions';

export interface ToolDefinition {
  name: string;
  description: string;
  parameters: Array<{ name: string; type: string; description: string; required: boolean }>;
  func: (...args: any[]) => any;
}

/**
 * Decorator-like function to define a tool.
 */
export function Tool(options: { description: string; name?: string }) {
  return function (func: Function): ToolDefinition {
    return {
      name: options.name || func.name,
      description: options.description,
      parameters: [],
      func: func as any,
    };
  };
}

export interface AgentResult {
  answer: string;
  status: 'completed' | 'max_iterations' | 'timeout' | 'cost_limit';
  totalIterations: number;
  totalToolCalls: number;
  totalCostUsd: number;
  totalLatencyMs: number;
  steps: Array<{ thought: string; toolName?: string; toolOutput?: string; isFinal: boolean }>;
}

export interface AgentOptions {
  name: string;
  systemPrompt: string;
  tools: ToolDefinition[];
  maxIterations?: number;
  timeoutBuffer?: number;
  maxCostUsd?: number;
}

export class Agent {
  public name: string;
  public systemPrompt: string;
  public tools: ToolDefinition[];
  public maxIterations: number;
  public timeoutBuffer: number;
  public maxCostUsd?: number;

  constructor(options: AgentOptions) {
    this.name = options.name;
    this.systemPrompt = options.systemPrompt;
    this.tools = options.tools;
    this.maxIterations = options.maxIterations || 10;
    this.timeoutBuffer = options.timeoutBuffer || 30;
    this.maxCostUsd = options.maxCostUsd;
  }

  async run(query: string, context: LambdaLLMContext): Promise<AgentResult> {
    const startTime = Date.now();
    const steps: AgentResult['steps'] = [];
    let toolCalls = 0;

    const toolDescriptions = this.tools.map(t => `- ${t.name}: ${t.description}`).join('\n');
    const systemContent = `${this.systemPrompt}\n\nTools available:\n${toolDescriptions}\n\nRespond with JSON: {"thought": "...", "tool_name": "...", "tool_input": {...}} or {"thought": "...", "final_answer": "..."}`;

    let messages = `System: ${systemContent}\n\nUser: ${query}\n\nAssistant:`;

    for (let i = 0; i < this.maxIterations; i++) {
      if (context.remainingTimeMs < this.timeoutBuffer * 1000) {
        return this.buildResult(steps, 'timeout', startTime, toolCalls, context.totalCost, 'Approaching timeout.');
      }

      if (this.maxCostUsd && context.totalCost >= this.maxCostUsd) {
        return this.buildResult(steps, 'cost_limit', startTime, toolCalls, context.totalCost, 'Cost limit reached.');
      }

      const response = await context.invoke(messages);
      const decision = this.parseDecision(response);

      if (decision.final_answer) {
        steps.push({ thought: decision.thought || '', isFinal: true });
        return this.buildResult(steps, 'completed', startTime, toolCalls, context.totalCost, decision.final_answer);
      }

      if (decision.tool_name) {
        const tool = this.tools.find(t => t.name === decision.tool_name);
        let toolOutput = 'Tool not found';
        if (tool) {
          try {
            toolOutput = String(tool.func(decision.tool_input || {}));
            toolCalls++;
          } catch (e: any) {
            toolOutput = `Error: ${e.message}`;
          }
        }

        steps.push({ thought: decision.thought || '', toolName: decision.tool_name, toolOutput, isFinal: false });
        messages += `\n${response}\nUser: Tool '${decision.tool_name}' returned: ${toolOutput}\nContinue.\nAssistant:`;
      }
    }

    return this.buildResult(steps, 'max_iterations', startTime, toolCalls, context.totalCost, 'Max iterations reached.');
  }

  private parseDecision(response: string): any {
    try {
      if (response.trim().startsWith('{')) return JSON.parse(response.trim());
      if (response.includes('```json')) {
        const json = response.split('```json')[1].split('```')[0];
        return JSON.parse(json.trim());
      }
      return { thought: 'Direct response', final_answer: response };
    } catch {
      return { thought: 'Could not parse', final_answer: response };
    }
  }

  private buildResult(
    steps: AgentResult['steps'], status: AgentResult['status'],
    startTime: number, toolCalls: number, cost: number, answer: string
  ): AgentResult {
    return {
      answer, status, steps,
      totalIterations: steps.length,
      totalToolCalls: toolCalls,
      totalCostUsd: cost,
      totalLatencyMs: Date.now() - startTime,
    };
  }
}
