/**
 * Type-safe Prompt template system.
 */

import { LambdaLLMContext } from './context';
import { ConfigurationError } from './exceptions';

export interface PromptOptions {
  template: string;
  inputSchema?: Record<string, string>;
  outputSchema?: Record<string, string>;
  name?: string;
  version?: string;
  maxTokens?: number;
  temperature?: number;
}

export class Prompt {
  public template: string;
  public inputSchema?: Record<string, string>;
  public outputSchema?: Record<string, string>;
  public name?: string;
  public version: string;
  public maxTokens: number;
  public temperature: number;
  private variables: string[];

  constructor(options: PromptOptions) {
    if (!options.template) throw new ConfigurationError('Prompt template cannot be empty');

    this.template = options.template;
    this.inputSchema = options.inputSchema;
    this.outputSchema = options.outputSchema;
    this.name = options.name;
    this.version = options.version || '1.0.0';
    this.maxTokens = options.maxTokens || 1024;
    this.temperature = options.temperature || 0.7;
    this.variables = this.extractVariables();
  }

  format(variables: Record<string, any>): string {
    this.validateInputs(variables);
    let result = this.template;
    for (const [key, value] of Object.entries(variables)) {
      result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), String(value));
    }
    return result;
  }

  async invoke(context: LambdaLLMContext, variables: Record<string, any>): Promise<any> {
    const formatted = this.format(variables);
    if (this.outputSchema) {
      return context.invokeStructured(formatted);
    }
    return context.invoke(formatted);
  }

  private extractVariables(): string[] {
    const matches = this.template.match(/(?<!\{)\{([a-zA-Z_][a-zA-Z0-9_]*)\}(?!\})/g) || [];
    return matches.map(m => m.slice(1, -1));
  }

  private validateInputs(variables: Record<string, any>): void {
    const provided = new Set(Object.keys(variables));
    const required = new Set(this.variables);
    const missing = [...required].filter(v => !provided.has(v));
    if (missing.length > 0) {
      throw new ConfigurationError(`Missing required variables: ${missing.join(', ')}`);
    }
  }
}
