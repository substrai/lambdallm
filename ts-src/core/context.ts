/**
 * LambdaLLM Context - passed to handler functions.
 */

import { ModelConfig, ModelResponse, calculateCost } from './models';
import { ModelInvocationError, TimeoutError } from './exceptions';

export class LambdaLLMContext {
  public model: ModelConfig;
  public timeoutBuffer: number;
  public maxRetries: number;
  public fallbackModel?: ModelConfig;
  private lambdaContext: any;
  private _totalCost = 0;
  private _invocationCount = 0;
  private _provider: any = null;

  constructor(options: {
    model: ModelConfig;
    timeoutBuffer?: number;
    maxRetries?: number;
    fallbackModel?: ModelConfig;
    lambdaContext?: any;
  }) {
    this.model = options.model;
    this.timeoutBuffer = options.timeoutBuffer || 5;
    this.maxRetries = options.maxRetries || 3;
    this.fallbackModel = options.fallbackModel;
    this.lambdaContext = options.lambdaContext;
  }

  get remainingTimeMs(): number {
    if (this.lambdaContext?.getRemainingTimeInMillis) {
      return this.lambdaContext.getRemainingTimeInMillis();
    }
    return 900_000;
  }

  get shouldCheckpoint(): boolean {
    return this.remainingTimeMs < this.timeoutBuffer * 1000;
  }

  get totalCost(): number {
    return this._totalCost;
  }

  get invocationCount(): number {
    return this._invocationCount;
  }

  async invoke(prompt: string, variables?: Record<string, any>): Promise<string> {
    if (this.shouldCheckpoint) {
      throw new TimeoutError('Approaching Lambda timeout');
    }

    const formatted = variables ? this.formatPrompt(prompt, variables) : prompt;
    const response = await this.invokeWithRetry(formatted, this.model);

    this._invocationCount++;
    this._totalCost += response.costUsd;

    return response.content;
  }

  async invokeStructured(prompt: string, variables?: Record<string, any>): Promise<any> {
    const schemaInstruction = '\n\nRespond in valid JSON format only.';
    const formatted = variables ? this.formatPrompt(prompt, variables) : prompt;
    const responseText = await this.invoke(formatted + schemaInstruction);

    try {
      let jsonStr = responseText;
      if (jsonStr.includes('```json')) {
        jsonStr = jsonStr.split('```json')[1].split('```')[0];
      } else if (jsonStr.includes('```')) {
        jsonStr = jsonStr.split('```')[1].split('```')[0];
      }
      return JSON.parse(jsonStr.trim());
    } catch {
      // Retry with explicit instruction
      const retryResponse = await this.invoke(formatted + '\n\nReturn ONLY valid JSON.');
      return JSON.parse(retryResponse.trim());
    }
  }

  getRawClient(service = 'bedrock-runtime'): any {
    // Escape hatch: return raw AWS SDK client
    try {
      const { BedrockRuntimeClient } = require('@aws-sdk/client-bedrock-runtime');
      return new BedrockRuntimeClient({ region: this.model.region || 'us-east-1' });
    } catch {
      throw new Error('Install @aws-sdk/client-bedrock-runtime for raw client access');
    }
  }

  private formatPrompt(template: string, variables: Record<string, any>): string {
    let result = template;
    for (const [key, value] of Object.entries(variables)) {
      result = result.replace(new RegExp(`\\{${key}\\}`, 'g'), String(value));
    }
    return result;
  }

  private async invokeWithRetry(prompt: string, config: ModelConfig): Promise<ModelResponse> {
    let lastError: Error | null = null;

    for (let attempt = 0; attempt < this.maxRetries; attempt++) {
      try {
        return await this.callProvider(prompt, config);
      } catch (e: any) {
        lastError = e;
        if (e instanceof ModelInvocationError && !e.retryable) throw e;
        const waitMs = Math.min(Math.pow(2, attempt) * 500, 10000);
        await new Promise(resolve => setTimeout(resolve, waitMs));
      }
    }

    if (this.fallbackModel) {
      try {
        return await this.callProvider(prompt, this.fallbackModel);
      } catch { /* fallback also failed */ }
    }

    throw new ModelInvocationError(
      `All ${this.maxRetries} retries exhausted. Last error: ${lastError?.message}`,
      false
    );
  }

  private async callProvider(prompt: string, config: ModelConfig): Promise<ModelResponse> {
    const start = Date.now();

    try {
      const { BedrockRuntimeClient, InvokeModelCommand } = require('@aws-sdk/client-bedrock-runtime');

      if (!this._provider) {
        this._provider = new BedrockRuntimeClient({ region: config.region || 'us-east-1' });
      }

      const body = this.buildRequestBody(prompt, config);
      const command = new InvokeModelCommand({
        modelId: config.modelId,
        body: JSON.stringify(body),
        contentType: 'application/json',
        accept: 'application/json',
      });

      const response = await this._provider.send(command);
      const responseBody = JSON.parse(new TextDecoder().decode(response.body));
      const { content, tokensIn, tokensOut } = this.parseResponse(responseBody, config.modelId);
      const latencyMs = Date.now() - start;
      const costUsd = calculateCost(config.modelId, tokensIn, tokensOut);

      return { content, modelId: config.modelId, tokensIn, tokensOut, latencyMs, costUsd };
    } catch (e: any) {
      if (e.name === 'ThrottlingException') {
        throw new ModelInvocationError(`Bedrock throttled: ${e.message}`, true);
      }
      throw new ModelInvocationError(`Bedrock invocation failed: ${e.message}`, true);
    }
  }

  private buildRequestBody(prompt: string, config: ModelConfig): any {
    const modelId = config.modelId;
    if (modelId.includes('anthropic')) {
      return {
        anthropic_version: 'bedrock-2023-05-31',
        max_tokens: config.maxTokens || 1024,
        temperature: config.temperature || 0.7,
        messages: [{ role: 'user', content: prompt }],
      };
    } else if (modelId.includes('amazon.titan')) {
      return {
        inputText: prompt,
        textGenerationConfig: {
          maxTokenCount: config.maxTokens || 1024,
          temperature: config.temperature || 0.7,
        },
      };
    } else if (modelId.includes('meta.llama')) {
      return { prompt, max_gen_len: config.maxTokens || 1024, temperature: config.temperature || 0.7 };
    }
    return {
      anthropic_version: 'bedrock-2023-05-31',
      max_tokens: config.maxTokens || 1024,
      messages: [{ role: 'user', content: prompt }],
    };
  }

  private parseResponse(body: any, modelId: string): { content: string; tokensIn: number; tokensOut: number } {
    if (modelId.includes('anthropic')) {
      return {
        content: body.content[0].text,
        tokensIn: body.usage?.input_tokens || 0,
        tokensOut: body.usage?.output_tokens || 0,
      };
    } else if (modelId.includes('amazon.titan')) {
      return {
        content: body.results[0].outputText,
        tokensIn: body.inputTextTokenCount || 0,
        tokensOut: body.results?.[0]?.tokenCount || 0,
      };
    } else if (modelId.includes('meta.llama')) {
      return {
        content: body.generation,
        tokensIn: body.prompt_token_count || 0,
        tokensOut: body.generation_token_count || 0,
      };
    }
    return { content: String(body), tokensIn: 0, tokensOut: 0 };
  }
}
