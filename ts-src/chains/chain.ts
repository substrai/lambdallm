/**
 * Multi-step chain pipeline for LambdaLLM.
 */

import { LambdaLLMContext } from '../core/context';
import { LambdaLLMError } from '../core/exceptions';

export interface StepOptions {
  name: string;
  prompt?: string;
  func?: (outputs: Record<string, any>) => any;
  condition?: (outputs: Record<string, any>) => boolean;
}

export class Step {
  public name: string;
  public prompt?: string;
  public func?: (outputs: Record<string, any>) => any;
  public condition?: (outputs: Record<string, any>) => boolean;

  constructor(options: StepOptions) {
    if (!options.prompt && !options.func) {
      throw new Error(`Step '${options.name}' must have either prompt or func`);
    }
    if (options.prompt && options.func) {
      throw new Error(`Step '${options.name}' cannot have both prompt and func`);
    }
    this.name = options.name;
    this.prompt = options.prompt;
    this.func = options.func;
    this.condition = options.condition;
  }

  get isLlmStep(): boolean { return !!this.prompt; }
  get isTransformStep(): boolean { return !!this.func; }
}

export interface ChainResult {
  chainName: string;
  status: 'completed' | 'checkpointed' | 'failed' | 'truncated';
  steps: Array<{ name: string; output: any; costUsd: number; latencyMs: number; skipped: boolean }>;
  finalOutput: any;
  totalCostUsd: number;
  totalLatencyMs: number;
  completedSteps: number;
  checkpoint?: any;
}

export interface ChainOptions {
  name: string;
  steps: Step[];
  timeoutStrategy?: 'fail-fast' | 'checkpoint' | 'truncate';
  maxTotalCost?: number;
}

export class Chain {
  public name: string;
  public steps: Step[];
  public timeoutStrategy: string;
  public maxTotalCost?: number;

  constructor(options: ChainOptions) {
    if (!options.steps.length) throw new Error('Chain must have at least one step');
    this.name = options.name;
    this.steps = options.steps;
    this.timeoutStrategy = options.timeoutStrategy || 'fail-fast';
    this.maxTotalCost = options.maxTotalCost;
  }

  async run(context: LambdaLLMContext, variables: Record<string, any>): Promise<ChainResult> {
    const startTime = Date.now();
    const outputs: Record<string, any> = { ...variables };
    const stepResults: ChainResult['steps'] = [];
    let totalCost = 0;

    for (let i = 0; i < this.steps.length; i++) {
      const step = this.steps[i];

      // Check timeout
      if (this.timeoutStrategy === 'checkpoint' && context.shouldCheckpoint) {
        return {
          chainName: this.name, status: 'checkpointed', steps: stepResults,
          finalOutput: stepResults[stepResults.length - 1]?.output,
          totalCostUsd: totalCost, totalLatencyMs: Date.now() - startTime,
          completedSteps: stepResults.filter(s => !s.skipped).length,
          checkpoint: { nextStepIndex: i, outputs },
        };
      }

      // Check cost
      if (this.maxTotalCost && totalCost >= this.maxTotalCost) {
        return {
          chainName: this.name, status: 'truncated', steps: stepResults,
          finalOutput: stepResults[stepResults.length - 1]?.output,
          totalCostUsd: totalCost, totalLatencyMs: Date.now() - startTime,
          completedSteps: stepResults.filter(s => !s.skipped).length,
        };
      }

      // Check condition
      if (step.condition && !step.condition(outputs)) {
        stepResults.push({ name: step.name, output: null, costUsd: 0, latencyMs: 0, skipped: true });
        continue;
      }

      const stepStart = Date.now();
      try {
        let output: any;
        if (step.isLlmStep) {
          const resolved = this.resolveVariables(step.prompt!, outputs);
          output = await context.invoke(resolved);
        } else if (step.func) {
          output = step.func(outputs);
        }

        const stepCost = context.totalCost - totalCost;
        totalCost = context.totalCost;
        outputs[step.name] = output;
        outputs[`${step.name}.output`] = output;

        stepResults.push({
          name: step.name, output, costUsd: stepCost,
          latencyMs: Date.now() - stepStart, skipped: false,
        });
      } catch (e: any) {
        stepResults.push({ name: step.name, output: null, costUsd: 0, latencyMs: Date.now() - stepStart, skipped: false });
        return {
          chainName: this.name, status: 'failed', steps: stepResults,
          finalOutput: null, totalCostUsd: totalCost, totalLatencyMs: Date.now() - startTime,
          completedSteps: stepResults.filter(s => !s.skipped).length,
        };
      }
    }

    return {
      chainName: this.name, status: 'completed', steps: stepResults,
      finalOutput: stepResults[stepResults.length - 1]?.output,
      totalCostUsd: totalCost, totalLatencyMs: Date.now() - startTime,
      completedSteps: stepResults.filter(s => !s.skipped).length,
    };
  }

  private resolveVariables(template: string, outputs: Record<string, any>): string {
    return template.replace(/\{([a-zA-Z_][a-zA-Z0-9_.]*)\}/g, (_, key) => {
      return key in outputs ? String(outputs[key]) : `{${key}}`;
    });
  }
}
