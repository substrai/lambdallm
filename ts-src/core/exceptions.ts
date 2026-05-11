/**
 * Exception hierarchy for LambdaLLM.
 */

export class LambdaLLMError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'LambdaLLMError';
  }
}

export class ModelInvocationError extends LambdaLLMError {
  public retryable: boolean;

  constructor(message: string, retryable = true) {
    super(message);
    this.name = 'ModelInvocationError';
    this.retryable = retryable;
  }
}

export class TimeoutError extends LambdaLLMError {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

export class BudgetExceededError extends LambdaLLMError {
  constructor(message: string) {
    super(message);
    this.name = 'BudgetExceededError';
  }
}

export class ConfigurationError extends LambdaLLMError {
  constructor(message: string) {
    super(message);
    this.name = 'ConfigurationError';
  }
}
