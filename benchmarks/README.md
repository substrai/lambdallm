# Benchmarks

Reproducible benchmark suite for LambdaLLM. All results were captured on May 11, 2026 using real AWS Bedrock API calls.

## Results Summary

| Metric | LambdaLLM | LangChain | Raw boto3 |
|--------|-----------|-----------|-----------|
| Package size (core) | **0.6 MB** | 139.4 MB | 23.8 MB |
| Avg latency (Claude 3 Haiku) | **617 ms** | 697 ms | 864 ms |
| Framework overhead | **-13.5%** (faster) | +12.9% | baseline |
| Cost per request | $0.000044 | $0.000044 | $0.000044 |
| Memory persistence | DynamoDB | Lost on Lambda | None |
| Timeout recovery | Checkpoint/resume | Crashes | None |

## Prerequisites

```bash
# 1. Python 3.10+
python3 --version

# 2. Install LambdaLLM
pip install "substrai-lambdallm[bedrock]"

# 3. Install comparison frameworks
pip install langchain langchain-aws

# 4. AWS credentials with Bedrock access
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-east-1"

# 5. Verify Bedrock access
aws bedrock list-foundation-models --region us-east-1 --query "modelSummaries[0].modelId"
```

## Enable Bedrock Models

Before running benchmarks, enable these models in AWS Console:
1. Go to: AWS Console > Amazon Bedrock > Model access
2. Enable:
   - Anthropic Claude 3 Haiku
   - Anthropic Claude 3 Sonnet
   - Amazon Titan Text Embeddings V2

## Running Benchmarks

### Full Benchmark Suite (all tests)

```bash
cd benchmarks/
python3 run_benchmarks.py
```

This runs 10 tests:
1. Package size comparison
2. Bedrock model latency (10 calls per model)
3. Framework overhead (LambdaLLM vs raw boto3)
4. Session state persistence
5. Chain execution (3-step pipeline)
6. Cost tracking accuracy
7. Prompt template validation
8. Agent tool system
9. Observability (tracer + metrics)
10. A/B testing

Results saved to: `results/full_benchmark_TIMESTAMP.json`

### Comparison Benchmark (LambdaLLM vs LangChain vs boto3)

```bash
python3 run_comparison.py
```

This runs:
- Package size comparison (5 configurations)
- Latency comparison (5 calls each framework)
- Titan Text Embeddings V2 test
- Feature-by-feature comparison
- Bedrock Agents analysis

Results saved to: `results/comparison_final_TIMESTAMP.json`

### Individual Benchmarks

```bash
# Package size only (no AWS calls needed)
python3 run_package_size.py

# Latency only (requires Bedrock access)
python3 run_latency.py

# Cold start (requires deployed Lambda)
bash run_coldstart.sh
```

## Interpreting Results

### Package Size
- Measured via `pip install <package> -t /tmp/dir` into isolated directories
- Represents the actual Lambda deployment package size
- LambdaLLM core (0.6MB) + boto3 (23.8MB) = 24.4MB total deployment

### Latency
- Wall-clock time from before API call to after response parsing
- Warm invocations only (first call excluded to eliminate TLS handshake)
- 500ms delay between calls to avoid throttling
- Same prompt used across all frameworks for fair comparison

### Framework Overhead
- Negative overhead (-13.5%) means LambdaLLM is FASTER than raw boto3
- Caused by module-level client caching and connection reuse
- Raw boto3 measurements may create fresh client references per call

### Cost
- Calculated from Bedrock published pricing (May 2026)
- Claude 3 Haiku: $0.00025/1K input tokens, $0.00125/1K output tokens
- Verified: framework reports $0.000015 for 14-in/9-out tokens
- Manual calculation: (14/1000)*0.00025 + (9/1000)*0.00125 = $0.0000148 (1.4% variance)

## Results Files

```
results/
├── full_benchmark_20260511_132912.json    # Complete 10-test suite
├── comparison_final_20260511_133848.json  # Framework comparison
└── README.md                              # This file
```

## Reproducing Results

Results will vary slightly due to:
- Network latency to AWS (your location vs us-east-1)
- Bedrock model load at time of testing
- AWS infrastructure variability

However, **relative comparisons** (LambdaLLM vs LangChain vs boto3) should remain consistent within the same benchmark run since all use the same prompt, model, and region.

## Environment Used for Published Results

```
Date: May 11, 2026
Machine: macOS (Apple Silicon, 16GB RAM)
Python: 3.14.4
boto3: 1.43.6
LambdaLLM: 1.0.1 (substrai-lambdallm)
LangChain: 1.2.18
langchain-aws: 1.4.6
Region: us-east-1
Models: Claude 3 Haiku, Claude 3 Sonnet, Titan Embed Text V2
```
