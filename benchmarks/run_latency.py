#!/usr/bin/env python3
"""Benchmark: Invocation Latency Comparison

Measures latency for LambdaLLM vs LangChain vs raw boto3.
Requires AWS credentials with Bedrock access.

Usage: python3 run_latency.py
"""

import os
import sys
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import boto3

PROMPT = "Summarize in 20 words: AWS Lambda is a serverless compute service that runs code in response to events."
MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
NUM_CALLS = 10
DELAY = 0.5

print("Latency Benchmark")
print("=" * 50)

client = boto3.client("bedrock-runtime", region_name="us-east-1")
results = {}

# 1. Raw boto3
print(f"\n1. Raw boto3 ({NUM_CALLS} calls)...")
raw_latencies = []
for i in range(NUM_CALLS):
    start = time.time()
    response = client.invoke_model(
        modelId=MODEL_ID,
        body=json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 100, "messages": [{"role": "user", "content": PROMPT}]}),
        contentType="application/json",
    )
    json.loads(response["body"].read())
    raw_latencies.append((time.time() - start) * 1000)
    time.sleep(DELAY)

results["raw_boto3"] = {
    "avg_ms": round(sum(raw_latencies) / len(raw_latencies), 1),
    "min_ms": round(min(raw_latencies), 1),
    "max_ms": round(max(raw_latencies), 1),
    "p50_ms": round(sorted(raw_latencies)[len(raw_latencies)//2], 1),
    "samples": raw_latencies,
}
print(f"   Avg: {results['raw_boto3']['avg_ms']}ms")

# 2. LambdaLLM
print(f"\n2. LambdaLLM ({NUM_CALLS} calls)...")
from lambdallm import handler, Model

class MockCtx:
    function_name = "bench"
    def get_remaining_time_in_millis(self): return 30000

@handler(model=Model.CLAUDE_3_HAIKU)
def llm_handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})
    result = context.invoke(body.get("prompt", "hello"))
    return {"statusCode": 200, "body": {"result": result}}

llm_latencies = []
for i in range(NUM_CALLS):
    start = time.time()
    llm_handler({"body": json.dumps({"prompt": PROMPT})}, MockCtx())
    llm_latencies.append((time.time() - start) * 1000)
    time.sleep(DELAY)

results["lambdallm"] = {
    "avg_ms": round(sum(llm_latencies) / len(llm_latencies), 1),
    "min_ms": round(min(llm_latencies), 1),
    "max_ms": round(max(llm_latencies), 1),
    "p50_ms": round(sorted(llm_latencies)[len(llm_latencies)//2], 1),
    "samples": llm_latencies,
}
print(f"   Avg: {results['lambdallm']['avg_ms']}ms")

# 3. LangChain
print(f"\n3. LangChain ({NUM_CALLS} calls)...")
try:
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(model_id=MODEL_ID, region_name="us-east-1")
    lc_latencies = []
    for i in range(NUM_CALLS):
        start = time.time()
        llm.invoke(PROMPT)
        lc_latencies.append((time.time() - start) * 1000)
        time.sleep(DELAY)

    results["langchain"] = {
        "avg_ms": round(sum(lc_latencies) / len(lc_latencies), 1),
        "min_ms": round(min(lc_latencies), 1),
        "max_ms": round(max(lc_latencies), 1),
        "p50_ms": round(sorted(lc_latencies)[len(lc_latencies)//2], 1),
        "samples": lc_latencies,
    }
    print(f"   Avg: {results['langchain']['avg_ms']}ms")
except Exception as e:
    print(f"   Error: {e}")
    results["langchain"] = {"error": str(e)}

# Summary
print(f"\n{'='*50}")
print(f"{'Framework':<15} {'Avg (ms)':<12} {'Min':<10} {'Max':<10}")
print(f"{'-'*47}")
for name, data in results.items():
    if "error" not in data:
        print(f"{name:<15} {data['avg_ms']:<12} {data['min_ms']:<10} {data['max_ms']:<10}")

output = os.path.join(RESULTS_DIR, f"latency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output, "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "model": MODEL_ID, "num_calls": NUM_CALLS, "results": results}, f, indent=2)

print(f"\nResults: {output}")
