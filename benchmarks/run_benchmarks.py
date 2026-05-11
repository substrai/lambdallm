#!/usr/bin/env python3
"""
LambdaLLM Full Benchmark Suite
Runs all feature tests with real AWS Bedrock calls.
Results saved to ~/Developer/substrai/lambdallm/benchmarks/results/
"""

import json
import time
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.expanduser('~/Developer/substrai/lambdallm/src'))

RESULTS_DIR = os.path.expanduser('~/Developer/substrai/lambdallm/benchmarks/results')
os.makedirs(RESULTS_DIR, exist_ok=True)

results = {"timestamp": datetime.now().isoformat(), "region": "us-east-1", "tests": {}}

print("=" * 60)
print("LambdaLLM FULL BENCHMARK SUITE")
print("=" * 60)
print()

# ============================================
# TEST 1: Package Size Comparison
# ============================================
print("TEST 1: Package Size Comparison")
print("-" * 40)

import subprocess
import tempfile
import shutil

packages = {
    "substrai-lambdallm (core)": "substrai-lambdallm",
    "substrai-lambdallm[bedrock]": "substrai-lambdallm[bedrock]",
    "boto3 (raw)": "boto3",
}

pkg_results = {}
for label, pkg in packages.items():
    tmp = tempfile.mkdtemp()
    subprocess.run(
        [sys.executable, "-m", "pip", "install", pkg, "-t", tmp, "--quiet"],
        capture_output=True
    )
    # Calculate size
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(tmp):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total_size += os.path.getsize(fp)
    size_mb = total_size / (1024 * 1024)
    pkg_results[label] = round(size_mb, 2)
    print(f"  {label}: {size_mb:.2f} MB")
    shutil.rmtree(tmp)

results["tests"]["package_size_mb"] = pkg_results
print()

# ============================================
# TEST 2: Real Bedrock Model Latency
# ============================================
print("TEST 2: Bedrock Model Latency (10 calls per model)")
print("-" * 40)

import boto3

client = boto3.client("bedrock-runtime", region_name="us-east-1")
PROMPT = "Summarize in exactly 20 words: AWS Lambda is a serverless compute service that runs code in response to events and automatically manages compute resources."

models_to_test = {
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
}

# Try sonnet too
try:
    test_body = json.dumps({"anthropic_version": "bedrock-2023-05-31", "max_tokens": 50, "messages": [{"role": "user", "content": "hi"}]})
    client.invoke_model(modelId="anthropic.claude-3-sonnet-20240229-v1:0", body=test_body, contentType="application/json")
    models_to_test["claude-3-sonnet"] = "anthropic.claude-3-sonnet-20240229-v1:0"
except:
    print("  (Sonnet not available, testing Haiku only)")

# Try Titan
try:
    test_body = json.dumps({"inputText": "hi", "textGenerationConfig": {"maxTokenCount": 50}})
    client.invoke_model(modelId="amazon.titan-text-express-v1", body=test_body, contentType="application/json")
    models_to_test["titan-express"] = "amazon.titan-text-express-v1"
except:
    print("  (Titan not available)")

model_results = {}
for name, model_id in models_to_test.items():
    latencies = []
    tokens_in_list = []
    tokens_out_list = []

    for i in range(10):
        if "anthropic" in model_id:
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": PROMPT}]
            })
        elif "titan" in model_id:
            body = json.dumps({
                "inputText": PROMPT,
                "textGenerationConfig": {"maxTokenCount": 100, "temperature": 0.7}
            })
        else:
            continue

        start = time.time()
        response = client.invoke_model(modelId=model_id, body=body, contentType="application/json")
        latency_ms = (time.time() - start) * 1000
        latencies.append(latency_ms)

        resp_body = json.loads(response["body"].read())

        if "anthropic" in model_id:
            tokens_in_list.append(resp_body["usage"]["input_tokens"])
            tokens_out_list.append(resp_body["usage"]["output_tokens"])
        elif "titan" in model_id:
            tokens_in_list.append(resp_body.get("inputTextTokenCount", 0))
            tokens_out_list.append(resp_body["results"][0].get("tokenCount", 0))

        time.sleep(0.5)

    avg_latency = sum(latencies) / len(latencies)
    model_results[name] = {
        "model_id": model_id,
        "avg_latency_ms": round(avg_latency, 1),
        "min_latency_ms": round(min(latencies), 1),
        "max_latency_ms": round(max(latencies), 1),
        "p50_latency_ms": round(sorted(latencies)[5], 1),
        "avg_tokens_in": round(sum(tokens_in_list) / len(tokens_in_list), 1),
        "avg_tokens_out": round(sum(tokens_out_list) / len(tokens_out_list), 1),
        "samples": len(latencies),
    }
    print(f"  {name}: avg={avg_latency:.0f}ms, p50={sorted(latencies)[5]:.0f}ms, tokens={tokens_in_list[-1]}in/{tokens_out_list[-1]}out")

results["tests"]["model_latency"] = model_results
print()

# ============================================
# TEST 3: LambdaLLM Framework Overhead
# ============================================
print("TEST 3: Framework Overhead (LambdaLLM vs raw boto3)")
print("-" * 40)

from lambdallm import handler, Prompt, Model
from lambdallm.core.models import ModelResponse

# Test with real Bedrock via LambdaLLM
class RealLambdaContext:
    function_name = "benchmark"
    def get_remaining_time_in_millis(self):
        return 30000

summarize = Prompt(
    template="Summarize in 20 words: {text}",
    input_schema={"text": str},
)

@handler(model=Model.CLAUDE_3_HAIKU, max_retries=2)
def lambdallm_handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})
    result = context.invoke("Summarize in 20 words: {text}", text=body.get("text", ""))
    return {"statusCode": 200, "body": {"result": result, "cost_usd": context.total_cost}}

# Run through LambdaLLM
lambdallm_latencies = []
lambdallm_costs = []
test_event = {"body": json.dumps({"text": "AWS Lambda is a serverless compute service that runs code in response to events."})}

for i in range(10):
    start = time.time()
    result = lambdallm_handler(test_event, RealLambdaContext())
    latency = (time.time() - start) * 1000
    lambdallm_latencies.append(latency)
    body = json.loads(result["body"]) if isinstance(result["body"], str) else result["body"]
    lambdallm_costs.append(body.get("cost_usd", 0))
    time.sleep(0.5)

# Run raw boto3 for comparison
raw_latencies = []
for i in range(10):
    start = time.time()
    response = client.invoke_model(
        modelId="anthropic.claude-3-haiku-20240307-v1:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "Summarize in 20 words: AWS Lambda is a serverless compute service."}]
        }),
        contentType="application/json",
    )
    json.loads(response["body"].read())
    latency = (time.time() - start) * 1000
    raw_latencies.append(latency)
    time.sleep(0.5)

framework_overhead = sum(lambdallm_latencies) / len(lambdallm_latencies) - sum(raw_latencies) / len(raw_latencies)

results["tests"]["framework_overhead"] = {
    "lambdallm_avg_ms": round(sum(lambdallm_latencies) / len(lambdallm_latencies), 1),
    "raw_boto3_avg_ms": round(sum(raw_latencies) / len(raw_latencies), 1),
    "overhead_ms": round(framework_overhead, 1),
    "overhead_percent": round((framework_overhead / (sum(raw_latencies) / len(raw_latencies))) * 100, 1),
    "lambdallm_avg_cost_usd": round(sum(lambdallm_costs) / len(lambdallm_costs), 6),
    "samples": 10,
}

print(f"  LambdaLLM avg: {sum(lambdallm_latencies)/len(lambdallm_latencies):.0f}ms")
print(f"  Raw boto3 avg: {sum(raw_latencies)/len(raw_latencies):.0f}ms")
print(f"  Framework overhead: {framework_overhead:.0f}ms ({framework_overhead/(sum(raw_latencies)/len(raw_latencies))*100:.1f}%)")
print(f"  Avg cost per request: ${sum(lambdallm_costs)/len(lambdallm_costs):.6f}")
print()

# ============================================
# TEST 4: Session State (Multi-Turn)
# ============================================
print("TEST 4: Session State (Multi-Turn Memory)")
print("-" * 40)

from lambdallm.state.session import Session
from lambdallm.state.memory import InMemoryStateStore

store = InMemoryStateStore()
session = Session(store="memory", max_messages=20)
session._store_instance = store
session.session_id = "benchmark-session"

# Simulate multi-turn
turns = [
    ("user", "My name is Gaurav and I work on LambdaLLM."),
    ("assistant", "Nice to meet you, Gaurav! LambdaLLM sounds interesting."),
    ("user", "I live in the United States."),
    ("assistant", "Great! The US has a thriving tech ecosystem."),
    ("user", "What is my name and what do I work on?"),
]

for role, content in turns:
    session.add_message(role, content)

session.save()

# Reload (simulates new Lambda invocation)
session2 = Session(store="memory", max_messages=20)
session2._store_instance = store
session2.load("benchmark-session")

history = session2.format_history()

# Ask the model with context
context_response = client.invoke_model(
    modelId="anthropic.claude-3-haiku-20240307-v1:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 100,
        "messages": [{"role": "user", "content": f"Based on this conversation, answer the last question:\n\n{history}"}]
    }),
    contentType="application/json",
)
context_result = json.loads(context_response["body"].read())
answer = context_result["content"][0]["text"].lower()

context_retained = "gaurav" in answer and "lambdallm" in answer

results["tests"]["session_state"] = {
    "turns_stored": session2.message_count,
    "state_persisted": session2.message_count == 5,
    "context_retained_in_response": context_retained,
    "model_answer": context_result["content"][0]["text"][:200],
}

print(f"  Messages stored: {session2.message_count}")
print(f"  State persisted across reload: {session2.message_count == 5}")
print(f"  Context retained in model answer: {context_retained}")
print(f"  Answer: {context_result['content'][0]['text'][:100]}...")
print()

# ============================================
# TEST 5: Chain Execution
# ============================================
print("TEST 5: Chain Execution (3-step pipeline)")
print("-" * 40)

from lambdallm import Chain, Step

chain = Chain(
    name="benchmark-chain",
    steps=[
        Step("extract", prompt="Extract key entities (people, companies, numbers) from: {input}"),
        Step("classify", prompt="Classify each entity by type from this list: {extract.output}"),
        Step("summarize", prompt="Write a one-sentence summary based on: {classify.output}"),
    ],
)

chain_start = time.time()
chain_result = chain.run(
    context=RealLambdaContext(),
    input="Gaurav Kumar Sinha founded SubstrAI in 2026. The company released LambdaLLM, which has been downloaded 1000 times from PyPI."
)
chain_latency = (time.time() - chain_start) * 1000

# Need a real context for chain - use handler
@handler(model=Model.CLAUDE_3_HAIKU)
def chain_handler(event, context):
    body = json.loads(event.get("body", "{}")) if isinstance(event.get("body"), str) else event.get("body", {})
    result = chain.run(context=context, input=body.get("text", "test"))
    return {"statusCode": 200, "body": {"status": result.status, "steps": result.completed_steps, "output": result.final_output}}

chain_event = {"body": json.dumps({"text": "Gaurav Kumar Sinha founded SubstrAI in 2026. The company released LambdaLLM, downloaded 1000 times."})}
chain_response = chain_handler(chain_event, RealLambdaContext())
chain_body = json.loads(chain_response["body"]) if isinstance(chain_response["body"], str) else chain_response["body"]

results["tests"]["chain_execution"] = {
    "steps": 3,
    "status": chain_body.get("status", "unknown"),
    "completed_steps": chain_body.get("steps", 0),
    "has_output": chain_body.get("output") is not None,
    "output_preview": str(chain_body.get("output", ""))[:200],
}

print(f"  Status: {chain_body.get('status')}")
print(f"  Steps completed: {chain_body.get('steps')}/3")
print(f"  Output: {str(chain_body.get('output', ''))[:100]}...")
print()

# ============================================
# TEST 6: Cost Tracking Accuracy
# ============================================
print("TEST 6: Cost Tracking Accuracy")
print("-" * 40)

# Cost data inline

# Calculate expected cost for known token counts
test_tokens_in = 14  # From our earlier test
test_tokens_out = 9
haiku_costs = {"input": 0.00025, "output": 0.00125}
expected_cost = (test_tokens_in / 1000) * haiku_costs["input"] + (test_tokens_out / 1000) * haiku_costs["output"]

# Get actual tracked cost from framework
@handler(model=Model.CLAUDE_3_HAIKU)
def cost_handler(event, context):
    context.invoke("Say hello in 5 words")
    return {"statusCode": 200, "body": {"tracked_cost": context.total_cost}}

cost_response = cost_handler({"body": "{}"}, RealLambdaContext())
cost_body = json.loads(cost_response["body"]) if isinstance(cost_response["body"], str) else cost_response["body"]
tracked_cost = cost_body.get("tracked_cost", 0)

results["tests"]["cost_tracking"] = {
    "tracked_cost_usd": tracked_cost,
    "expected_range_min": 0.000001,
    "expected_range_max": 0.001,
    "within_expected_range": 0.000001 <= tracked_cost <= 0.001,
    "cost_model_prices": haiku_costs,
}

print(f"  Tracked cost: ${tracked_cost:.6f}")
print(f"  Within expected range: {0.000001 <= tracked_cost <= 0.001}")
print()

# ============================================
# TEST 7: Prompt Template Validation
# ============================================
print("TEST 7: Prompt Template Features")
print("-" * 40)

from lambdallm.core.prompt import Prompt
from lambdallm.core.exceptions import ConfigurationError

tests_passed = 0
tests_total = 5

# Test 1: Variable extraction
p = Prompt(template="Hello {name}, you are {age}")
if set(p._variables) == {"name", "age"}: tests_passed += 1

# Test 2: Format works
if p.format(name="Gaurav", age="30") == "Hello Gaurav, you are 30": tests_passed += 1

# Test 3: Missing var raises error
try:
    p.format(name="test")
except (ConfigurationError, KeyError):
    tests_passed += 1

# Test 4: Type validation
p2 = Prompt(template="{count} items", input_schema={"count": int})
try:
    p2.format(count="not int")
except ConfigurationError:
    tests_passed += 1

# Test 5: Serialization round-trip
p3 = Prompt(template="{x}", input_schema={"x": str}, name="test", version="2.0")
data = p3.to_dict()
p4 = Prompt.from_dict(data)
if p4.name == "test" and p4.version == "2.0": tests_passed += 1

results["tests"]["prompt_templates"] = {
    "tests_passed": tests_passed,
    "tests_total": tests_total,
    "pass_rate": tests_passed / tests_total,
}

print(f"  Passed: {tests_passed}/{tests_total}")
print()

# ============================================
# TEST 8: Tool System
# ============================================
print("TEST 8: Agent Tool System")
print("-" * 40)

from lambdallm.agents import Tool, ToolRegistry

@Tool(description="Calculate a math expression")
def calculate(expression: str) -> float:
    """Calculate math.
    Args:
        expression: Math expression.
    """
    allowed = set("0123456789+-*/.() ")
    if all(c in allowed for c in expression):
        return eval(expression)
    return 0.0

@Tool(description="Get current time")
def get_time() -> str:
    """Get current UTC time."""
    return datetime.utcnow().isoformat()

registry = ToolRegistry([calculate, get_time])

# Test tool invocation
calc_result = registry.invoke("calculate", expression="100 * 1.15")
time_result = registry.invoke("get_time")

# Test schema generation
schemas = registry.get_schemas()

results["tests"]["tool_system"] = {
    "tools_registered": registry.count,
    "calculate_result": calc_result,
    "calculate_correct": calc_result == 115.0,
    "time_result_valid": len(time_result) > 10,
    "schemas_generated": len(schemas),
    "schema_has_params": len(schemas[0]["input_schema"]["properties"]) > 0,
}

print(f"  Tools registered: {registry.count}")
print(f"  calculate('100 * 1.15') = {calc_result} (correct: {calc_result == 115.0})")
print(f"  get_time() = {time_result}")
print(f"  Schemas generated: {len(schemas)} with params")
print()

# ============================================
# TEST 9: Observability
# ============================================
print("TEST 9: Observability (Tracer + Metrics)")
print("-" * 40)

from lambdallm.observability.tracer import Tracer
from lambdallm.observability.metrics import MetricsEmitter

tracer = Tracer(enabled=True)

# Simulate spans
with tracer.span("model.invoke", {"model": "haiku"}) as span:
    span.set_attribute("tokens_in", 14)
    time.sleep(0.01)

with tracer.span("tool.calculate") as span:
    span.set_attribute("expression", "2+2")

summary = tracer.get_trace_summary()

emitter = MetricsEmitter(namespace="Benchmark", enabled=True)
emitter.record("test.latency", 150.0, "Milliseconds")
emitter.record_model_invocation("haiku", 100, 50, 200.0, 0.0003)

results["tests"]["observability"] = {
    "tracer_spans": summary["span_count"],
    "tracer_errors": summary["errors"],
    "tracer_total_duration_ms": round(summary["total_duration_ms"], 1),
    "metrics_buffered": emitter.pending_count,
    "all_working": summary["span_count"] == 2 and emitter.pending_count > 0,
}

print(f"  Tracer spans: {summary['span_count']}")
print(f"  Tracer errors: {summary['errors']}")
print(f"  Metrics buffered: {emitter.pending_count}")
print()

# ============================================
# TEST 10: A/B Testing
# ============================================
print("TEST 10: A/B Testing")
print("-" * 40)

from lambdallm.observability.ab_testing import Experiment, Variant

exp = Experiment(name="benchmark-test", variants=[
    Variant("control", weight=0.5, prompt_template="Summarize: {text}"),
    Variant("treatment", weight=0.5, prompt_template="Briefly summarize: {text}"),
])

selections = [exp.select_variant().name for _ in range(1000)]
control_pct = selections.count("control") / 1000

# Sticky session test
v1 = exp.select_variant(user_id="user-123")
v2 = exp.select_variant(user_id="user-123")
sticky_works = v1.name == v2.name

results["tests"]["ab_testing"] = {
    "control_percent": round(control_pct * 100, 1),
    "treatment_percent": round((1 - control_pct) * 100, 1),
    "distribution_balanced": 40 <= control_pct * 100 <= 60,
    "sticky_sessions_work": sticky_works,
}

print(f"  Distribution: control={control_pct*100:.1f}%, treatment={(1-control_pct)*100:.1f}%")
print(f"  Balanced (40-60%): {40 <= control_pct*100 <= 60}")
print(f"  Sticky sessions: {sticky_works}")
print()

# ============================================
# SAVE ALL RESULTS
# ============================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(RESULTS_DIR, f"full_benchmark_{timestamp}.json")

with open(output_file, "w") as f:
    json.dump(results, f, indent=2)

# Print summary
print("=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
print()

all_tests = results["tests"]
print(f"Results saved to: {output_file}")
print()
print("SUMMARY:")
print(f"  Package size (core): {all_tests['package_size_mb'].get('substrai-lambdallm (core)', 'N/A')} MB")
print(f"  Framework overhead: {all_tests['framework_overhead']['overhead_ms']}ms ({all_tests['framework_overhead']['overhead_percent']}%)")
print(f"  Session state works: {all_tests['session_state']['state_persisted']}")
print(f"  Chain completed: {all_tests['chain_execution']['status']}")
print(f"  Cost tracking: ${all_tests['cost_tracking']['tracked_cost_usd']:.6f}")
print(f"  Prompt tests: {all_tests['prompt_templates']['tests_passed']}/{all_tests['prompt_templates']['tests_total']}")
print(f"  Tool system: {all_tests['tool_system']['tools_registered']} tools, calc correct={all_tests['tool_system']['calculate_correct']}")
print(f"  Observability: {all_tests['observability']['all_working']}")
print(f"  A/B testing: balanced={all_tests['ab_testing']['distribution_balanced']}, sticky={all_tests['ab_testing']['sticky_sessions_work']}")
