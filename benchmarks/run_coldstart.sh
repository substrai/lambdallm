#!/bin/bash
# Benchmark: Lambda Cold Start Measurement
#
# Requires: Deployed Lambda function
# Usage: bash run_coldstart.sh <function-name>
#
# This script forces cold starts by updating the function config,
# then measures Init Duration from CloudWatch logs.

FUNCTION_NAME=${1:-"lambdallm-benchmark-dev"}
REGION="us-east-1"
NUM_RUNS=20
RESULTS_FILE="results/coldstart_$(date +%Y%m%d_%H%M%S).json"

echo "Cold Start Benchmark"
echo "===================="
echo "Function: $FUNCTION_NAME"
echo "Region: $REGION"
echo "Runs: $NUM_RUNS"
echo ""

echo "[]" > $RESULTS_FILE

for i in $(seq 1 $NUM_RUNS); do
  echo -n "Run $i/$NUM_RUNS: "

  # Force cold start by changing env var
  aws lambda update-function-configuration \
    --function-name $FUNCTION_NAME \
    --environment "Variables={BENCHMARK_RUN=$i}" \
    --region $REGION > /dev/null 2>&1

  # Wait for update
  aws lambda wait function-updated \
    --function-name $FUNCTION_NAME \
    --region $REGION 2>/dev/null

  sleep 2

  # Invoke and capture logs
  aws lambda invoke \
    --function-name $FUNCTION_NAME \
    --payload '{"body": "{\"text\": \"benchmark test\", \"max_words\": 10}"}' \
    --region $REGION \
    --log-type Tail \
    --query 'LogResult' \
    --output text /tmp/bench_response.json 2>/dev/null | base64 -d > /tmp/bench_log.txt

  # Extract Init Duration
  INIT_DURATION=$(grep "Init Duration" /tmp/bench_log.txt | grep -oP "Init Duration: \K[0-9.]+" || echo "0")
  DURATION=$(grep "Duration:" /tmp/bench_log.txt | head -1 | grep -oP "Duration: \K[0-9.]+" || echo "0")
  MEMORY=$(grep "Max Memory" /tmp/bench_log.txt | grep -oP "Max Memory Used: \K[0-9]+" || echo "0")

  echo "Init: ${INIT_DURATION}ms, Duration: ${DURATION}ms, Memory: ${MEMORY}MB"

  # Append to results
  python3 -c "
import json
with open('$RESULTS_FILE', 'r') as f: data = json.load(f)
data.append({'run': $i, 'init_duration_ms': float('${INIT_DURATION}'), 'duration_ms': float('${DURATION}'), 'memory_mb': int('${MEMORY}' or '0')})
with open('$RESULTS_FILE', 'w') as f: json.dump(data, f, indent=2)
"
done

echo ""
echo "Results saved to: $RESULTS_FILE"
echo ""

# Summary
python3 -c "
import json
with open('$RESULTS_FILE') as f: data = json.load(f)
inits = [d['init_duration_ms'] for d in data if d['init_duration_ms'] > 0]
durations = [d['duration_ms'] for d in data if d['duration_ms'] > 0]
if inits:
    print(f'Cold Start (Init Duration):')
    print(f'  Avg: {sum(inits)/len(inits):.1f}ms')
    print(f'  Min: {min(inits):.1f}ms')
    print(f'  Max: {max(inits):.1f}ms')
    print(f'  Samples: {len(inits)}')
if durations:
    print(f'Total Duration:')
    print(f'  Avg: {sum(durations)/len(durations):.1f}ms')
"
