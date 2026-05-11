#!/usr/bin/env python3
"""Benchmark: Full Framework Comparison

Runs all comparison tests: package size, latency, features.
Requires AWS credentials with Bedrock access.

Usage: python3 run_comparison.py
"""

import subprocess
import sys
import os

print("Running full comparison benchmark suite...")
print("This will take approximately 2-3 minutes.\n")

# Run package size (no AWS needed)
print("Step 1/3: Package size...")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "run_package_size.py")])

print("\n")

# Run latency (needs AWS)
print("Step 2/3: Latency comparison...")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "run_latency.py")])

print("\n")

# Run full benchmark
print("Step 3/3: Full feature tests...")
subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "run_benchmarks.py")])

print("\n" + "=" * 50)
print("ALL BENCHMARKS COMPLETE")
print("Results in: benchmarks/results/")
print("=" * 50)
