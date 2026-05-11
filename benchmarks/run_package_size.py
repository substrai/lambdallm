#!/usr/bin/env python3
"""Benchmark: Package Size Comparison

Measures deployment package size for each framework.
No AWS credentials needed.

Usage: python3 run_package_size.py
"""

import os
import sys
import json
import shutil
import tempfile
import subprocess
from datetime import datetime

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

packages = {
    "substrai-lambdallm (core)": ["substrai-lambdallm"],
    "substrai-lambdallm + boto3": ["substrai-lambdallm", "boto3"],
    "langchain + langchain-aws": ["langchain", "langchain-aws"],
    "langchain full stack": ["langchain", "langchain-aws", "langchain-community"],
    "boto3 only": ["boto3"],
}

print("Package Size Benchmark")
print("=" * 50)

results = {}
for label, pkgs in packages.items():
    tmp = tempfile.mkdtemp()
    for pkg in pkgs:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg, "-t", tmp, "--quiet"],
            capture_output=True
        )
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, dn, fns in os.walk(tmp) for f in fns
    )
    size_mb = round(total / (1024 * 1024), 2)
    results[label] = size_mb
    print(f"  {label}: {size_mb} MB")
    shutil.rmtree(tmp)

output = os.path.join(RESULTS_DIR, f"package_size_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
with open(output, "w") as f:
    json.dump({"timestamp": datetime.now().isoformat(), "package_sizes_mb": results}, f, indent=2)

print(f"\nResults: {output}")
print(f"\nLambdaLLM is {results.get('langchain full stack', 1) / max(results.get('substrai-lambdallm (core)', 1), 0.01):.0f}x smaller than LangChain")
