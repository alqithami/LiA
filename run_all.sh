#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper that runs a quick smoke test and then two heavier runs.

python run_pipeline.py --config config/quick.json

# Uncomment as needed:
# python run_pipeline.py --config config/full.json
# python run_pipeline.py --config config/robustness.json
