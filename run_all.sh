#!/bin/bash

# LIA Experiment Pipeline
# Main script to run the entire experimental pipeline

set -e  # Exit on error

# Use the virtual environment python
PYTHON_CMD=".venv/bin/python"

# Check if virtual environment exists
if [ ! -f "$PYTHON_CMD" ]; then
    echo "Virtual environment not found. Please run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
fi

# Create necessary directories
mkdir -p results/data
mkdir -p results/figures
mkdir -p results/tables
mkdir -p results/topologies

echo "=== LIA Experiment Pipeline ==="
echo "Starting experiments at $(date)"

# Step 1: Generate network topologies
echo "Generating network topologies..."
$PYTHON_CMD run_pipeline.py

echo "All experiments completed at $(date)"
echo "Results saved to the 'results/' directory"
echo "=== Pipeline completed successfully ==="

