#!/bin/bash
set -euo pipefail
echo "Running main comparison experiment..."
python run_experiment.py --config configs/experiment/main_comparison.yaml --experiment main --output-dir results/main
echo "Main comparison complete."
