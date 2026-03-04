#!/bin/bash
# Run baseline systems
set -euo pipefail
echo "Running baseline systems..."
python run_experiment.py --config configs/experiment/main_comparison.yaml --experiment main --output-dir results/baselines
echo "Baselines complete."
