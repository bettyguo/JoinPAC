#!/bin/bash
set -euo pipefail
echo "Running ablation study..."
python run_experiment.py --config configs/experiment/ablation.yaml --experiment ablation --output-dir results/ablation
echo "Ablation study complete."
