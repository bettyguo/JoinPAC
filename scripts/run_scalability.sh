#!/bin/bash
set -euo pipefail
echo "Running scalability experiments..."
python run_experiment.py --config configs/experiment/scalability.yaml --experiment scalability --output-dir results/scalability
echo "Scalability experiments complete."
