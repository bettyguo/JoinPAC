#!/bin/bash
# Run all experiments end-to-end
set -euo pipefail

echo "Running all PAC-CE experiments"
echo "=============================="

python run_experiment.py --config configs/experiment/main_comparison.yaml --experiment main --output-dir results/main
python run_experiment.py --config configs/experiment/scalability.yaml --experiment scalability --output-dir results/scalability
python run_experiment.py --config configs/experiment/ablation.yaml --experiment ablation --output-dir results/ablation
python run_experiment.py --config configs/experiment/main_comparison.yaml --experiment drift --output-dir results/drift
python run_experiment.py --config configs/experiment/main_comparison.yaml --experiment hybrid --output-dir results/hybrid

echo ""
echo "All experiments complete. Results in results/"
