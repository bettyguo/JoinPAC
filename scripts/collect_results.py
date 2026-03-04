#!/usr/bin/env python
"""Aggregate raw results from all experiments."""

import json
import sys
from pathlib import Path


def collect_results(results_dir: str = "results") -> dict:
    """Collect and aggregate all experiment results."""
    base = Path(results_dir)
    aggregated = {}

    for exp_dir in sorted(base.iterdir()):
        if not exp_dir.is_dir():
            continue
        result_file = exp_dir / "experiment_results.json"
        if result_file.exists():
            with open(result_file) as f:
                aggregated[exp_dir.name] = json.load(f)

    output = base / "aggregated_results.json"
    with open(output, "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Aggregated {len(aggregated)} experiments to {output}")
    return aggregated


if __name__ == "__main__":
    collect_results(sys.argv[1] if len(sys.argv) > 1 else "results")
