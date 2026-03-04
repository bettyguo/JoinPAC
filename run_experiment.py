#!/usr/bin/env python
"""Experiment runner with multi-run execution and result aggregation."""

import json
import logging
import sys
import time
from pathlib import Path

import click
import numpy as np

from pac_ce.core.config import PACCEConfig
from pac_ce.core.pipeline import ExperimentPipeline
from pac_ce.utils.reproducibility import set_seed, warmup_system

logger = logging.getLogger(__name__)


@click.command()
@click.option("--config", required=True, help="Experiment config file")
@click.option("--experiment", required=True,
              type=click.Choice(["main", "scalability", "ablation", "drift", "hybrid"]))
@click.option("--output-dir", default="results", help="Output directory")
@click.option("--num-runs", default=5, help="Number of experimental runs")
@click.option("--warmup", default=3, help="Number of warmup runs")
@click.option("--seed", default=42, type=int, help="Random seed")
def main(
    config: str,
    experiment: str,
    output_dir: str,
    num_runs: int,
    warmup: int,
    seed: int,
) -> None:
    """Run a specific experiment with multi-run aggregation."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    cfg = PACCEConfig.from_yaml(config)
    cfg.experiment.num_runs = num_runs
    cfg.experiment.warmup_runs = warmup
    cfg.experiment.seed = seed
    cfg.output_dir = output_dir

    logger.info("Running experiment: %s", experiment)
    logger.info("Runs: %d (+ %d warmup), Seed: %d", num_runs, warmup, seed)

    set_seed(seed)
    warmup_system(iterations=warmup)

    pipeline = ExperimentPipeline(cfg)
    pipeline.load_data()
    pipeline.train_models()

    start = time.time()

    if experiment == "main":
        pipeline.compute_bounds()
        pipeline.run_hybrid_estimation()
    elif experiment == "drift":
        pipeline.run_drift_detection()
    elif experiment == "hybrid":
        pipeline.run_hybrid_estimation()
    elif experiment == "scalability":
        pipeline.compute_bounds()
    elif experiment == "ablation":
        pipeline.compute_bounds()
        pipeline.run_hybrid_estimation()

    elapsed = time.time() - start
    logger.info("Experiment %s completed in %.1fs", experiment, elapsed)

    out = Path(output_dir) / experiment
    out.mkdir(parents=True, exist_ok=True)
    pipeline.save_results(out)

    click.echo(f"Results saved to {out}/")


if __name__ == "__main__":
    main()
