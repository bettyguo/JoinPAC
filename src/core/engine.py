"""Main engine for PAC-CE experiments."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import click
import numpy as np

from pac_ce.core.config import PACCEConfig
from pac_ce.core.pipeline import ExperimentPipeline
from pac_ce.utils.reproducibility import set_seed, get_system_info, log_experiment_config


logger = logging.getLogger(__name__)


class PACCEEngine:
    """Main engine orchestrating PAC-CE experiments."""

    def __init__(self, config: PACCEConfig) -> None:
        self.config = config
        self.results: dict[str, Any] = {}
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure structured logging."""
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def run(self) -> dict[str, Any]:
        """Execute the full experiment pipeline."""
        logger.info("Starting PAC-CE experiment pipeline")
        start_time = time.time()

        set_seed(self.config.experiment.seed)
        sys_info = get_system_info()
        log_experiment_config(self.config)

        logger.info("System info: %s", sys_info)

        pipeline = ExperimentPipeline(self.config)

        logger.info("Phase 1: Loading benchmark data")
        pipeline.load_data()

        logger.info("Phase 2: Training CE models")
        pipeline.train_models()

        logger.info("Phase 3: Computing theoretical bounds")
        pipeline.compute_bounds()

        logger.info("Phase 4: Running drift detection")
        pipeline.run_drift_detection()

        logger.info("Phase 5: Running hybrid estimation")
        pipeline.run_hybrid_estimation()

        logger.info("Phase 6: Collecting results")
        self.results = pipeline.collect_results()

        elapsed = time.time() - start_time
        logger.info("Pipeline completed in %.1f seconds", elapsed)
        self.results["elapsed_time"] = elapsed
        self.results["system_info"] = sys_info

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline.save_results(output_dir)

        return self.results


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to config file")
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--seed", default=None, type=int, help="Override random seed")
@click.option("--log-level", default=None, help="Override log level")
def main(
    config: str,
    output_dir: str | None,
    seed: int | None,
    log_level: str | None,
) -> None:
    """PAC-CE: PAC-Theoretic Framework for Cardinality Estimation."""
    cfg = PACCEConfig.from_yaml(config)
    if output_dir:
        cfg.output_dir = output_dir
    if seed is not None:
        cfg.experiment.seed = seed
    if log_level:
        cfg.log_level = log_level

    engine = PACCEEngine(cfg)
    results = engine.run()

    click.echo(f"\nExperiment complete. Results saved to {cfg.output_dir}/")
    if "cross_benchmark_ratio" in results:
        click.echo(f"  Cross-benchmark prediction ratio: {results['cross_benchmark_ratio']:.2f}x")


if __name__ == "__main__":
    main()
