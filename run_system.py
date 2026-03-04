#!/usr/bin/env python
"""Main system entry point for PAC-CE experiments."""

import sys
from pathlib import Path

import click

from pac_ce.core.config import PACCEConfig
from pac_ce.core.engine import PACCEEngine


@click.command()
@click.option("--config", default="configs/default.yaml", help="Path to config file")
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--seed", default=None, type=int, help="Override random seed")
@click.option("--log-level", default=None, type=click.Choice(["DEBUG", "INFO", "WARNING"]))
def main(config: str, output_dir: str | None, seed: int | None, log_level: str | None) -> None:
    """PAC-CE: Run the cardinality estimation framework."""
    config_path = Path(config)
    if not config_path.exists():
        click.echo(f"Config file not found: {config}", err=True)
        sys.exit(1)

    cfg = PACCEConfig.from_yaml(config_path)
    if output_dir:
        cfg.output_dir = output_dir
    if seed is not None:
        cfg.experiment.seed = seed
    if log_level:
        cfg.log_level = log_level

    engine = PACCEEngine(cfg)
    results = engine.run()

    click.echo(f"\nResults saved to {cfg.output_dir}/")


if __name__ == "__main__":
    main()
