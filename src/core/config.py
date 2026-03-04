"""Configuration management for PAC-CE experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark dataset."""

    name: str = "job"
    data_dir: str = "benchmarks/data"
    num_tables: int = 21
    num_queries: int = 113
    scale_factor: int = 1


@dataclass
class ModelConfig:
    """Configuration for a CE model."""

    architecture: str = "mscn"
    num_layers: int = 5
    hidden_dim: int = 256
    learning_rate: float = 1e-3
    batch_size: int = 128
    epochs: int = 100
    dropout: float = 0.1


@dataclass
class ExperimentConfig:
    """Configuration for running experiments."""

    num_runs: int = 5
    warmup_runs: int = 3
    seed: int = 42
    confidence_threshold: float = 0.3
    drift_threshold: float = 0.45
    calibration_constant: float = 2.3
    reporting: str = "median with 95% CI over 5 runs"


@dataclass
class HybridConfig:
    """Configuration for the hybrid estimator."""

    confidence_threshold: float = 0.3
    sigma_multiplier: float = 1.0
    it_bound_type: str = "safebound"
    fallback_strategy: str = "geometric_mean"


@dataclass
class PACCEConfig:
    """Top-level configuration for PAC-CE."""

    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    hybrid: HybridConfig = field(default_factory=HybridConfig)
    output_dir: str = "results"
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, path: str | Path) -> PACCEConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}

        config = cls()
        if "benchmark" in raw:
            config.benchmark = BenchmarkConfig(**raw["benchmark"])
        if "model" in raw:
            config.model = ModelConfig(**raw["model"])
        if "experiment" in raw:
            config.experiment = ExperimentConfig(**raw["experiment"])
        if "hybrid" in raw:
            config.hybrid = HybridConfig(**raw["hybrid"])
        if "output_dir" in raw:
            config.output_dir = raw["output_dir"]
        if "log_level" in raw:
            config.log_level = raw["log_level"]
        return config

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to a YAML file."""
        from dataclasses import asdict

        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)
