"""Tests for core engine and configuration."""

import pytest
from pac_ce.core.config import PACCEConfig


class TestConfig:
    def test_default_config(self):
        config = PACCEConfig()
        assert config.benchmark.name == "job"
        assert config.experiment.num_runs == 5
        assert config.experiment.seed == 42

    def test_config_from_yaml(self, tmp_path):
        yaml_content = """
benchmark:
  name: stats-ceb
  num_queries: 146
experiment:
  seed: 123
"""
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text(yaml_content)
        config = PACCEConfig.from_yaml(config_file)
        assert config.benchmark.name == "stats-ceb"
        assert config.experiment.seed == 123

    def test_config_to_yaml(self, tmp_path):
        config = PACCEConfig()
        output = tmp_path / "output.yaml"
        config.to_yaml(output)
        assert output.exists()
        loaded = PACCEConfig.from_yaml(output)
        assert loaded.experiment.seed == config.experiment.seed
