"""Tests for statistical analysis."""

import numpy as np
import pytest
from pac_ce.evaluation.statistics import (
    compute_confidence_interval,
    paired_t_test,
    bootstrap_ci,
)


class TestConfidenceInterval:
    def test_basic_ci(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lower, upper = compute_confidence_interval(data)
        assert mean == pytest.approx(3.0)
        assert lower < mean
        assert upper > mean

    def test_single_value(self):
        data = np.array([5.0])
        mean, lower, upper = compute_confidence_interval(data)
        assert mean == 5.0


class TestPairedTTest:
    def test_identical_distributions(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_t_test(a, a)
        assert result["p_value"] > 0.05

    def test_different_distributions(self):
        a = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        b = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = paired_t_test(a, b)
        assert result["p_value"] < 0.05


class TestBootstrapCI:
    def test_basic_bootstrap(self):
        data = np.random.randn(100) + 5.0
        point, lower, upper = bootstrap_ci(data)
        assert lower < point < upper
        assert abs(point - 5.0) < 1.0
