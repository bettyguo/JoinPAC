"""Tests for evaluation metrics."""

import numpy as np
import pytest
from pac_ce.evaluation.metrics import (
    compute_q_error,
    compute_log_q_error,
    compute_latency_percentiles,
    compute_rademacher_bound,
    compute_sample_complexity,
    compute_spectral_norm_product,
)


class TestQError:
    def test_perfect_estimation(self):
        est = np.array([100.0, 200.0, 300.0])
        true = np.array([100.0, 200.0, 300.0])
        q_errors = compute_q_error(est, true)
        np.testing.assert_array_almost_equal(q_errors, [1.0, 1.0, 1.0])

    def test_overestimation(self):
        est = np.array([200.0])
        true = np.array([100.0])
        q_errors = compute_q_error(est, true)
        assert q_errors[0] == 2.0

    def test_underestimation(self):
        est = np.array([50.0])
        true = np.array([100.0])
        q_errors = compute_q_error(est, true)
        assert q_errors[0] == 2.0

    def test_log_q_error_symmetry(self):
        est = np.array([200.0, 50.0])
        true = np.array([100.0, 100.0])
        log_errors = compute_log_q_error(est, true)
        np.testing.assert_array_almost_equal(log_errors, [np.log(2), np.log(2)])


class TestLatencyPercentiles:
    def test_basic_percentiles(self):
        latencies = np.arange(1, 101, dtype=float)
        result = compute_latency_percentiles(latencies)
        assert "p50" in result
        assert "p99" in result
        assert result["mean"] == pytest.approx(50.5, abs=0.1)


class TestRademacherBound:
    def test_bound_positive(self):
        bound = compute_rademacher_bound(
            gamma=100.0, encoding_norm=1.0, num_layers=5, num_samples=1000
        )
        assert bound > 0

    def test_bound_decreases_with_samples(self):
        b1 = compute_rademacher_bound(100.0, 1.0, 5, 100)
        b2 = compute_rademacher_bound(100.0, 1.0, 5, 10000)
        assert b2 < b1


class TestSampleComplexity:
    def test_increases_with_joins(self):
        m1 = compute_sample_complexity(3, 2, 0.1)
        m2 = compute_sample_complexity(10, 2, 0.1)
        assert m2 > m1

    def test_increases_with_d_pred(self):
        m1 = compute_sample_complexity(5, 2, 0.1)
        m2 = compute_sample_complexity(5, 8, 0.1)
        assert m2 > m1


class TestSpectralNorm:
    def test_identity_matrix(self):
        matrices = [np.eye(10)]
        gamma = compute_spectral_norm_product(matrices)
        assert abs(gamma - 1.0) < 0.1
