"""Tests for query processing."""

import pytest
import numpy as np
from pac_ce.query.optimizer import HybridOptimizer


class TestHybridOptimizer:
    def test_high_confidence_uses_learned(self):
        opt = HybridOptimizer(confidence_threshold=0.3)
        encodings = np.random.randn(100, 10)
        opt.set_training_encodings(encodings)
        query_enc = encodings[0] + np.random.randn(10) * 0.01
        result = opt.hybrid_estimate(None, 100.0, 500.0, query_enc)
        assert result == 100.0

    def test_low_confidence_geometric_mean(self):
        opt = HybridOptimizer(confidence_threshold=0.99)
        encodings = np.random.randn(10, 10)
        opt.set_training_encodings(encodings)
        far_enc = np.ones(10) * 100
        learned, it_bound = 100.0, 400.0
        result = opt.hybrid_estimate(None, learned, it_bound, far_enc)
        expected = np.sqrt(learned * it_bound)
        assert abs(result - expected) < 1.0

    def test_learned_exceeds_bound_uses_it(self):
        opt = HybridOptimizer(confidence_threshold=0.99)
        encodings = np.random.randn(10, 10)
        opt.set_training_encodings(encodings)
        far_enc = np.ones(10) * 100
        result = opt.hybrid_estimate(None, 1000.0, 500.0, far_enc)
        assert result == 500.0
