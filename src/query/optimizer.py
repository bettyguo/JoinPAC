"""Query optimization with PAC-theoretic cardinality estimation."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from pac_ce.storage.structures import QueryStructure

logger = logging.getLogger(__name__)


class HybridOptimizer:
    """Query optimizer using hybrid cardinality estimation."""

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        sigma_multiplier: float = 1.0,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.sigma_multiplier = sigma_multiplier
        self.training_encodings: np.ndarray | None = None

    def set_training_encodings(self, encodings: np.ndarray) -> None:
        """Set the training query encodings for confidence computation."""
        self.training_encodings = encodings
        self._median_distance = float(np.median(
            np.linalg.norm(encodings[:, None] - encodings[None, :], axis=-1)
        ))

    def compute_confidence(self, query_encoding: np.ndarray) -> float:
        """Compute confidence score for a query."""
        if self.training_encodings is None:
            return 0.0

        distances = np.linalg.norm(
            self.training_encodings - query_encoding, axis=1
        )
        d_nn = float(np.min(distances))
        sigma = self._median_distance * self.sigma_multiplier
        confidence = float(np.exp(-(d_nn**2) / (2 * sigma**2)))
        return confidence

    def hybrid_estimate(
        self,
        query: QueryStructure,
        learned_estimate: float,
        it_bound: float,
        query_encoding: np.ndarray,
    ) -> float:
        """Produce a hybrid cardinality estimate."""
        conf = self.compute_confidence(query_encoding)

        if conf >= self.confidence_threshold:
            return learned_estimate
        elif learned_estimate > it_bound:
            return it_bound
        else:
            return float(np.sqrt(learned_estimate * it_bound))
