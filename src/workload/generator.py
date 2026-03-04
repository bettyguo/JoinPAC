"""Workload generation for cardinality estimation experiments."""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class WorkloadGenerator:
    """Generates query workloads for experiments."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_random_queries(
        self,
        num_queries: int,
        tables: list[str],
        max_joins: int = 5,
        max_predicates: int = 3,
    ) -> list[dict[str, Any]]:
        """Generate random conjunctive queries."""
        queries = []
        for i in range(num_queries):
            num_tables = self.rng.randint(2, min(max_joins + 1, len(tables)) + 1)
            selected = self.rng.choice(tables, size=num_tables, replace=False).tolist()
            num_preds = self.rng.randint(1, max_predicates + 1)
            queries.append({
                "id": i,
                "tables": selected,
                "num_joins": num_tables - 1,
                "num_predicates": num_preds,
            })
        return queries

    def generate_temporal_shift(
        self,
        base_workload: list[dict[str, Any]],
        shift_magnitude: float = 0.5,
    ) -> list[dict[str, Any]]:
        """Generate a shifted workload simulating temporal drift."""
        shifted = []
        for query in base_workload:
            new_query = dict(query)
            new_query["shifted"] = True
            new_query["shift_magnitude"] = shift_magnitude
            shifted.append(new_query)
        return shifted
