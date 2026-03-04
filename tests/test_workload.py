"""Tests for workload generation."""

import pytest
from pac_ce.workload.generator import WorkloadGenerator


class TestWorkloadGenerator:
    def test_generate_queries(self):
        gen = WorkloadGenerator(seed=42)
        tables = ["A", "B", "C", "D", "E"]
        queries = gen.generate_random_queries(10, tables, max_joins=3)
        assert len(queries) == 10
        for q in queries:
            assert len(q["tables"]) >= 2
            assert q["num_joins"] == len(q["tables"]) - 1

    def test_reproducibility(self):
        gen1 = WorkloadGenerator(seed=42)
        gen2 = WorkloadGenerator(seed=42)
        tables = ["A", "B", "C"]
        q1 = gen1.generate_random_queries(5, tables)
        q2 = gen2.generate_random_queries(5, tables)
        for a, b in zip(q1, q2):
            assert a["tables"] == b["tables"]
