"""Data loader for benchmark datasets."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


BENCHMARK_INFO = {
    "job": {
        "name": "Join Order Benchmark",
        "num_tables": 21,
        "num_queries": 113,
        "source": "https://github.com/gregrahn/join-order-benchmark",
        "attr_disjoint_pct": 89.0,
    },
    "stats-ceb": {
        "name": "STATS Cardinality Estimation Benchmark",
        "num_tables": 8,
        "num_queries": 146,
        "source": "https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark",
        "attr_disjoint_pct": 94.0,
    },
    "tpc-ds": {
        "name": "TPC-DS Decision Support Benchmark",
        "num_templates": 99,
        "source": "https://www.tpc.org/tpcds/",
        "scale_factors": [1, 10, 100],
    },
}


def get_benchmark_info(name: str) -> dict[str, Any]:
    """Get metadata for a benchmark."""
    if name not in BENCHMARK_INFO:
        raise ValueError(f"Unknown benchmark: {name}. Available: {list(BENCHMARK_INFO)}")
    return BENCHMARK_INFO[name]


def load_benchmark_queries(benchmark: str, data_dir: str | Path) -> list[dict[str, Any]]:
    """Load queries for a benchmark."""
    data_dir = Path(data_dir)
    query_file = data_dir / benchmark / "queries.json"

    if query_file.exists():
        import json
        with open(query_file) as f:
            return json.load(f)

    logger.warning("Query file not found: %s. Run setup_benchmarks.sh", query_file)
    return []
