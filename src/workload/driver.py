"""Workload driver for running experiments."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


class WorkloadDriver:
    """Drives workload execution for experiments."""

    def __init__(
        self,
        num_runs: int = 5,
        warmup_runs: int = 3,
    ) -> None:
        self.num_runs = num_runs
        self.warmup_runs = warmup_runs

    def run_workload(
        self,
        queries: list[dict[str, Any]],
        estimator_fn: Callable[[dict[str, Any]], float],
        true_cardinalities: dict[int, int] | None = None,
    ) -> dict[str, Any]:
        """Run a workload and collect estimation results."""
        all_run_results: list[list[dict[str, Any]]] = []

        total_runs = self.warmup_runs + self.num_runs
        for run_idx in range(total_runs):
            is_warmup = run_idx < self.warmup_runs
            run_results = []

            for query in queries:
                start = time.perf_counter()
                estimate = estimator_fn(query)
                elapsed = time.perf_counter() - start

                result: dict[str, Any] = {
                    "query_id": query.get("id", -1),
                    "estimate": estimate,
                    "inference_time_ms": elapsed * 1000,
                }

                if true_cardinalities and query.get("id") in true_cardinalities:
                    true_card = true_cardinalities[query["id"]]
                    result["true_cardinality"] = true_card
                    if true_card > 0 and estimate > 0:
                        result["q_error"] = max(estimate / true_card, true_card / estimate)

                run_results.append(result)

            if not is_warmup:
                all_run_results.append(run_results)

        return self._aggregate_runs(all_run_results)

    def _aggregate_runs(
        self, all_runs: list[list[dict[str, Any]]]
    ) -> dict[str, Any]:
        """Aggregate results across multiple runs."""
        if not all_runs:
            return {}

        q_errors_per_query: dict[int, list[float]] = {}
        for run in all_runs:
            for result in run:
                qid = result["query_id"]
                if "q_error" in result:
                    q_errors_per_query.setdefault(qid, []).append(result["q_error"])

        all_q_errors = []
        for qid, errors in q_errors_per_query.items():
            all_q_errors.append(float(np.median(errors)))

        if all_q_errors:
            arr = np.array(all_q_errors)
            return {
                "median_q_error": float(np.median(arr)),
                "mean_q_error": float(np.mean(arr)),
                "p99_q_error": float(np.percentile(arr, 99)),
                "p95_q_error": float(np.percentile(arr, 95)),
                "num_queries": len(all_q_errors),
                "num_runs": len(all_runs),
            }
        return {"num_queries": 0, "num_runs": len(all_runs)}
