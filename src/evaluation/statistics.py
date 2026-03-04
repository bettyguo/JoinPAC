"""Statistical analysis utilities."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def compute_confidence_interval(
    data: NDArray[np.float64],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute mean and confidence interval."""
    n = len(data)
    mean = float(np.mean(data))
    if n < 2:
        return mean, mean, mean

    se = float(stats.sem(data))
    h = se * float(stats.t.ppf((1 + confidence) / 2, n - 1))
    return mean, mean - h, mean + h


def paired_t_test(
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> dict[str, float]:
    """Perform paired t-test between two sets of measurements."""
    stat, p_value = stats.ttest_rel(a, b)
    diff = a - b
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0.0
    return {
        "t_statistic": float(stat),
        "p_value": float(p_value),
        "mean_difference": mean_diff,
        "cohens_d": cohens_d,
    }


def bootstrap_ci(
    data: NDArray[np.float64],
    statistic_fn: callable = np.mean,
    num_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(data)
    bootstrap_stats = np.array([
        statistic_fn(rng.choice(data, size=n, replace=True))
        for _ in range(num_resamples)
    ])
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(bootstrap_stats, alpha * 100))
    upper = float(np.percentile(bootstrap_stats, (1 - alpha) * 100))
    point = float(statistic_fn(data))
    return point, lower, upper


def speedup_analysis(
    baseline_times: NDArray[np.float64],
    improved_times: NDArray[np.float64],
) -> dict[str, float]:
    """Compute speedup statistics."""
    speedups = baseline_times / np.maximum(improved_times, 1e-10)
    mean_speedup, lower, upper = compute_confidence_interval(speedups)
    return {
        "mean_speedup": mean_speedup,
        "median_speedup": float(np.median(speedups)),
        "ci_lower": lower,
        "ci_upper": upper,
    }
