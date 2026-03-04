"""Performance metrics for cardinality estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_q_error(
    estimates: NDArray[np.float64],
    true_cards: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute q-error between estimated and true cardinalities."""
    estimates = np.maximum(estimates, 1.0)
    true_cards = np.maximum(true_cards, 1.0)
    return np.maximum(estimates / true_cards, true_cards / estimates)


def compute_log_q_error(
    estimates: NDArray[np.float64],
    true_cards: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Compute log q-error (symmetric, additive form)."""
    q_errors = compute_q_error(estimates, true_cards)
    return np.log(q_errors)


def compute_throughput(
    num_queries: int,
    total_time_seconds: float,
) -> float:
    """Compute query throughput in queries per second."""
    if total_time_seconds <= 0:
        return 0.0
    return num_queries / total_time_seconds


def compute_latency_percentiles(
    latencies_ms: NDArray[np.float64],
    percentiles: list[float] | None = None,
) -> dict[str, float]:
    """Compute latency percentiles."""
    if percentiles is None:
        percentiles = [50.0, 90.0, 95.0, 99.0, 99.9]

    result = {}
    for p in percentiles:
        key = f"p{p:.0f}" if p == int(p) else f"p{p}"
        result[key] = float(np.percentile(latencies_ms, p))

    result["mean"] = float(np.mean(latencies_ms))
    result["min"] = float(np.min(latencies_ms))
    result["max"] = float(np.max(latencies_ms))
    return result


def compute_spectral_norm_product(
    weight_matrices: list[NDArray[np.float64]],
    num_iterations: int = 10,
) -> float:
    """Compute the spectral norm product Gamma for a neural network."""
    gamma = 1.0
    for W in weight_matrices:
        u = np.random.randn(W.shape[0])
        u = u / np.linalg.norm(u)
        for _ in range(num_iterations):
            v = W.T @ u
            v = v / np.linalg.norm(v)
            u = W @ v
            u = u / np.linalg.norm(u)
        spectral_norm = float(u @ W @ v)
        gamma *= abs(spectral_norm)
    return gamma


def compute_rademacher_bound(
    gamma: float,
    encoding_norm: float,
    num_layers: int,
    num_samples: int,
    calibration_constant: float = 2.3,
) -> float:
    """Compute the Rademacher complexity generalization bound."""
    bound = (
        calibration_constant
        * encoding_norm
        * gamma
        * np.sqrt(2 * num_layers * np.log(2))
        / np.sqrt(num_samples)
    )
    return float(bound)


def compute_sample_complexity(
    num_joins: int,
    d_pred: int,
    epsilon: float,
    delta: float = 0.05,
) -> int:
    """Compute theoretical sample complexity for PAC learnability."""
    m = (num_joins**3 * d_pred + np.log(1.0 / delta)) / (epsilon**2)
    return int(np.ceil(m))


def compute_proxy_a_distance(
    source_encodings: NDArray[np.float64],
    target_encodings: NDArray[np.float64],
    num_folds: int = 5,
) -> float:
    """Compute the proxy A-distance for drift detection."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    X = np.vstack([source_encodings, target_encodings])
    y = np.array([1] * len(source_encodings) + [0] * len(target_encodings))

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    scores = cross_val_score(clf, X, y, cv=num_folds, scoring="accuracy")

    epsilon_h = 1.0 - float(np.mean(scores))
    d_a = 2.0 * (1.0 - 2.0 * epsilon_h)
    return max(0.0, d_a)
