<div align="center">

# PAC-CE: PAC Learnability of Join Cardinality Estimation

**A learning-theoretic framework for cardinality estimation with decomposition theorem and practical guarantees**

</div>

---

## Overview

Learned cardinality estimation (CE) achieves strong empirical accuracy on in-distribution queries, yet practitioners lack principled answers to three deployment-critical questions: how much training data suffices, when will accuracy degrade under workload shift, and how can worst-case errors be bounded on unseen queries.

We develop a learning-theoretic framework addressing all three. Our central result is a **decomposition theorem** proving that join CE is PAC-learnable with sample complexity O(k^3 * d_pred / epsilon^2), **independent of database size**, for attribute-disjoint acyclic queries---covering 89% of JOB and 94% of STATS-CEB.

### Key Contributions

1. **Join Cardinality PAC Learnability:** Decomposition theorem with sample complexity independent of database size, providing formal justification for factor-graph approaches
2. **Rademacher Complexity Analysis:** Spectral norm bounds exploiting join graph sparsity, tighter than generic theory by sqrt(m+k+p)
3. **Domain Adaptation for Drift Detection:** Algorithm achieving 91% precision on natural temporal shifts at 4x lower cost than CardOOD, without ground-truth cardinalities
4. **Hybrid Estimation:** Confidence-based switching between learned CE and information-theoretic bounds, reducing OOD P99 q-error by 20x

---

## Installation

### Requirements

- Python 3.10 or 3.11
- Linux (Ubuntu 22.04+ recommended)

### Setup

```bash
conda create -n pac_ce python=3.11 -y
conda activate pac_ce
pip install -e .
# Optional: database connectors
pip install -e ".[db]"
```

---

## Quick Start

```bash
# 1. Run the system
python run_system.py --config configs/default.yaml

# 2. Quick demo
python demo.py
```

---

## Benchmarks

| Benchmark | Tables | Queries | Attr-Disjoint | Source |
|-----------|--------|---------|---------------|--------|
| JOB | 21 | 113 | 89% | [GitHub](https://github.com/gregrahn/join-order-benchmark) |
| STATS-CEB | 8 | 146 | 94% | [GitHub](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark) |
| TPC-DS | 24 | 99 templates | N/A | [TPC](https://www.tpc.org/tpcds/) |

```

---

## Experiments

| Experiment | Description | Command |
|------------|-------------|---------|
| Main Comparison | Cross-benchmark prediction and hybrid estimation | `make run-main` |
| Scalability | TPC-DS across scale factors 1, 10, 100 | `make run-scalability` |
| Ablation | Component contribution analysis | `make run-ablation` |
| Microbenchmark | Per-query overhead measurements | `make run-micro` |

```bash
make run-all               # all experiments
```

---

## Results

### Cross-Benchmark Prediction (JOB -> STATS-CEB)

| Architecture | Gamma | Predicted | Empirical | Ratio | Effect Size |
|-------------|-------|-----------|-----------|-------|-------------|
| MSCN | 142.3 | 0.89 | 0.71 | 1.25 | 0.42 |
| NeuroCard | 98.7 | 0.76 | 0.63 | 1.21 | 0.38 |
| DeepDB | 156.2 | 0.94 | 0.75 | 1.25 | 0.45 |
| FLAT | 112.4 | 0.82 | 0.68 | 1.21 | 0.40 |
| **Mean** | --- | --- | --- | **1.23** | 0.41 |

### Hybrid Estimation (q-error, mean +/- 95% CI)

| Method | Median | P99 | OOD Median | OOD P99 |
|--------|--------|-----|------------|---------|
| MSCN | 2.1 +/- 0.2 | 182 +/- 24 | 15.3 +/- 2.1 | 2841 +/- 412 |
| NeuroCard | 1.8 +/- 0.2 | 156 +/- 21 | 12.8 +/- 1.8 | 2156 +/- 324 |
| SafeBound | 8.4 +/- 0.8 | 42 +/- 6 | 8.9 +/- 0.9 | 48 +/- 7 |
| **Hybrid (NC)** | **2.0 +/- 0.2** | **35 +/- 5** | **3.8 +/- 0.4** | **108 +/- 15** |

### Drift Detection (Natural Temporal Shifts)

| Scenario | d_A | Q-Error | Detected | Ground Truth |
|----------|-----|---------|----------|-------------|
| IMDB 2013->2015 | 0.18 | 1.4x | No | Benign |
| IMDB 2013->2021 | 0.52 | 4.2x | Yes | Harmful |
| IMDB 2013->2023 | 0.68 | 7.8x | Yes | Harmful |
| SO 2019Q1->2022Q1 | 0.54 | 3.9x | Yes | Harmful |
| JOB->STATS-CEB | 0.81 | 12.3x | Yes | Harmful |

**Precision: 91% | Recall: 84% | r^2: 0.89**

### End-to-End Query Execution (JOB, seconds)

| Estimator | Median | Mean | P95 |
|-----------|--------|------|-----|
| PostgreSQL | 12.4 +/- 1.2 | 28.7 +/- 3.4 | 89.2 +/- 12.1 |
| NeuroCard | 7.8 +/- 0.7 | 17.1 +/- 1.9 | 52.3 +/- 7.4 |
| **Hybrid (NC)** | **7.5 +/- 0.7** | **14.8 +/- 1.6** | **38.6 +/- 5.2** |
| True Cards | 6.8 +/- 0.6 | 12.1 +/- 1.3 | 32.4 +/- 4.1 |

### Scalability (TPC-DS)

| Scale | MSCN Time | NC Time | Hybrid Time | MSCN Q-Err | NC Q-Err | Hybrid Q-Err |
|-------|-----------|---------|-------------|------------|----------|--------------|
| SF1 | 0.3h | 1.2h | 0.4h | 2.8 +/- 0.3 | 2.1 +/- 0.2 | 2.3 +/- 0.2 |
| SF10 | 0.5h | 2.8h | 0.6h | 3.1 +/- 0.3 | 2.4 +/- 0.2 | 2.5 +/- 0.2 |
| SF100 | 0.8h | 8.4h | 1.0h | 3.4 +/- 0.4 | 2.7 +/- 0.3 | 2.8 +/- 0.3 |

---

## Reproducibility

- Pinned dependencies ([requirements.txt](requirements.txt))
- System warmup before measurement
- 5 independent runs with median-seed reporting and 95% CIs
- All experiment configs provided
- Hardware specification documented

### Hardware

| Setup | CPU | RAM | Storage | GPU | Time (Full) |
|-------|-----|-----|---------|-----|-------------|
| Minimum | 8 cores | 32GB | 100GB SSD | None | --- |
| Paper | 64-core AMD EPYC | 512GB | 2TB NVMe | A100 40GB | ~24h |

---

## Acknowledgements

We thank the anonymous reviewers for their valuable feedback.

---

<div align="center">

**[Report Bug](https://github.com/bettyguo/JoinPAC/issues)** |
**[Request Feature](https://github.com/bettyguo/JoinPAC/issues)**

</div>
