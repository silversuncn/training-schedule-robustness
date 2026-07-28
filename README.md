# Training-Schedule Robustness of Lightweight Classifiers under Limited Labels

> **Training-Schedule Robustness of Lightweight Classifiers under Limited Labels**  
> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for an empirical study
of learning-rate schedule sensitivity in a fixed lightweight logistic-loss
stochastic-gradient classifier under limited labels. The bundle includes
processed result tables, a sanitized figure, a lightweight verification script,
and tests for checking matrix completeness and headline values.

Repository URL: https://github.com/silversuncn/training-schedule-robustness

The formal matrix contains all 1,080 expected runs across three built-in
datasets, four train-per-class budgets, ten deterministic stratified seeds, and
nine schedule settings. The primary metric is macro-F1.

## Repository Structure

```text
.
├── README.md
├── CITATION.cff
├── LICENSE
├── requirements.txt
├── data/
│   ├── README.md
│   ├── public_summary.json
│   ├── formal_results.csv
│   ├── formal_sensitivity.csv
│   └── formal_statistical_analysis.json
├── figures/
│   └── formal_macro_f1_sensitivity_20260728.svg
├── src/
│   └── verify_public_results.py
└── tests/
    └── test_public_results.py
```

## Experimental Setup

| Dimension | Values | Count |
| --- | --- | ---: |
| Datasets | `wine`, `breast_cancer`, `digits` | 3 |
| Train-per-class budgets | 4, 8, 16, 32 | 4 |
| Stratified seeds | 14, 52, 88, 121, 169, 203, 247, 314, 415, 509 | 10 |
| Schedule families | `constant`, `invscaling`, `adaptive` | 3 |
| Initial learning rates | 0.0003, 0.001, 0.003 | 3 |

Row-count check:

```text
3 datasets x 4 budgets x 10 seeds x 9 schedule settings = 1,080 runs
3 datasets x 4 budgets x 3 eta0 values x 3 family comparisons = 108 paired comparisons
```

## Hardware & Environment

The reported matrix was produced on CPU with Python 3.11, NumPy, SciPy, and
scikit-learn. Fit-time and iteration-count fields are implementation diagnostics
and should not be treated as portable deployment benchmarks.

## Key Results

- Schedule sensitivity is strongest for `digits@32`, `digits@16`, and `wine@32`.
- Low-budget `breast_cancer` strata show weaker schedule sensitivity.
- All 58 Holm-significant family comparisons involve `constant` or `adaptive`
  outperforming `invscaling`.
- No `adaptive` versus `constant` comparison reaches Holm significance.
- The results support validating inverse-scaling defaults in the target setting;
  they do not establish a universal best schedule.

## Requirements

The included verification script uses only the Python standard library. The
`requirements.txt` file records the main packages used by the experiment
environment.

Verify the public tables:

```bash
python src/verify_public_results.py
```

Run the tests:

```bash
python -m unittest discover -s tests -q
```

## Citation

```bibtex
@article{sun2026trainingschedulerobustness,
  title = {Training-Schedule Robustness of Lightweight Classifiers under Limited Labels},
  author = {Sun, Yaowen},
  year = {2026}
}
```
