# Training-Schedule Robustness of Lightweight Classifiers under Limited Labels

> **Training-Schedule Robustness of Lightweight Classifiers under Limited Labels**  
> Yaowen Sun

## Overview

This repository contains a sanitized reproduction bundle for an empirical study
of learning-rate schedule sensitivity in fixed lightweight logistic-loss
stochastic-gradient classifiers under limited labels. The bundle includes
processed result tables, sanitized figures, a lightweight verification script,
and tests for checking matrix completeness and headline values.

Repository URL: https://github.com/silversuncn/training-schedule-robustness

The primary tabular matrix contains all 1,080 expected runs across three
built-in datasets, four train-per-class budgets, ten deterministic stratified
seeds, and nine schedule settings. A real-text AG News extension adds 240 runs
with TF-IDF features, four train-per-class budgets, ten seeds, and six schedule
settings. The primary metric is macro-F1.

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
│   ├── formal_statistical_analysis.json
│   ├── text_schedule_formal_ag_news_results.csv
│   ├── text_schedule_formal_ag_news_schedule_means.csv
│   ├── text_schedule_formal_ag_news_sensitivity.csv
│   ├── text_schedule_formal_ag_news_pairwise.csv
│   ├── text_schedule_formal_ag_news_config.json
│   └── text_schedule_formal_ag_news_summary.json
├── figures/
│   ├── formal_macro_f1_sensitivity_20260728.svg
│   ├── ag_news_macro_f1_sensitivity_20260801.svg
│   └── ag_news_macro_f1_sensitivity_20260801.png
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

AG News text extension:

| Dimension | Values | Count |
| --- | --- | ---: |
| Dataset | `ag_news` official train/test split | 1 |
| Evaluation subset | 2,000 balanced official test examples, 500/class | 1 |
| Train-per-class budgets | 16, 32, 64, 128 | 4 |
| Seeds | 14, 52, 88, 121, 169, 203, 247, 314, 415, 509 | 10 |
| Schedule settings | `constant_001`, `invscaling_001`, `adaptive_001`, `optimal_builtin`, `constant_01`, `adaptive_01` | 6 |

Row-count check:

```text
3 datasets x 4 budgets x 10 seeds x 9 schedule settings = 1,080 runs
3 datasets x 4 budgets x 3 eta0 values x 3 family comparisons = 108 paired comparisons
1 text dataset x 4 budgets x 10 seeds x 6 schedule settings = 240 AG News runs
4 text budgets x C(6, 2) schedule-pair comparisons = 60 AG News paired comparisons
```

## Hardware & Environment

The reported matrices were produced on CPU with Python 3.11, NumPy, SciPy,
scikit-learn, and cached HuggingFace Datasets for AG News loading. The AG News
path is classical TF-IDF plus SGD; no Transformer or neural model is evaluated.
Fit-time and iteration-count fields are implementation diagnostics and should
not be treated as portable deployment benchmarks.

## Key Results

- Schedule sensitivity is strongest for `digits@32`, `digits@16`, and `wine@32`.
- Low-budget `breast_cancer` strata show weaker schedule sensitivity.
- All 58 Holm-significant family comparisons involve `constant` or `adaptive`
  outperforming `invscaling`.
- No `adaptive` versus `constant` comparison reaches Holm significance.
- AG News is schedule-sensitive at all four evaluated text budgets; the largest
  text macro-F1 range is 0.3087 at 64 examples per class.
- Higher-step `constant_01` and `adaptive_01` improve over their low-step
  counterparts in every AG News budget stratum.
- The results support validating inverse-scaling defaults in the target setting;
  they do not establish a universal best schedule or neural-model claim.

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
