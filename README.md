# Training-Schedule Robustness of Lightweight Classifiers under Limited Labels

> **Training-Schedule Robustness of Lightweight Classifiers under Limited Labels**  
> Yaowen Sun; Xin Zhang; Qian Zhang

## Overview

This bundle contains public numeric artifacts for an empirical study of learning-rate schedule sensitivity in lightweight logistic-loss SGD classifiers under limited labels. It includes sanitized tables, two manuscript-referenced figures, and small local checks for matrix completeness and headline values.

## Repository Structure

```text
.
├── README.md
├── LICENSE
├── MANIFEST.sha256
├── requirements.txt
├── data/
│   ├── README.md
│   ├── public_summary.json
│   ├── primary_runs_v2.csv
│   ├── primary_summary_v2.json
│   ├── primary_statistical_analysis_v2.json
│   ├── ag_news_runs_v2.csv
│   ├── ag_news_schedule_means_v2.csv
│   ├── ag_news_sensitivity_v2.csv
│   ├── ag_news_pairwise_v2.csv
│   ├── ag_news_summary_v2.json
│   └── ag_news_statistical_analysis_v2.json
├── figures/
│   ├── formal_macro_f1_sensitivity_20260728.png
│   └── fig2_ag_news_sensitivity_v2.png
├── src/
│   └── verify_public_results.py
└── tests/
    └── test_public_results.py
```

## Data Files

The primary matrix covers three built-in datasets, four train-per-class budgets, ten stratified seeds, and nine schedule settings, giving 1,080 runs. The AG News matrix covers four train-per-class budgets, ten seeds, and six schedule settings, giving 240 runs.

Seed-paired comparisons are summarized at the primary family level for 108 comparisons and at the AG News schedule level for 60 comparisons.

## Experimental Setup

Primary datasets are `wine`, `breast_cancer`, and `digits` with train-per-class budgets 4, 8, 16, and 32. Primary schedule settings combine `constant`, `invscaling`, and `adaptive` families with initial learning rates 0.0003, 0.001, and 0.003.

AG News uses TF-IDF features with the same logistic-loss SGD learner and train-per-class budgets 16, 32, 64, and 128. Its six schedule settings are `constant_001`, `invscaling_001`, `adaptive_001`, `optimal_builtin`, `constant_01`, and `adaptive_01`.

## Key Results

The primary matrix has 58 of 108 Holm-adjusted paired family comparisons below 0.05, all involving `invscaling` underperformance. The AG News extension has 27 of 60 Holm-adjusted schedule comparisons below 0.05.

The largest AG News separation appears at 64 examples per class: mean macro-F1 ranges by 0.3207 and mean accuracy ranges by 0.2709 across schedule settings. Fit-time and iteration fields are implementation diagnostics rather than portable deployment benchmarks.

## Hardware & Environment

The recorded v2 package snapshot identifies Linux under WSL2 with NumPy
`2.4.4`, SciPy `1.17.1`, Pandas `3.0.1`, scikit-learn `1.8.0`, Matplotlib
`3.10.8`, and Datasets `4.8.4`. The public verification script is CPU-only and
uses bundled CSV/JSON files.

## Verification

The verification script uses only the Python standard library:

```bash
python src/verify_public_results.py
python -m unittest discover -s tests -q
```

## Requirements

The bundled verifier uses the Python standard library. Re-running the original
matrix or figure generation requires the packages listed in `requirements.txt`.

## Citation

Title: Training-Schedule Robustness of Lightweight Classifiers under Limited Labels

Authors: Yaowen Sun; Xin Zhang; Qian Zhang

Year: 2026

## License

Released under the MIT License. See `LICENSE`.
