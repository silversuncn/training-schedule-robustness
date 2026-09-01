# Training-Schedule Robustness of a Lightweight Classifier under Limited Labels

> **Training-Schedule Robustness of a Lightweight Classifier under Limited Labels**
> Yaowen Sun; Xin Zhang; Qian Zhang

## Overview

This bundle contains sanitized derived numeric artifacts for an empirical study
of learning-rate schedule sensitivity in a lightweight logistic-loss SGD
classifier under limited labels. It includes aggregate tables, two
manuscript-referenced figures, and small local consistency checks for matrix
completeness and headline values.

This repository does not redistribute raw external datasets and does not include
the original training pipeline, statistical-analysis scripts, or figure
generation scripts. It is a public consistency bundle for the reported derived
results.

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
│   ├── primary_sensitivity_v2.csv
│   ├── primary_summary_v2.json
│   ├── primary_statistical_analysis_v2.json
│   ├── ag_news_runs_v2.csv
│   ├── ag_news_schedule_means_v2.csv
│   ├── ag_news_sensitivity_v2.csv
│   ├── ag_news_pairwise_v2.csv
│   ├── ag_news_summary_v2.json
│   ├── ag_news_statistical_analysis_v2.json
│   ├── signflip_primary_108.json
│   └── signflip_agnews_84.json
├── figures/
│   ├── formal_macro_f1_sensitivity_20260728.png
│   └── fig2_ag_news_sensitivity_v3.png
├── src/
│   └── verify_public_results.py
└── tests/
    └── test_public_results.py
```

## Data Files

The primary matrix covers three built-in datasets, four train-per-class budgets,
ten stratified seeds, and nine schedule settings, giving 1,080 runs. Its
12 dataset-budget sensitivity rows are bundled in `primary_sensitivity_v2.csv`.
The AG News matrix covers four train-per-class budgets, ten seeds, and seven
schedule settings, giving 280 runs. All public AG News rows use
`test_selection_seed=20260830`.

Seed-paired comparisons are summarized at the primary family level for 108
comparisons and at the AG News schedule level for 84 comparisons.
The sign-flip sensitivity files report 10,000-iteration paired permutation
checks for the same 108 primary and 84 AG News comparisons.

## Experimental Setup

Primary datasets are `wine`, `breast_cancer`, and `digits` with train-per-class
budgets 4, 8, 16, and 32. Primary schedule settings combine `constant`,
`invscaling`, and `adaptive` families with initial learning rates 0.0003,
0.001, and 0.003.

AG News uses TF-IDF features with the same logistic-loss SGD learner and
train-per-class budgets 16, 32, 64, and 128. Its seven schedule settings are
`constant_001`, `invscaling_001`, `adaptive_001`, `optimal_builtin`,
`constant_01`, `invscaling_01`, and `adaptive_01`.

## Key Results

The primary matrix has 58 of 108 Holm-adjusted paired family comparisons below
0.05, all involving `invscaling` underperformance. The AG News extension has
36 of 84 Holm-adjusted schedule comparisons below
0.05.
The largest primary sensitivity rows are `digits` at budget 32
(macro-F1 range 0.1169), `digits` at budget 16 (0.1053), and `wine` at budget
32 (0.0873).
In the 10,000-iteration sign-flip sensitivity check, 0 of 108 primary and
0 of 84 AG News comparisons remain Holm-significant; paired-t/Holm and
sign-flip/Holm conclusions agree for 50 of 108 primary and 48 of 84 AG News
comparisons.

The largest AG News separation appears at 64 examples per class: mean macro-F1
ranges by 0.3207 and mean accuracy ranges by
0.2709 across schedule settings. Fit-time and
iteration fields are implementation diagnostics rather than portable deployment
benchmarks.

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

Title: Training-Schedule Robustness of a Lightweight Classifier under Limited Labels

Authors: Yaowen Sun; Xin Zhang; Qian Zhang

Year: 2026

## License

Released under the MIT License. See `LICENSE`.
