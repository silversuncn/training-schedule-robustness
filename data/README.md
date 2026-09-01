# Data Files

The tables contain public derived numeric artifacts for the training-schedule
robustness study.

- `primary_runs_v2.csv`: 1,080 primary rows over three datasets, four budgets, ten seeds, and nine schedule settings.
- `primary_sensitivity_v2.csv`: 12 primary dataset-budget sensitivity rows with macro-F1 ranges, best/worst schedules, runtime ratios, and interpretability flags.
- `primary_summary_v2.json`: primary matrix dimensions and fit protocol.
- `primary_statistical_analysis_v2.json`: 108 seed-paired family comparisons with effect sizes and Holm adjustment.
- `ag_news_runs_v2.csv`: 280 AG News rows over four budgets, ten seeds, seven schedule settings, and `test_selection_seed=20260830`.
- `ag_news_schedule_means_v2.csv`: AG News mean metrics by budget and schedule.
- `ag_news_sensitivity_v2.csv`: AG News per-budget macro-F1, accuracy, log-loss, runtime, and iteration ranges.
- `ag_news_pairwise_v2.csv`: 84 AG News seed-paired schedule comparisons.
- `ag_news_summary_v2.json`: AG News dimensions, public split details, and headline ranges.
- `ag_news_statistical_analysis_v2.json`: AG News comparison summary with Holm adjustment.
- `signflip_primary_108.json`: 108 primary sign-flip sensitivity comparisons using 10,000 paired sign permutations.
- `signflip_agnews_84.json`: 84 AG News sign-flip sensitivity comparisons using 10,000 paired sign permutations.
- `public_summary.json`: compact title, author names, row counts, headline values, and figure list.

The bundle is limited to sanitized derived artifacts and a consistency checker;
it does not include original raw datasets, the original training pipeline,
statistical-analysis scripts, or figure-generation scripts.
