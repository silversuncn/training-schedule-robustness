#!/usr/bin/env python3
"""Verify public result tables for the training-schedule robustness bundle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

EXPECTED_DATASETS = {"wine", "breast_cancer", "digits"}
EXPECTED_BUDGETS = {4, 8, 16, 32}
EXPECTED_SEEDS = {14, 52, 88, 121, 169, 203, 247, 314, 415, 509}
EXPECTED_SCHEDULES = {
    "constant_0003", "constant_001", "constant_003",
    "invscaling_0003", "invscaling_001", "invscaling_003",
    "adaptive_0003", "adaptive_001", "adaptive_003",
}
EXPECTED_TEXT_BUDGETS = {16, 32, 64, 128}
EXPECTED_TEXT_SCHEDULES = {
    "constant_001", "invscaling_001", "adaptive_001",
    "optimal_builtin", "constant_01", "adaptive_01",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def assert_close(name: str, observed: float, expected: float, tolerance: float = 0.0001) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{name}: observed {observed}, expected {expected}")


def build_report() -> dict[str, object]:
    summary = json.loads((DATA / "public_summary.json").read_text(encoding="utf-8"))
    rows = read_csv("formal_results.csv")
    sensitivity = read_csv("formal_sensitivity.csv")
    text_rows = read_csv("text_schedule_formal_ag_news_results.csv")
    text_means = read_csv("text_schedule_formal_ag_news_schedule_means.csv")
    text_sensitivity = read_csv("text_schedule_formal_ag_news_sensitivity.csv")
    text_pairwise = read_csv("text_schedule_formal_ag_news_pairwise.csv")
    stat = json.loads((DATA / "formal_statistical_analysis.json").read_text(encoding="utf-8"))
    text_summary = json.loads((DATA / "text_schedule_formal_ag_news_summary.json").read_text(encoding="utf-8"))

    if len(rows) != 1080:
        raise AssertionError(f"formal results rows: observed {len(rows)}, expected 1080")
    if len(sensitivity) != 12:
        raise AssertionError(f"sensitivity rows: observed {len(sensitivity)}, expected 12")
    if len(stat["aggregate_rows"]) != 108:
        raise AssertionError(f"aggregate rows: observed {len(stat['aggregate_rows'])}, expected 108")
    if len(stat["paired_family_comparisons"]) != 108:
        raise AssertionError("paired family comparison count must be 108")
    if len(text_rows) != 240:
        raise AssertionError(f"text results rows: observed {len(text_rows)}, expected 240")
    if len(text_means) != 24:
        raise AssertionError(f"text schedule means rows: observed {len(text_means)}, expected 24")
    if len(text_sensitivity) != 4:
        raise AssertionError(f"text sensitivity rows: observed {len(text_sensitivity)}, expected 4")
    if len(text_pairwise) != 60:
        raise AssertionError(f"text paired comparison rows: observed {len(text_pairwise)}, expected 60")

    datasets = {row["dataset"] for row in rows}
    budgets = {int(row["train_per_class"]) for row in rows}
    seeds = {int(row["seed"]) for row in rows}
    schedules = {row["schedule"] for row in rows}
    statuses = {row["status"] for row in rows}
    duplicate_keys = len(rows) - len({(row["dataset"], row["train_per_class"], row["seed"], row["schedule"]) for row in rows})

    if datasets != EXPECTED_DATASETS:
        raise AssertionError(f"datasets: observed {sorted(datasets)}")
    if budgets != EXPECTED_BUDGETS:
        raise AssertionError(f"budgets: observed {sorted(budgets)}")
    if seeds != EXPECTED_SEEDS:
        raise AssertionError(f"seeds: observed {sorted(seeds)}")
    if schedules != EXPECTED_SCHEDULES:
        raise AssertionError(f"schedules: observed {sorted(schedules)}")
    if statuses != {"PASS"}:
        raise AssertionError(f"statuses: observed {sorted(statuses)}")
    if duplicate_keys != 0:
        raise AssertionError(f"duplicate keys: observed {duplicate_keys}")
    if sum(int(row["convergence_warning_count"]) for row in rows) != 0:
        raise AssertionError("convergence warnings must be zero")

    sensitivity_by_key = {(row["dataset"], int(row["train_per_class"])): row for row in sensitivity}
    strongest = sensitivity_by_key[("digits", 32)]
    assert_close("digits@32 macro-F1 range", float(strongest["macro_f1_range"]), 0.1169, tolerance=0.0001)
    if strongest["best_schedule"] != "constant_003" or strongest["worst_schedule"] != "invscaling_0003":
        raise AssertionError("digits@32 best/worst schedules do not match expected headline")

    decision = stat["strengthened_matrix_decision"]
    if decision["significant_comparisons"] != 58 or decision["total_comparisons"] != 108:
        raise AssertionError("Holm-significant comparison counts do not match expected headline")
    if summary["row_counts"]["formal_results_rows"] != 1080:
        raise AssertionError("public summary row count mismatch")

    text_datasets = {row["dataset"] for row in text_rows}
    text_budgets = {int(row["train_per_class"]) for row in text_rows}
    text_seeds = {int(row["seed"]) for row in text_rows}
    text_schedules = {row["schedule"] for row in text_rows}
    text_statuses = {row["status"] for row in text_rows}
    text_duplicate_keys = len(text_rows) - len({
        (row["dataset"], row["train_per_class"], row["seed"], row["schedule"])
        for row in text_rows
    })
    if text_datasets != {"ag_news"}:
        raise AssertionError(f"text datasets: observed {sorted(text_datasets)}")
    if text_budgets != EXPECTED_TEXT_BUDGETS:
        raise AssertionError(f"text budgets: observed {sorted(text_budgets)}")
    if text_seeds != EXPECTED_SEEDS:
        raise AssertionError(f"text seeds: observed {sorted(text_seeds)}")
    if text_schedules != EXPECTED_TEXT_SCHEDULES:
        raise AssertionError(f"text schedules: observed {sorted(text_schedules)}")
    if text_statuses != {"PASS"}:
        raise AssertionError(f"text statuses: observed {sorted(text_statuses)}")
    if text_duplicate_keys != 0:
        raise AssertionError(f"text duplicate keys: observed {text_duplicate_keys}")
    for row in text_rows:
        for field in ["accuracy", "macro_f1", "log_loss", "fit_seconds", "n_iter"]:
            value = float(row[field])
            if not math.isfinite(value):
                raise AssertionError(f"text nonfinite {field}: {value}")

    text_sensitivity_by_budget = {int(row["train_per_class"]): row for row in text_sensitivity}
    ag64 = text_sensitivity_by_budget[64]
    assert_close("AG News @64 macro-F1 range", float(ag64["macro_f1_range"]), 0.3087, tolerance=0.0001)
    if ag64["best_schedule"] != "adaptive_01" or ag64["worst_schedule"] != "constant_001":
        raise AssertionError("AG News @64 best/worst schedules do not match expected headline")
    text_ext = summary["text_extension"]
    if text_ext["row_counts"]["text_formal_results_rows"] != 240:
        raise AssertionError("public summary text row count mismatch")
    if text_summary["status"] != "P21_TEXT_FORMAL_READY_FOR_INTEGRATION_REVIEW":
        raise AssertionError("text summary status mismatch")

    return {
        "status": "PASS",
        "title": summary["title"],
        "row_counts": {
            "formal_results_rows": len(rows),
            "sensitivity_rows": len(sensitivity),
            "aggregate_rows": len(stat["aggregate_rows"]),
            "paired_family_comparisons": len(stat["paired_family_comparisons"]),
            "text_formal_results_rows": len(text_rows),
            "text_schedule_means_rows": len(text_means),
            "text_sensitivity_rows": len(text_sensitivity),
            "text_pairwise_rows": len(text_pairwise),
        },
        "datasets": sorted(datasets),
        "text_datasets": sorted(text_datasets),
        "budgets": sorted(budgets),
        "text_budgets": sorted(text_budgets),
        "seeds": sorted(seeds),
        "schedule_count": len(schedules),
        "text_schedule_count": len(text_schedules),
        "duplicate_keys": duplicate_keys,
        "text_duplicate_keys": text_duplicate_keys,
        "convergence_warning_count": 0,
        "strongest_stratum": "digits@32",
        "strongest_macro_f1_range": round(float(strongest["macro_f1_range"]), 4),
        "holm_significant_family_comparisons": decision["significant_comparisons"],
        "strongest_text_stratum": "ag_news@64",
        "strongest_text_macro_f1_range": round(float(ag64["macro_f1_range"]), 4),
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
