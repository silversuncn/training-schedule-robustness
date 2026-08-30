#!/usr/bin/env python3
"""Verify public result tables for the training-schedule robustness bundle."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

PRIMARY_DATASETS = {"wine", "breast_cancer", "digits"}
PRIMARY_BUDGETS = {4, 8, 16, 32}
SEEDS = {14, 52, 88, 121, 169, 203, 247, 314, 415, 509}
PRIMARY_SCHEDULES = {
    "constant_0003",
    "constant_001",
    "constant_003",
    "invscaling_0003",
    "invscaling_001",
    "invscaling_003",
    "adaptive_0003",
    "adaptive_001",
    "adaptive_003",
}
AG_NEWS_BUDGETS = {16, 32, 64, 128}
AG_NEWS_SCHEDULES = {
    "constant_001",
    "invscaling_001",
    "adaptive_001",
    "optimal_builtin",
    "constant_01",
    "adaptive_01",
}


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def assert_close(label: str, observed: float, expected: float, tolerance: float = 0.0001) -> None:
    if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=tolerance):
        raise AssertionError(f"{label}: observed {observed}, expected {expected}")


def assert_finite(rows: list[dict[str, str]], fields: list[str]) -> None:
    for row in rows:
        for field in fields:
            value = float(row[field])
            if not math.isfinite(value):
                raise AssertionError(f"{field} is not finite: {value}")


def rounded_headline(value: float) -> float:
    return math.floor(value * 10000.0 + 0.500000001) / 10000.0


def build_report() -> dict[str, object]:
    public_summary = read_json("public_summary.json")
    primary_rows = read_csv("primary_runs_v2.csv")
    primary_summary = read_json("primary_summary_v2.json")
    primary_stats = read_json("primary_statistical_analysis_v2.json")
    ag_rows = read_csv("ag_news_runs_v2.csv")
    ag_means = read_csv("ag_news_schedule_means_v2.csv")
    ag_sensitivity = read_csv("ag_news_sensitivity_v2.csv")
    ag_pairwise = read_csv("ag_news_pairwise_v2.csv")
    ag_summary = read_json("ag_news_summary_v2.json")
    ag_stats = read_json("ag_news_statistical_analysis_v2.json")

    if len(primary_rows) != 1080:
        raise AssertionError(f"primary rows: observed {len(primary_rows)}, expected 1080")
    if len(ag_rows) != 240:
        raise AssertionError(f"AG News rows: observed {len(ag_rows)}, expected 240")
    if len(ag_means) != 24:
        raise AssertionError(f"AG News mean rows: observed {len(ag_means)}, expected 24")
    if len(ag_sensitivity) != 4:
        raise AssertionError(f"AG News sensitivity rows: observed {len(ag_sensitivity)}, expected 4")
    if len(ag_pairwise) != 60:
        raise AssertionError(f"AG News pairwise rows: observed {len(ag_pairwise)}, expected 60")

    primary_keys = {
        (row["dataset"], row["train_per_class"], row["seed"], row["schedule"])
        for row in primary_rows
    }
    ag_keys = {
        (row["dataset"], row["train_per_class"], row["seed"], row["schedule"])
        for row in ag_rows
    }
    if len(primary_keys) != len(primary_rows):
        raise AssertionError("primary matrix has duplicate keys")
    if len(ag_keys) != len(ag_rows):
        raise AssertionError("AG News matrix has duplicate keys")

    if {row["dataset"] for row in primary_rows} != PRIMARY_DATASETS:
        raise AssertionError("primary dataset set mismatch")
    if {int(row["train_per_class"]) for row in primary_rows} != PRIMARY_BUDGETS:
        raise AssertionError("primary budget set mismatch")
    if {int(row["seed"]) for row in primary_rows} != SEEDS:
        raise AssertionError("primary seed set mismatch")
    if {row["schedule"] for row in primary_rows} != PRIMARY_SCHEDULES:
        raise AssertionError("primary schedule set mismatch")
    if {int(row["max_iter"]) for row in primary_rows} != {1000}:
        raise AssertionError("primary fit iteration cap mismatch")
    if sum(int(row["convergence_warning_count"]) for row in primary_rows) != 0:
        raise AssertionError("primary convergence warning count must be zero")
    assert_finite(primary_rows, ["test_log_loss", "test_macro_f1", "final_learning_rate"])

    if {row["dataset"] for row in ag_rows} != {"ag_news"}:
        raise AssertionError("AG News dataset set mismatch")
    if {int(row["train_per_class"]) for row in ag_rows} != AG_NEWS_BUDGETS:
        raise AssertionError("AG News budget set mismatch")
    if {int(row["seed"]) for row in ag_rows} != SEEDS:
        raise AssertionError("AG News seed set mismatch")
    if {int(row["test_selection_seed"]) for row in ag_rows} != {20260830}:
        raise AssertionError("AG News test-selection seed mismatch")
    if {row["schedule"] for row in ag_rows} != AG_NEWS_SCHEDULES:
        raise AssertionError("AG News schedule set mismatch")
    if sum(int(row["convergence_warning_count"]) for row in ag_rows) != 0:
        raise AssertionError("AG News convergence warning count must be zero")
    assert_finite(ag_rows, ["accuracy", "macro_f1", "log_loss", "fit_seconds"])

    ag64 = {int(row["train_per_class"]): row for row in ag_sensitivity}[64]
    assert_close("AG News 64 macro-F1 range", float(ag64["macro_f1_range"]), 0.3207227955755691)
    assert_close("AG News 64 accuracy range", float(ag64["accuracy_range"]), 0.2708499999999999)
    if ag64["best_schedule"] != "adaptive_01" or ag64["worst_schedule"] != "constant_001":
        raise AssertionError("AG News 64 best/worst schedule mismatch")

    if primary_summary["row_count"] != 1080 or primary_summary["expected_rows"] != 1080:
        raise AssertionError("primary summary row count mismatch")
    if primary_stats["family_size"] != 108 or len(primary_stats["comparisons"]) != 108:
        raise AssertionError("primary paired comparison count mismatch")
    if ag_summary["row_count"] != 240 or ag_summary["expected_rows"] != 240:
        raise AssertionError("AG News summary row count mismatch")
    if ag_stats["family_size"] != 60 or len(ag_stats["comparisons"]) != 60:
        raise AssertionError("AG News paired comparison count mismatch")

    headline = public_summary["headline_values"]
    assert_close("public macro-F1 range", float(headline["ag_news_max_macro_f1_range"]), 0.3207)
    assert_close("public accuracy range", float(headline["ag_news_max_accuracy_range"]), 0.2709)
    if headline["primary_holm_significant_family_comparisons"] != 58:
        raise AssertionError("primary Holm comparison headline mismatch")
    if headline["ag_news_holm_significant_comparisons"] != 27:
        raise AssertionError("AG News Holm comparison headline mismatch")

    return {
        "result": "PASS",
        "row_counts": {
            "primary_runs": len(primary_rows),
            "ag_news_runs": len(ag_rows),
            "primary_paired_comparisons": len(primary_stats["comparisons"]),
            "ag_news_paired_comparisons": len(ag_pairwise),
        },
        "headline_values": {
            "ag_news_macro_f1_range": rounded_headline(float(ag64["macro_f1_range"])),
            "ag_news_accuracy_range": rounded_headline(float(ag64["accuracy_range"])),
            "primary_holm_significant": headline["primary_holm_significant_family_comparisons"],
            "ag_news_holm_significant": headline["ag_news_holm_significant_comparisons"],
        },
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
