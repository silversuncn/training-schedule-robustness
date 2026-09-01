#!/usr/bin/env python3
"""Verify public result tables for the training-schedule robustness bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
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
    "invscaling_01",
    "adaptive_01",
}
FORBIDDEN_PUBLIC_TOKENS = (
    "/" + "home" + "/",
    "/" + "Users" + "/",
    "Evo" + "Scientist",
    "clau" + "de",
    "open" + "ai",
    "s" + "k-",
)


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(name: str) -> dict[str, object]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def rounded_display(value: float, places: int) -> float:
    scale = 10 ** places
    return math.floor(value * scale + 0.500000001) / scale


def ensure_no_public_path_fields(rows: list[dict[str, str]]) -> None:
    forbidden = {"run_dir", "workdir", "output_dir", "path"}
    present = forbidden.intersection(rows[0].keys() if rows else set())
    if present:
        raise AssertionError(f"forbidden path-like columns present: {sorted(present)}")


def assert_sanitized_json(label: str, payload: dict[str, object]) -> None:
    text = json.dumps(payload, sort_keys=True)
    hits = [
        token for token in FORBIDDEN_PUBLIC_TOKENS
        if token.lower() in text.lower()
    ]
    if hits:
        raise AssertionError(f"{label} contains non-public tokens: {hits}")


def assert_public_source(
    label: str,
    payload: dict[str, object],
    expected_source: str,
    expected_sha256: str,
) -> None:
    if payload["source_csv"] != expected_source:
        raise AssertionError(f"{label} source_csv mismatch")
    if payload["source_csv_sha256"] != expected_sha256:
        raise AssertionError(f"{label} source_csv_sha256 mismatch")
    assert_sanitized_json(label, payload)


def assert_signflip_summary(
    label: str,
    payload: dict[str, object],
    *,
    expected_status: str,
    expected_count: int,
    expected_paired_t_holm: int,
    expected_signflip_holm: int,
    expected_agreement: int,
    expected_disagreement: int,
) -> None:
    if payload["status"] != expected_status:
        raise AssertionError(f"{label} status mismatch")
    summary = payload["summary"]
    comparisons = payload["comparisons"]
    if summary["comparison_count"] != expected_count or len(comparisons) != expected_count:
        raise AssertionError(f"{label} comparison count mismatch")
    if summary["paired_t_holm_significant_count"] != expected_paired_t_holm:
        raise AssertionError(f"{label} paired-t Holm count mismatch")
    if summary["signflip_holm_significant_count"] != expected_signflip_holm:
        raise AssertionError(f"{label} sign-flip Holm count mismatch")
    if summary["conclusion_agreement_count"] != expected_agreement:
        raise AssertionError(f"{label} agreement count mismatch")
    if summary["conclusion_disagreement_count"] != expected_disagreement:
        raise AssertionError(f"{label} disagreement count mismatch")
    if summary["signflip_iterations"] != 10000 or summary["signflip_base_seed"] != 20260901:
        raise AssertionError(f"{label} sign-flip protocol mismatch")

    paired_t_holm = sum(1 for comparison in comparisons if comparison["holm_significant_0_05"])
    signflip_holm = sum(1 for comparison in comparisons if comparison["signflip_holm_significant_0_05"])
    agreements = sum(
        1 for comparison in comparisons
        if comparison["conclusion_agrees_t_holm_vs_signflip_holm"]
    )
    if paired_t_holm != expected_paired_t_holm:
        raise AssertionError(f"{label} comparison-level paired-t Holm count mismatch")
    if signflip_holm != expected_signflip_holm:
        raise AssertionError(f"{label} comparison-level sign-flip Holm count mismatch")
    if agreements != expected_agreement:
        raise AssertionError(f"{label} comparison-level agreement count mismatch")


def build_report() -> dict[str, object]:
    public_summary = read_json("public_summary.json")
    primary_rows = read_csv("primary_runs_v2.csv")
    primary_sensitivity = read_csv("primary_sensitivity_v2.csv")
    primary_summary = read_json("primary_summary_v2.json")
    primary_stats = read_json("primary_statistical_analysis_v2.json")
    ag_rows = read_csv("ag_news_runs_v2.csv")
    ag_means = read_csv("ag_news_schedule_means_v2.csv")
    ag_sensitivity = read_csv("ag_news_sensitivity_v2.csv")
    ag_pairwise = read_csv("ag_news_pairwise_v2.csv")
    ag_summary = read_json("ag_news_summary_v2.json")
    ag_stats = read_json("ag_news_statistical_analysis_v2.json")
    signflip_primary = read_json("signflip_primary_108.json")
    signflip_ag_news = read_json("signflip_agnews_84.json")

    ensure_no_public_path_fields(primary_rows)
    ensure_no_public_path_fields(ag_rows)

    if len(primary_rows) != 1080:
        raise AssertionError(f"primary rows: observed {len(primary_rows)}, expected 1080")
    if len(primary_sensitivity) != 12:
        raise AssertionError(f"primary sensitivity rows: observed {len(primary_sensitivity)}, expected 12")
    if len(ag_rows) != 280:
        raise AssertionError(f"AG News rows: observed {len(ag_rows)}, expected 280")
    if len(ag_means) != 28:
        raise AssertionError(f"AG News mean rows: observed {len(ag_means)}, expected 28")
    if len(ag_sensitivity) != 4:
        raise AssertionError(f"AG News sensitivity rows: observed {len(ag_sensitivity)}, expected 4")
    if len(ag_pairwise) != 84:
        raise AssertionError(f"AG News pairwise rows: observed {len(ag_pairwise)}, expected 84")

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

    primary_macro_f1 = defaultdict(list)
    for row in primary_rows:
        key = (row["dataset"], int(row["train_per_class"]), row["schedule"])
        primary_macro_f1[key].append(float(row["test_macro_f1"]))
    primary_means = {
        key: sum(values) / len(values)
        for key, values in primary_macro_f1.items()
    }
    primary_sensitivity_keys = {
        (row["dataset"], int(row["train_per_class"]))
        for row in primary_sensitivity
    }
    expected_primary_sensitivity_keys = {
        (dataset, budget)
        for dataset in PRIMARY_DATASETS
        for budget in PRIMARY_BUDGETS
    }
    if primary_sensitivity_keys != expected_primary_sensitivity_keys:
        raise AssertionError("primary sensitivity key set mismatch")
    for row in primary_sensitivity:
        dataset = row["dataset"]
        budget = int(row["train_per_class"])
        schedule_means = {
            schedule: primary_means[(dataset, budget, schedule)]
            for schedule in PRIMARY_SCHEDULES
        }
        best_value = max(schedule_means.values())
        worst_value = min(schedule_means.values())
        tied_best_schedules = {
            schedule for schedule, value in schedule_means.items()
            if math.isclose(value, best_value, rel_tol=0.0, abs_tol=1e-12)
        }
        tied_worst_schedules = {
            schedule for schedule, value in schedule_means.items()
            if math.isclose(value, worst_value, rel_tol=0.0, abs_tol=1e-12)
        }
        computed_range = best_value - worst_value
        assert_close(f"{dataset}@{budget} primary range", float(row["macro_f1_range"]), computed_range)
        assert_close(f"{dataset}@{budget} best macro-F1", float(row["best_macro_f1"]), best_value)
        assert_close(f"{dataset}@{budget} worst macro-F1", float(row["worst_macro_f1"]), worst_value)
        if row["best_schedule"] not in tied_best_schedules or row["worst_schedule"] not in tied_worst_schedules:
            raise AssertionError(f"{dataset}@{budget} best/worst schedule mismatch")
        if int(row["schedule_count"]) != 9:
            raise AssertionError(f"{dataset}@{budget} primary schedule count mismatch")
        if int(row["convergence_warning_count"]) != 0:
            raise AssertionError(f"{dataset}@{budget} primary sensitivity warning mismatch")
        if float(row["runtime_ratio"]) < 1.0:
            raise AssertionError(f"{dataset}@{budget} primary runtime ratio must be >= 1")

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
    primary_sensitivity_by_key = {
        (row["dataset"], int(row["train_per_class"])): row
        for row in primary_sensitivity
    }
    primary_display_expectations = {
        ("breast_cancer", 4): ("adaptive_0003", 0.9163, "adaptive_003", 0.9107, 0.0056, 1.09),
        ("breast_cancer", 8): ("constant_001", 0.9307, "invscaling_0003", 0.9212, 0.0094, 1.17),
        ("breast_cancer", 16): ("adaptive_003", 0.9400, "invscaling_0003", 0.9170, 0.0230, 1.28),
        ("breast_cancer", 32): ("constant_003", 0.9456, "invscaling_0003", 0.9197, 0.0260, 1.32),
        ("digits", 4): ("constant_001", 0.8030, "invscaling_001", 0.7438, 0.0592, 3.82),
        ("digits", 8): ("constant_003", 0.8532, "invscaling_001", 0.7730, 0.0802, 5.60),
        ("digits", 16): ("adaptive_003", 0.9036, "invscaling_0003", 0.7983, 0.1053, 7.26),
        ("digits", 32): ("constant_003", 0.9336, "invscaling_0003", 0.8166, 0.1169, 7.43),
        ("wine", 4): ("adaptive_003", 0.9115, "invscaling_001", 0.8736, 0.0379, 1.25),
        ("wine", 8): ("adaptive_003", 0.9587, "invscaling_001", 0.8914, 0.0673, 1.37),
        ("wine", 16): ("adaptive_003", 0.9616, "invscaling_001", 0.8870, 0.0746, 1.67),
        ("wine", 32): ("adaptive_003", 0.9678, "invscaling_001", 0.8805, 0.0873, 2.33),
    }
    for key, expected in primary_display_expectations.items():
        row = primary_sensitivity_by_key[key]
        best_schedule, best_f1, worst_schedule, worst_f1, macro_range, runtime_ratio = expected
        if row["best_schedule"] != best_schedule or row["worst_schedule"] != worst_schedule:
            raise AssertionError(f"{key} displayed primary schedule mismatch")
        assert_close(f"{key} displayed primary range", rounded_display(float(row["macro_f1_range"]), 4), macro_range)
        assert_close(f"{key} displayed primary best F1", rounded_display(float(row["best_macro_f1"]), 4), best_f1)
        assert_close(f"{key} displayed primary worst F1", rounded_display(float(row["worst_macro_f1"]), 4), worst_f1)
        assert_close(f"{key} displayed primary runtime ratio", rounded_display(float(row["runtime_ratio"]), 2), runtime_ratio)

    headline = public_summary["headline_values"]
    assert_close("AG News 64 macro-F1 range", float(ag64["macro_f1_range"]), float(headline["ag_news_max_macro_f1_range"]))
    assert_close("AG News 64 accuracy range", float(ag64["accuracy_range"]), float(headline["ag_news_max_accuracy_range"]))
    if ag64["best_schedule"] != "adaptive_01" or ag64["worst_schedule"] != "constant_001":
        raise AssertionError("AG News 64 best/worst schedule mismatch")

    if primary_summary["row_count"] != 1080 or primary_summary["expected_rows"] != 1080:
        raise AssertionError("primary summary row count mismatch")
    if public_summary["row_counts"]["primary_sensitivity_rows"] != 12:
        raise AssertionError("primary sensitivity public summary row count mismatch")
    if primary_stats["family_size"] != 108 or len(primary_stats["comparisons"]) != 108:
        raise AssertionError("primary paired comparison count mismatch")
    if ag_summary["row_count"] != 280 or ag_summary["expected_rows"] != 280:
        raise AssertionError("AG News summary row count mismatch")
    ag_comparison_count = ag_stats.get("comparison_count", ag_stats.get("family_size"))
    if ag_comparison_count != 84 or len(ag_stats["comparisons"]) != 84:
        raise AssertionError("AG News paired comparison count mismatch")

    computed_ag_significant = sum(
        1 for comparison in ag_stats["comparisons"]
        if comparison["holm_significant_0_05"]
    )
    if headline["primary_holm_significant_family_comparisons"] != 58:
        raise AssertionError("primary Holm comparison headline mismatch")
    if headline["ag_news_holm_significant_comparisons"] != computed_ag_significant:
        raise AssertionError("AG News Holm comparison headline mismatch")

    expected_primary_hash = sha256_file(DATA / "primary_runs_v2.csv")
    expected_ag_hash = sha256_file(DATA / "ag_news_runs_v2.csv")
    assert_public_source(
        "primary sign-flip",
        signflip_primary,
        "data/primary_runs_v2.csv",
        expected_primary_hash,
    )
    assert_public_source(
        "AG News sign-flip",
        signflip_ag_news,
        "data/ag_news_runs_v2.csv",
        expected_ag_hash,
    )
    assert_signflip_summary(
        "primary sign-flip",
        signflip_primary,
        expected_status="PASS_PRIMARY_SIGNFLIP_108",
        expected_count=108,
        expected_paired_t_holm=58,
        expected_signflip_holm=0,
        expected_agreement=50,
        expected_disagreement=58,
    )
    assert_signflip_summary(
        "AG News sign-flip",
        signflip_ag_news,
        expected_status="PASS_AGNEWS_SIGNFLIP_84",
        expected_count=84,
        expected_paired_t_holm=36,
        expected_signflip_holm=0,
        expected_agreement=48,
        expected_disagreement=36,
    )
    signflip_summary = public_summary["signflip_sensitivity"]
    if signflip_summary["primary_108"]["signflip_holm_significant_count"] != 0:
        raise AssertionError("primary sign-flip public summary mismatch")
    if signflip_summary["ag_news_84"]["signflip_holm_significant_count"] != 0:
        raise AssertionError("AG News sign-flip public summary mismatch")
    if signflip_summary["primary_108"]["conclusion_agreement_count"] != 50:
        raise AssertionError("primary sign-flip agreement summary mismatch")
    if signflip_summary["ag_news_84"]["conclusion_agreement_count"] != 48:
        raise AssertionError("AG News sign-flip agreement summary mismatch")

    return {
        "result": "PASS",
        "row_counts": {
            "primary_runs": len(primary_rows),
            "primary_sensitivity_rows": len(primary_sensitivity),
            "ag_news_runs": len(ag_rows),
            "primary_paired_comparisons": len(primary_stats["comparisons"]),
            "ag_news_paired_comparisons": len(ag_pairwise),
            "primary_signflip_comparisons": len(signflip_primary["comparisons"]),
            "ag_news_signflip_comparisons": len(signflip_ag_news["comparisons"]),
        },
        "headline_values": {
            "ag_news_macro_f1_range": rounded_headline(float(ag64["macro_f1_range"])),
            "ag_news_accuracy_range": rounded_headline(float(ag64["accuracy_range"])),
            "primary_holm_significant": headline["primary_holm_significant_family_comparisons"],
            "ag_news_holm_significant": headline["ag_news_holm_significant_comparisons"],
            "primary_signflip_holm_significant": signflip_primary["summary"]["signflip_holm_significant_count"],
            "ag_news_signflip_holm_significant": signflip_ag_news["summary"]["signflip_holm_significant_count"],
            "primary_signflip_conclusion_agreement": signflip_primary["summary"]["conclusion_agreement_count"],
            "ag_news_signflip_conclusion_agreement": signflip_ag_news["summary"]["conclusion_agreement_count"],
        },
    }


def main() -> None:
    print(json.dumps(build_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
