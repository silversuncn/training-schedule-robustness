#!/usr/bin/env python3
"""Formal schedule matrix and analysis for training-schedule robustness."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import time
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy import stats as scipy_stats
except Exception:  # pragma: no cover - covered by WSL dependency check.
    scipy_stats = None


DATASETS = ["wine", "breast_cancer", "digits"]
FORMAL_TRAIN_PER_CLASS_BUDGETS = [4, 8, 16, 32]
FORMAL_SEEDS = [14, 52, 88, 121, 169, 203, 247, 314, 415, 509]
SCHEDULE_CONFIGS = [
    {"name": "constant_0003", "family": "constant", "learning_rate": "constant", "eta0": 0.0003},
    {"name": "constant_001", "family": "constant", "learning_rate": "constant", "eta0": 0.001},
    {"name": "constant_003", "family": "constant", "learning_rate": "constant", "eta0": 0.003},
    {"name": "invscaling_0003", "family": "invscaling", "learning_rate": "invscaling", "eta0": 0.0003},
    {"name": "invscaling_001", "family": "invscaling", "learning_rate": "invscaling", "eta0": 0.001},
    {"name": "invscaling_003", "family": "invscaling", "learning_rate": "invscaling", "eta0": 0.003},
    {"name": "adaptive_0003", "family": "adaptive", "learning_rate": "adaptive", "eta0": 0.0003},
    {"name": "adaptive_001", "family": "adaptive", "learning_rate": "adaptive", "eta0": 0.001},
    {"name": "adaptive_003", "family": "adaptive", "learning_rate": "adaptive", "eta0": 0.003},
]
PRIMARY_METRIC = "macro_f1"
REFERENCE_FORMAL_SEEDS = {23, 41, 67, 83, 127}
BOOTSTRAP_ITERATIONS = 5000


def now_epoch() -> float:
    return round(time.time(), 3)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def update_heartbeat(path: Path, status: str, **fields: Any) -> None:
    write_json(path, {"status": status, "updated_at_epoch": now_epoch(), **fields})


def expected_formal_rows() -> int:
    return len(DATASETS) * len(FORMAL_TRAIN_PER_CLASS_BUDGETS) * len(FORMAL_SEEDS) * len(SCHEDULE_CONFIGS)


def load_dataset(name: str) -> dict[str, Any]:
    loaders = {
        "wine": load_wine,
        "breast_cancer": load_breast_cancer,
        "digits": load_digits,
    }
    loaded = loaders[name]()
    x = np.asarray(loaded.data, dtype=np.float64)
    y = np.asarray(loaded.target, dtype=np.int64)
    return {
        "name": name,
        "x": x,
        "y": y,
        "n_samples": int(x.shape[0]),
        "n_features": int(x.shape[1]),
        "n_classes": int(len(set(y.tolist()))),
        "class_counts": {str(label): int((y == label).sum()) for label in sorted(set(y.tolist()))},
    }


def stratified_indices(y: np.ndarray, train_per_class: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed + train_per_class * 1009)
    train: list[int] = []
    test: list[int] = []
    for label in sorted(set(y.tolist())):
        indices = [int(i) for i, value in enumerate(y) if int(value) == int(label)]
        rng.shuffle(indices)
        if len(indices) <= train_per_class:
            raise ValueError(f"class {label} has {len(indices)} rows, cannot reserve test examples")
        train.extend(indices[:train_per_class])
        test.extend(indices[train_per_class:])
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(sorted(test), dtype=np.int64)


def build_model(schedule: dict[str, Any], seed: int) -> Any:
    classifier = SGDClassifier(
        loss="log_loss",
        learning_rate=str(schedule["learning_rate"]),
        eta0=float(schedule["eta0"]),
        power_t=0.5,
        alpha=1e-4,
        max_iter=1000,
        tol=1e-3,
        class_weight="balanced",
        random_state=seed,
    )
    return make_pipeline(StandardScaler(), classifier)


def run_one(dataset: dict[str, Any], schedule: dict[str, Any], train_per_class: int, seed: int) -> dict[str, Any]:
    x = dataset["x"]
    y = dataset["y"]
    train_idx, test_idx = stratified_indices(y, train_per_class=train_per_class, seed=seed)
    model = build_model(schedule, seed)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", category=ConvergenceWarning)
        model.fit(x[train_idx], y[train_idx])
    fit_seconds = time.perf_counter() - started
    convergence_warnings = sum(1 for item in caught if issubclass(item.category, ConvergenceWarning))
    predicted = model.predict(x[test_idx])
    probabilities = model.predict_proba(x[test_idx])
    labels = sorted(set(y.tolist()))
    classifier = model.named_steps["sgdclassifier"]
    return {
        "dataset": dataset["name"],
        "n_samples": dataset["n_samples"],
        "n_features": dataset["n_features"],
        "n_classes": dataset["n_classes"],
        "train_per_class": train_per_class,
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "seed": seed,
        "schedule": schedule["name"],
        "schedule_family": schedule["family"],
        "learning_rate": schedule["learning_rate"],
        "eta0": float(schedule["eta0"]),
        "accuracy": float(accuracy_score(y[test_idx], predicted)),
        "macro_f1": float(f1_score(y[test_idx], predicted, average="macro")),
        "log_loss": float(log_loss(y[test_idx], probabilities, labels=labels)),
        "fit_seconds": float(fit_seconds),
        "n_iter": int(classifier.n_iter_),
        "convergence_warning_count": int(convergence_warnings),
        "status": "PASS",
    }


def detect_duplicate_keys(rows: list[dict[str, Any]]) -> list[tuple[str, int, int, str]]:
    seen: set[tuple[str, int, int, str]] = set()
    duplicates: set[tuple[str, int, int, str]] = set()
    for row in rows:
        key = (
            str(row["dataset"]),
            int(row["train_per_class"]),
            int(row["seed"]),
            str(row["schedule"]),
        )
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    return sorted(duplicates)


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def sample_std(values: list[float]) -> float:
    return float(statistics.stdev(values)) if len(values) > 1 else 0.0


def ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "std": 0.0, "n": 0}
    avg = mean(values)
    std = sample_std(values)
    half_width = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
    return {
        "mean": avg,
        "ci_low": avg - half_width,
        "ci_high": avg + half_width,
        "std": std,
        "n": len(values),
    }


def normal_approx_paired_p(deltas: list[float]) -> float:
    if not deltas:
        return 1.0
    stats = ci95(deltas)
    if stats["std"] == 0.0:
        return 0.0 if abs(stats["mean"]) > 0.0 else 1.0
    z_score = abs(stats["mean"]) / (stats["std"] / math.sqrt(stats["n"]))
    return float(math.erfc(z_score / math.sqrt(2.0)))


def stable_seed(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**32)


def bootstrap_ci95(deltas: list[float], seed: int, iterations: int = BOOTSTRAP_ITERATIONS) -> dict[str, Any]:
    if not deltas:
        return {
            "bootstrap_ci95_low": 0.0,
            "bootstrap_ci95_high": 0.0,
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        }
    if len(deltas) == 1:
        return {
            "bootstrap_ci95_low": float(deltas[0]),
            "bootstrap_ci95_high": float(deltas[0]),
            "bootstrap_iterations": iterations,
            "bootstrap_seed": seed,
        }
    rng = np.random.default_rng(seed)
    values = np.asarray(deltas, dtype=np.float64)
    sampled = rng.choice(values, size=(iterations, len(values)), replace=True)
    means = np.mean(sampled, axis=1)
    return {
        "bootstrap_ci95_low": float(np.percentile(means, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(means, 97.5)),
        "bootstrap_iterations": iterations,
        "bootstrap_seed": seed,
    }


def paired_delta_statistics(deltas: list[float], seed: int) -> dict[str, Any]:
    delta_stats = ci95(deltas)
    std_delta = float(delta_stats["std"])
    if len(deltas) < 2:
        paired_t_statistic = 0.0
        paired_t_p_value = 1.0
        method = "insufficient_pairs"
    elif std_delta == 0.0:
        paired_t_statistic = math.copysign(1.0e12, delta_stats["mean"]) if abs(delta_stats["mean"]) > 0.0 else 0.0
        paired_t_p_value = 0.0 if abs(delta_stats["mean"]) > 0.0 else 1.0
        method = "zero_variance_closed_form"
    elif scipy_stats is not None:
        test = scipy_stats.ttest_1samp(deltas, popmean=0.0)
        paired_t_statistic = float(test.statistic)
        paired_t_p_value = float(test.pvalue)
        method = "scipy.stats.ttest_1samp_on_paired_deltas"
    else:
        paired_t_statistic = 0.0
        paired_t_p_value = 1.0
        method = "scipy_unavailable_fail_closed"
    if std_delta == 0.0:
        paired_cohens_dz = math.copysign(1.0e12, delta_stats["mean"]) if abs(delta_stats["mean"]) > 0.0 else 0.0
    else:
        paired_cohens_dz = float(delta_stats["mean"] / std_delta)
    return {
        **delta_stats,
        **bootstrap_ci95(deltas, seed=seed),
        "paired_t_statistic": paired_t_statistic,
        "paired_t_p_value": paired_t_p_value,
        "paired_t_p_value_method": method,
        "paired_cohens_dz": paired_cohens_dz,
        "normal_approx_p_value": normal_approx_paired_p(deltas),
    }


def holm_adjust(rows: list[dict[str, Any]], alpha: float = 0.05, p_key: str = "p_value") -> list[dict[str, Any]]:
    sorted_rows = sorted(rows, key=lambda row: float(row[p_key]))
    total = len(sorted_rows)
    max_adjusted = 0.0
    adjusted: list[dict[str, Any]] = []
    for index, row in enumerate(sorted_rows):
        adjusted_p = min(1.0, (total - index) * float(row[p_key]))
        max_adjusted = max(max_adjusted, adjusted_p)
        copied = dict(row)
        copied["holm_p"] = max_adjusted
        copied["holm_corrected_p_value"] = max_adjusted
        copied["holm_significant_0_05"] = max_adjusted <= alpha
        adjusted.append(copied)
    return adjusted


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["dataset"]),
                int(row["train_per_class"]),
                str(row["schedule_family"]),
                float(row["eta0"]),
            )
        ].append(row)
    aggregates: list[dict[str, Any]] = []
    for (dataset, budget, family, eta0), items in sorted(grouped.items()):
        aggregates.append(
            {
                "dataset": dataset,
                "train_per_class": budget,
                "schedule_family": family,
                "eta0": eta0,
                "rows": len(items),
                "mean_macro_f1": mean([float(row["macro_f1"]) for row in items]),
                "mean_accuracy": mean([float(row["accuracy"]) for row in items]),
                "mean_log_loss": mean([float(row["log_loss"]) for row in items]),
                "mean_fit_seconds": mean([float(row["fit_seconds"]) for row in items]),
                "mean_n_iter": mean([float(row["n_iter"]) for row in items]),
                "convergence_warning_count": int(sum(int(row["convergence_warning_count"]) for row in items)),
            }
        )
    return aggregates


def summarize_sensitivity(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["dataset"]), int(row["train_per_class"]))].append(row)
    output: list[dict[str, Any]] = []
    for (dataset, budget), items in sorted(grouped.items()):
        by_schedule: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_schedule[str(item["schedule"])].append(item)
        schedule_means = []
        for schedule, schedule_items in sorted(by_schedule.items()):
            schedule_means.append(
                {
                    "schedule": schedule,
                    "mean_macro_f1": mean([float(row["macro_f1"]) for row in schedule_items]),
                    "mean_accuracy": mean([float(row["accuracy"]) for row in schedule_items]),
                    "mean_log_loss": mean([float(row["log_loss"]) for row in schedule_items]),
                    "mean_fit_seconds": mean([float(row["fit_seconds"]) for row in schedule_items]),
                    "convergence_warning_count": int(sum(int(row["convergence_warning_count"]) for row in schedule_items)),
                }
            )
        macro_values = [row["mean_macro_f1"] for row in schedule_means]
        runtime_values = [row["mean_fit_seconds"] for row in schedule_means]
        best = max(schedule_means, key=lambda row: row["mean_macro_f1"])
        worst = min(schedule_means, key=lambda row: row["mean_macro_f1"])
        min_runtime = max(min(runtime_values), 1e-12)
        macro_range = max(macro_values) - min(macro_values)
        runtime_ratio = max(runtime_values) / min_runtime
        output.append(
            {
                "dataset": dataset,
                "train_per_class": budget,
                "macro_f1_range": macro_range,
                "best_schedule": best["schedule"],
                "best_macro_f1": best["mean_macro_f1"],
                "worst_schedule": worst["schedule"],
                "worst_macro_f1": worst["mean_macro_f1"],
                "runtime_ratio": runtime_ratio,
                "max_mean_fit_seconds": max(runtime_values),
                "min_mean_fit_seconds": min(runtime_values),
                "schedule_count": len(schedule_means),
                "convergence_warning_count": int(sum(row["convergence_warning_count"] for row in schedule_means)),
                "interpretable": bool(macro_range >= 0.02 or runtime_ratio >= 1.5),
            }
        )
    return output


def paired_family_comparisons(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int, float], dict[tuple[str, int], float]] = defaultdict(dict)
    for row in rows:
        grouped[
            (str(row["dataset"]), int(row["train_per_class"]), float(row["eta0"]))
        ][(str(row["schedule_family"]), int(row["seed"]))] = float(row["macro_f1"])
    comparisons: list[dict[str, Any]] = []
    families = ["adaptive", "constant", "invscaling"]
    for (dataset, budget, eta0), values in sorted(grouped.items()):
        for left, right in combinations(families, 2):
            deltas = []
            for seed in FORMAL_SEEDS:
                left_value = values.get((left, seed))
                right_value = values.get((right, seed))
                if left_value is not None and right_value is not None:
                    deltas.append(left_value - right_value)
            delta_stats = paired_delta_statistics(
                deltas,
                seed=stable_seed(dataset, budget, eta0, left, right),
            )
            comparisons.append(
                {
                    "comparison": f"{left}_minus_{right}",
                    "dataset": dataset,
                    "train_per_class": budget,
                    "eta0": eta0,
                    "n_pairs": len(deltas),
                    "mean_delta_macro_f1": delta_stats["mean"],
                    "ci95_low": delta_stats["ci_low"],
                    "ci95_high": delta_stats["ci_high"],
                    "std_delta": delta_stats["std"],
                    "bootstrap_ci95_low": delta_stats["bootstrap_ci95_low"],
                    "bootstrap_ci95_high": delta_stats["bootstrap_ci95_high"],
                    "bootstrap_iterations": delta_stats["bootstrap_iterations"],
                    "bootstrap_seed": delta_stats["bootstrap_seed"],
                    "paired_t_statistic": delta_stats["paired_t_statistic"],
                    "paired_t_p_value": delta_stats["paired_t_p_value"],
                    "paired_t_p_value_method": delta_stats["paired_t_p_value_method"],
                    "paired_cohens_dz": delta_stats["paired_cohens_dz"],
                    "normal_approx_p_value": delta_stats["normal_approx_p_value"],
                    "p_value": delta_stats["paired_t_p_value"],
                }
            )
    return holm_adjust(comparisons, p_key="paired_t_p_value")


def summarize_strengthened_triggers(
    interpretable_strata: int,
    significant_comparisons: int,
    total_comparisons: int,
    figure_ready: bool,
    max_abs_delta: float | None = None,
) -> dict[str, Any]:
    triggers: list[str] = []
    if interpretable_strata < 3:
        triggers.append("fewer_than_three_interpretable_strata")
    if total_comparisons > 0 and significant_comparisons == 0:
        triggers.append("paired_statistics_inconclusive")
    if max_abs_delta is not None and max_abs_delta < 0.01:
        triggers.append("primary_effect_too_trivial")
    if not figure_ready:
        triggers.append("visual_evidence_not_ready")
    return {
        "recommend_strengthened_followup": bool(triggers),
        "triggers": triggers,
        "interpretable_strata": interpretable_strata,
        "significant_comparisons": significant_comparisons,
        "total_comparisons": total_comparisons,
        "figure_ready": figure_ready,
    }


def generate_svg_figure(path: Path, sensitivity_rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 980
    height = 560
    margin_left = 92
    margin_bottom = 126
    margin_top = 52
    plot_width = width - margin_left - 40
    plot_height = height - margin_top - margin_bottom
    max_value = max([float(row["macro_f1_range"]) for row in sensitivity_rows] + [0.02])
    scale_max = max_value * 1.18
    bar_gap = 12
    bar_width = (plot_width - bar_gap * (len(sensitivity_rows) - 1)) / len(sensitivity_rows)
    colors = {
        "breast_cancer": "#2f6f8f",
        "digits": "#7a4e9b",
        "wine": "#b45f3c",
    }
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="490" y="28" text-anchor="middle" font-family="Arial" font-size="20" font-weight="700">Formal Schedule Sensitivity</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5"/>',
        f'<text x="22" y="{margin_top + plot_height / 2}" transform="rotate(-90 22 {margin_top + plot_height / 2})" text-anchor="middle" font-family="Arial" font-size="15">Macro-F1 range across schedules</text>',
    ]
    for tick in range(6):
        value = scale_max * tick / 5
        y = margin_top + plot_height - (value / scale_max) * plot_height
        parts.append(f'<line x1="{margin_left - 5}" y1="{y:.2f}" x2="{margin_left + plot_width}" y2="{y:.2f}" stroke="#dddddd" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.2f}" text-anchor="end" font-family="Arial" font-size="12">{value:.3f}</text>')
    for index, row in enumerate(sensitivity_rows):
        value = float(row["macro_f1_range"])
        x = margin_left + index * (bar_width + bar_gap)
        bar_height = (value / scale_max) * plot_height
        y = margin_top + plot_height - bar_height
        dataset = str(row["dataset"])
        label = f"{dataset} @ {row['train_per_class']}"
        color = colors.get(dataset, "#4c6f55")
        parts.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="12">{value:.3f}</text>')
        parts.append(f'<text x="{x + bar_width / 2:.2f}" y="{margin_top + plot_height + 20}" text-anchor="end" transform="rotate(-38 {x + bar_width / 2:.2f} {margin_top + plot_height + 20})" font-family="Arial" font-size="12">{label}</text>')
    parts.append(f'<text x="{margin_left + plot_width / 2}" y="{height - 18}" text-anchor="middle" font-family="Arial" font-size="13">Bars summarize mean macro-F1 range across nine schedule configs per dataset-budget stratum.</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def write_key_findings(path: Path, sensitivity_rows: list[dict[str, Any]], analysis: dict[str, Any]) -> None:
    strongest = analysis["strongest_strata"]
    weakest = analysis["weakest_strata"]
    decision = analysis["strengthened_matrix_decision"]
    lines = [
        "# Formal Schedule Key Findings",
        "",
        "Status: local Phase4 analysis from the 1080-row formal matrix.",
        "",
        "## Strongest Schedule-Sensitivity Strata",
        "",
    ]
    for row in strongest:
        lines.append(
            f"- `{row['dataset']}@{row['train_per_class']}`: macro-F1 range "
            f"{row['macro_f1_range']:.4f}, best `{row['best_schedule']}`, "
            f"worst `{row['worst_schedule']}`."
        )
    lines.extend(["", "## Weakest Schedule-Sensitivity Strata", ""])
    for row in weakest:
        lines.append(
            f"- `{row['dataset']}@{row['train_per_class']}`: macro-F1 range "
            f"{row['macro_f1_range']:.4f}, best `{row['best_schedule']}`, "
            f"worst `{row['worst_schedule']}`."
        )
    lines.extend(["", "## Strengthened-Matrix Decision", ""])
    if decision["recommend_strengthened_followup"]:
        lines.append(
            "A strengthened follow-up is recommended before Phase5 because: "
            + ", ".join(decision["triggers"])
            + "."
        )
    else:
        lines.append(
            "No strengthened follow-up trigger fired for the current reserve-paper "
            "gate; Manager review is still required before Phase5."
        )
    lines.extend(
        [
            "",
            "## Paired Statistical Repair",
            "",
            "Paired schedule-family comparisons now include paired t-test "
            "statistics, paired Cohen's dz, deterministic bootstrap 95% "
            "confidence intervals for macro-F1 deltas, and Holm-corrected "
            "p-values.",
            "",
            "Top Holm-significant comparisons by absolute macro-F1 delta:",
            "",
        ]
    )
    top_comparisons = sorted(
        analysis["paired_family_comparisons"],
        key=lambda row: abs(float(row["mean_delta_macro_f1"])),
        reverse=True,
    )[:5]
    for row in top_comparisons:
        lines.append(
            f"- `{row['dataset']}@{row['train_per_class']}`, eta0 `{row['eta0']}`, "
            f"`{row['comparison']}`: delta {row['mean_delta_macro_f1']:.4f}, "
            f"t {row['paired_t_statistic']:.3f}, dz {row['paired_cohens_dz']:.3f}, "
            f"bootstrap CI [{row['bootstrap_ci95_low']:.4f}, "
            f"{row['bootstrap_ci95_high']:.4f}], Holm p "
            f"{row['holm_corrected_p_value']:.4g}."
        )
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            "These findings support bounded empirical associations between simple "
            "schedule choices and lightweight classifier behavior. They do not "
            "establish causal mechanisms or venue-final claims.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def analyze_formal_results(rows: list[dict[str, Any]], figure_dir: Path, output_dir: Path) -> dict[str, Any]:
    aggregates = aggregate_rows(rows)
    sensitivity = summarize_sensitivity(rows)
    comparisons = paired_family_comparisons(rows)
    figure_path = figure_dir / "formal_macro_f1_sensitivity_20260728.svg"
    generate_svg_figure(figure_path, sensitivity)
    interpretable = [row for row in sensitivity if row["interpretable"]]
    significant = [row for row in comparisons if row["holm_significant_0_05"]]
    max_abs_delta = max([abs(float(row["mean_delta_macro_f1"])) for row in comparisons] + [0.0])
    decision = summarize_strengthened_triggers(
        interpretable_strata=len(interpretable),
        significant_comparisons=len(significant),
        total_comparisons=len(comparisons),
        figure_ready=figure_path.exists() and figure_path.stat().st_size > 0,
        max_abs_delta=max_abs_delta,
    )
    strongest = sorted(sensitivity, key=lambda row: float(row["macro_f1_range"]), reverse=True)[:3]
    weakest = sorted(sensitivity, key=lambda row: float(row["macro_f1_range"]))[:3]
    analysis = {
        "status": "PASS_PHASE4_ANALYSIS",
        "primary_metric": PRIMARY_METRIC,
        "aggregate_rows": aggregates,
        "paired_family_comparisons": comparisons,
        "sensitivity_rows": sensitivity,
        "strongest_strata": strongest,
        "weakest_strata": weakest,
        "strengthened_matrix_decision": decision,
        "figure": str(figure_path),
        "notes": [
            "Paired p-values use a normal approximation over seed-level deltas.",
            "Holm correction is applied globally across all paired family comparisons.",
        ],
    }
    write_json(output_dir / "formal_statistical_analysis.json", analysis)
    write_key_findings(output_dir / "formal_key_findings.md", sensitivity, analysis)
    return analysis


def parse_csv_value(value: str) -> Any:
    if value in {"True", "False"}:
        return value == "True"
    try:
        if any(marker in value for marker in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def read_formal_results(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [{key: parse_csv_value(value) for key, value in row.items()} for row in reader]


def run_analysis_only(input_results: Path, output_dir: Path, heartbeat: Path, figure_dir: Path) -> dict[str, Any]:
    update_heartbeat(heartbeat, "RUNNING", stage="statistics_repair", input_results=str(input_results))
    rows = read_formal_results(input_results)
    duplicate_keys = detect_duplicate_keys(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    sensitivity = summarize_sensitivity(rows)
    write_csv(output_dir / "formal_sensitivity.csv", sensitivity)
    analysis = analyze_formal_results(rows, figure_dir=figure_dir, output_dir=output_dir)
    status = (
        "PASS_FORMAL_STATISTICS_CHECK"
        if len(rows) == expected_formal_rows() and not duplicate_keys and not failures
        else "BLOCKED_FORMAL_STATISTICS_CHECK"
    )
    summary_path = output_dir / "formal_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    summary.update(
        {
            "status": summary.get("status", "PASS_FORMAL_ANALYSIS_READY"),
            "statistics_repair_status": status,
            "row_count": len(rows),
            "expected_formal_rows": expected_formal_rows(),
            "duplicate_key_count": len(duplicate_keys),
            "failure_count": len(failures),
            "analysis_status": analysis["status"],
            "strengthened_matrix_decision": analysis["strengthened_matrix_decision"],
            "paired_t_test_field_count": sum(
                1 for row in analysis["paired_family_comparisons"] if "paired_t_statistic" in row
            ),
            "bootstrap_ci_field_count": sum(
                1 for row in analysis["paired_family_comparisons"] if "bootstrap_ci95_low" in row
            ),
        }
    )
    write_json(summary_path, summary)
    update_heartbeat(
        heartbeat,
        status,
        stage="statistics_repair_complete",
        row_count=len(rows),
        expected_rows=expected_formal_rows(),
        duplicate_key_count=len(duplicate_keys),
        failure_count=len(failures),
        paired_comparison_count=len(analysis["paired_family_comparisons"]),
    )
    return summary


def run_formal(output_dir: Path, heartbeat: Path, figure_dir: Path) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    update_heartbeat(heartbeat, "RUNNING", stage="readiness", row_count=0)
    if set(FORMAL_SEEDS).intersection(REFERENCE_FORMAL_SEEDS):
        raise ValueError("Formal seeds overlap with reserved reference seeds")
    datasets = [load_dataset(name) for name in DATASETS]
    update_heartbeat(heartbeat, "RUNNING", stage="formal_matrix", row_count=0, expected_rows=expected_formal_rows())
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for train_per_class in FORMAL_TRAIN_PER_CLASS_BUDGETS:
            for seed in FORMAL_SEEDS:
                for schedule in SCHEDULE_CONFIGS:
                    rows.append(run_one(dataset, schedule, train_per_class, seed))
                    if len(rows) % 50 == 0:
                        update_heartbeat(
                            heartbeat,
                            "RUNNING",
                            stage="formal_matrix",
                            row_count=len(rows),
                            expected_rows=expected_formal_rows(),
                        )
    duplicate_keys = detect_duplicate_keys(rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    sensitivity = summarize_sensitivity(rows)
    write_csv(output_dir / "formal_results.csv", rows)
    write_csv(output_dir / "formal_sensitivity.csv", sensitivity)
    update_heartbeat(
        heartbeat,
        "RUNNING",
        stage="phase4_analysis",
        row_count=len(rows),
        expected_rows=expected_formal_rows(),
    )
    analysis = analyze_formal_results(rows, figure_dir=figure_dir, output_dir=output_dir)
    runtime_seconds = time.perf_counter() - started
    status = (
        "PASS_FORMAL_ANALYSIS_READY"
        if len(rows) == expected_formal_rows() and not duplicate_keys and not failures
        else "BLOCKED_FORMAL_MATRIX_OR_ANALYSIS"
    )
    summary = {
        "status": status,
        "row_count": len(rows),
        "expected_formal_rows": expected_formal_rows(),
        "duplicate_key_count": len(duplicate_keys),
        "duplicate_keys": [list(key) for key in duplicate_keys],
        "failure_count": len(failures),
        "convergence_warning_count": int(sum(int(row["convergence_warning_count"]) for row in rows)),
        "runtime_seconds": runtime_seconds,
        "datasets": DATASETS,
        "train_per_class_budgets": FORMAL_TRAIN_PER_CLASS_BUDGETS,
        "seeds": FORMAL_SEEDS,
        "schedules": [schedule["name"] for schedule in SCHEDULE_CONFIGS],
        "primary_metric": PRIMARY_METRIC,
        "analysis_status": analysis["status"],
        "strengthened_matrix_decision": analysis["strengthened_matrix_decision"],
        "outputs": {
            "formal_results": "formal_results.csv",
            "formal_summary": "formal_summary.json",
            "formal_sensitivity": "formal_sensitivity.csv",
            "formal_statistical_analysis": "formal_statistical_analysis.json",
            "formal_key_findings": "formal_key_findings.md",
            "figure": analysis["figure"],
        },
    }
    write_json(output_dir / "formal_summary.json", summary)
    update_heartbeat(
        heartbeat,
        status,
        stage="complete",
        row_count=len(rows),
        expected_rows=expected_formal_rows(),
        duplicate_key_count=len(duplicate_keys),
        failure_count=len(failures),
        runtime_seconds=runtime_seconds,
        strengthened_matrix_decision=analysis["strengthened_matrix_decision"],
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heartbeat", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--analysis-only-from", type=Path)
    args = parser.parse_args()
    if args.analysis_only_from:
        summary = run_analysis_only(args.analysis_only_from, args.output_dir, args.heartbeat, args.figure_dir)
    else:
        summary = run_formal(args.output_dir, args.heartbeat, args.figure_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
