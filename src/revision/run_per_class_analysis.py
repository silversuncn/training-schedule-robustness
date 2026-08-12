#!/usr/bin/env python3
"""Revision experiment: per-class wine F1 for schedule-family diagnosis."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report


def _add_main_source_path() -> None:
    here = Path(__file__).resolve()
    candidates = [
        here.parent,
        here.parent.parent / "01_experiment" / "source_code",
        here.parent.parent.parent / "01_experiment" / "source_code",
    ]
    for candidate in candidates:
        if (candidate / "run_formal_schedule_matrix.py").exists():
            sys.path.insert(0, str(candidate))
            return
    raise RuntimeError("run_formal_schedule_matrix.py not found")


_add_main_source_path()
from run_formal_schedule_matrix import (  # noqa: E402
    FORMAL_SEEDS,
    FORMAL_TRAIN_PER_CLASS_BUDGETS,
    SCHEDULE_CONFIGS,
    build_model,
    load_dataset,
    stratified_indices,
    write_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("wine")
    x = dataset["x"]
    y = dataset["y"]
    class_labels = [int(label) for label in sorted(set(y.tolist()))]
    rows: list[dict[str, Any]] = []
    for budget in FORMAL_TRAIN_PER_CLASS_BUDGETS:
        for seed in FORMAL_SEEDS:
            train_idx, test_idx = stratified_indices(y, train_per_class=budget, seed=seed)
            for schedule in SCHEDULE_CONFIGS:
                model = build_model(schedule, seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=ConvergenceWarning)
                    model.fit(x[train_idx], y[train_idx])
                predicted = model.predict(x[test_idx])
                report = classification_report(y[test_idx], predicted, labels=class_labels, output_dict=True, zero_division=0)
                for label in class_labels:
                    stats = report[str(label)]
                    rows.append(
                        {
                            "budget": budget,
                            "seed": seed,
                            "schedule": schedule["name"],
                            "schedule_family": schedule["family"],
                            "class_label": label,
                            "precision": f"{float(stats['precision']):.4f}",
                            "recall": f"{float(stats['recall']):.4f}",
                            "f1": f"{float(stats['f1-score']):.4f}",
                            "support": int(stats["support"]),
                        }
                    )
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["schedule_family"]), int(row["class_label"]))].append(row)
    summary_rows = []
    for family, label in sorted(grouped):
        items = grouped[(family, label)]
        summary_rows.append(
            {
                "class_label": label,
                "mean_f1": float(f"{float(np.mean([float(row['f1']) for row in items])):.4f}"),
                "mean_precision": float(f"{float(np.mean([float(row['precision']) for row in items])):.4f}"),
                "mean_recall": float(f"{float(np.mean([float(row['recall']) for row in items])):.4f}"),
                "schedule_family": family,
                "std_f1": float(f"{float(np.std([float(row['f1']) for row in items], ddof=1)):.4f}"),
            }
        )
    family_label_mean = {(row["schedule_family"], row["class_label"]): row["mean_f1"] for row in summary_rows}
    inv_degradation = {
        label: min(
            family_label_mean.get(("invscaling", label), 0.0) - family_label_mean.get(("constant", label), 0.0),
            family_label_mean.get(("invscaling", label), 0.0) - family_label_mean.get(("adaptive", label), 0.0),
        )
        for label in class_labels
    }
    payload = {
        "summary": summary_rows,
        "worst_class": {
            "class_label": int(min(inv_degradation, key=inv_degradation.get)),
            "invscaling_min_delta_vs_other_families": float(f"{min(inv_degradation.values()):.4f}"),
        },
    }
    write_csv(args.output_dir / "wine_per_class_f1.csv", rows)
    write_json(args.output_dir / "wine_per_class_summary.json", payload)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
