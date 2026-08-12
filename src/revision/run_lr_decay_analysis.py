#!/usr/bin/env python3
"""Revision experiment: trace learning-rate decay, loss, and macro-F1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


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
from run_formal_schedule_matrix import load_dataset, stratified_indices, write_csv, write_json  # noqa: E402


DATASETS = ["wine", "breast_cancer", "digits"]
BUDGETS = [4, 32]
SEEDS = [14, 52, 88]
SCHEDULES = [
    {"name": "constant_001", "learning_rate": "constant", "eta0": 0.001},
    {"name": "invscaling_001", "learning_rate": "invscaling", "eta0": 0.001},
    {"name": "adaptive_001", "learning_rate": "adaptive", "eta0": 0.001},
]
POWER_T = 0.5


def rounded(value: float, digits: int) -> float:
    return float(f"{value:.{digits}f}")


def effective_lr(schedule: dict[str, Any], epoch: int) -> float:
    if schedule["learning_rate"] == "invscaling":
        return float(schedule["eta0"]) / (float(epoch) ** POWER_T)
    return float(schedule["eta0"])


def train_trace(dataset_name: str, budget: int, schedule: dict[str, Any], seed: int, epochs: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset = load_dataset(dataset_name)
    x = dataset["x"]
    y = dataset["y"]
    train_idx, test_idx = stratified_indices(y, train_per_class=budget, seed=seed)
    x_train = x[train_idx]
    y_train = y[train_idx]
    x_test = x[test_idx]
    y_test = y[test_idx]
    classes = np.asarray(sorted(set(y.tolist())), dtype=np.int64)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    weight_by_class = {int(label): float(weight) for label, weight in zip(classes, weights)}
    sample_weight = np.asarray([weight_by_class[int(label)] for label in y_train], dtype=np.float64)

    model = SGDClassifier(
        loss="log_loss",
        learning_rate=str(schedule["learning_rate"]),
        eta0=float(schedule["eta0"]),
        power_t=POWER_T,
        alpha=1e-4,
        max_iter=1,
        tol=None,
        random_state=seed,
    )
    rows: list[dict[str, Any]] = []
    losses: list[float] = []
    for epoch in range(1, epochs + 1):
        model.partial_fit(x_train_scaled, y_train, classes=classes, sample_weight=sample_weight)
        train_prob = model.predict_proba(x_train_scaled)
        train_loss = float(log_loss(y_train, train_prob, labels=classes))
        test_pred = model.predict(x_test_scaled)
        test_f1 = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
        losses.append(train_loss)
        rows.append(
            {
                "dataset": dataset_name,
                "budget": budget,
                "schedule": schedule["name"],
                "seed": seed,
                "epoch": epoch,
                "effective_lr": f"{effective_lr(schedule, epoch):.8f}",
                "train_loss": f"{train_loss:.6f}",
                "test_macro_f1": f"{test_f1:.6f}",
            }
        )
    window = losses[-20:] if len(losses) >= 20 else losses
    converged = bool(window and (max(window) - min(window) < 1.0e-4))
    summary = {
        "budget": budget,
        "converged": converged,
        "dataset": dataset_name,
        "final_epoch": epochs,
        "final_loss": rounded(losses[-1], 6),
        "final_lr": rounded(effective_lr(schedule, epochs), 8),
        "final_test_f1": rounded(float(rows[-1]["test_macro_f1"]), 6),
        "schedule": schedule["name"],
        "seed": seed,
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for budget in BUDGETS:
            for schedule in SCHEDULES:
                for seed in SEEDS:
                    rows, summary = train_trace(dataset, budget, schedule, seed, args.epochs)
                    trajectory_rows.extend(rows)
                    summary_rows.append(summary)
    write_csv(args.output_dir / "lr_decay_trajectories.csv", trajectory_rows)
    write_json(args.output_dir / "lr_decay_summary.json", summary_rows)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(trajectory_rows), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
