#!/usr/bin/env python3
"""Revision experiment: condition numbers for low-budget digits matrices."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler


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
from run_formal_schedule_matrix import FORMAL_SEEDS, load_dataset, stratified_indices, write_csv, write_json  # noqa: E402


BUDGETS = [4, 8, 16, 32]


def finite_condition_number(x_scaled: np.ndarray) -> float:
    singular = np.linalg.svd(x_scaled, compute_uv=False)
    if singular.size == 0:
        return 0.0
    max_s = float(np.max(singular))
    min_s = float(np.min(singular))
    floor = np.finfo(np.float64).eps
    return max_s / max(min_s, floor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset("digits")
    rows: list[dict[str, Any]] = []
    for budget in BUDGETS:
        for seed in FORMAL_SEEDS:
            train_idx, _ = stratified_indices(dataset["y"], train_per_class=budget, seed=seed)
            x_scaled = StandardScaler().fit_transform(dataset["x"][train_idx])
            rank = int(np.linalg.matrix_rank(x_scaled))
            condition = finite_condition_number(x_scaled)
            rows.append(
                {
                    "dataset": "digits",
                    "budget": budget,
                    "seed": seed,
                    "n_rows": int(x_scaled.shape[0]),
                    "n_features": int(x_scaled.shape[1]),
                    "matrix_rank": rank,
                    "condition_number": f"{condition:.2f}",
                }
            )
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["budget"])].append(row)
    summary = []
    for budget in sorted(grouped):
        values = [float(row["condition_number"]) for row in grouped[budget]]
        ranks = [int(row["matrix_rank"]) for row in grouped[budget]]
        n_features = int(grouped[budget][0]["n_features"])
        summary.append(
            {
                "budget": budget,
                "dataset": "digits",
                "max_rank": max(ranks),
                "mean_condition_number": float(f"{float(np.mean(values)):.2f}"),
                "min_rank": min(ranks),
                "n_seeds": len(values),
                "std_condition_number": float(f"{float(np.std(values, ddof=1)):.2f}"),
                "underdetermined": bool(any(rank < n_features for rank in ranks)),
            }
        )
    write_csv(args.output_dir / "condition_number_analysis.csv", rows)
    write_json(args.output_dir / "condition_number_summary.json", summary)
    print(json.dumps({"script": Path(__file__).name, "status": "PASS", "rows": len(rows), "output_dir": str(args.output_dir)}, sort_keys=True))


if __name__ == "__main__":
    main()
