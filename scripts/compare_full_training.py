"""Compare two controlled full generative training runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


def _load_history(path: Path) -> tuple[list[dict[str, float]], str]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty history: {path}")

    validation_columns = [
        name for name in rows[0]
        if name.startswith("validation_")
    ]
    if len(validation_columns) != 1:
        raise ValueError(
            f"expected exactly one validation metric in {path}, got {validation_columns}"
        )
    validation_column = validation_columns[0]
    metric = validation_column.removeprefix("validation_")
    train_column = f"train_{metric}"
    required = {"epoch", train_column, validation_column, "epoch_compute_seconds"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"history {path} is missing columns: {sorted(missing)}")

    parsed: list[dict[str, float]] = []
    for row in rows:
        parsed.append(
            {
                "epoch": float(row["epoch"]),
                "train": float(row[train_column]),
                "validation": float(row[validation_column]),
                "epoch_compute_seconds": float(row["epoch_compute_seconds"]),
            }
        )
    return parsed, metric


def _load_standardizers(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping standardizers artifact: {path}")
    return payload


def _assert_same_standardizers(left: dict[str, Any], right: dict[str, Any]) -> None:
    for key in ("weighting", "physical_nondimensionalization", "train_sample_ids"):
        if left[key] != right[key]:
            raise ValueError(f"standardizer metadata differs at {key}")

    for group in ("inputs", "targets"):
        if tuple(left[group]["channel_names"]) != tuple(right[group]["channel_names"]):
            raise ValueError(f"{group} standardizer channels differ")
        torch.testing.assert_close(
            left[group]["mean"],
            right[group]["mean"],
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            left[group]["scale"],
            right[group]["scale"],
            rtol=0.0,
            atol=0.0,
        )


def _normalized_config(path: Path) -> dict[str, Any]:
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(cfg, dict):
        raise TypeError(f"expected mapping config: {path}")
    cfg = dict(cfg)
    cfg.pop("run_name", None)
    model = dict(cfg["model"])
    model["sparse_attention_backend"] = "__controlled_backend__"
    cfg["model"] = model
    return cfg


def _ratio(contender: float, baseline: float) -> dict[str, float]:
    ratio = contender / baseline
    return {
        "baseline": baseline,
        "contender": contender,
        "contender_over_baseline": ratio,
        "change_percent": (ratio - 1.0) * 100.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scatter-run", type=Path, required=True)
    parser.add_argument("--optimized-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    scatter_summary = json.loads((args.scatter_run / "summary.json").read_text())
    optimized_summary = json.loads((args.optimized_run / "summary.json").read_text())

    scatter_manifest = json.loads(
        (args.scatter_run / "dataset_split_manifest.json").read_text()
    )
    optimized_manifest = json.loads(
        (args.optimized_run / "dataset_split_manifest.json").read_text()
    )
    for key in ("train_ids", "validation_ids", "test_ids"):
        if scatter_manifest.get(key) != optimized_manifest.get(key):
            raise ValueError(f"dataset split differs at {key}")

    _assert_same_standardizers(
        _load_standardizers(args.scatter_run / "standardizers.pt"),
        _load_standardizers(args.optimized_run / "standardizers.pt"),
    )
    if _normalized_config(args.scatter_run / "resolved_config.yaml") != _normalized_config(
        args.optimized_run / "resolved_config.yaml"
    ):
        raise ValueError(
            "resolved configurations differ beyond run_name and sparse_attention_backend"
        )

    scatter_history, scatter_metric = _load_history(args.scatter_run / "history.csv")
    optimized_history, optimized_metric = _load_history(args.optimized_run / "history.csv")
    if scatter_metric != optimized_metric:
        raise ValueError(
            f"training metrics differ: {scatter_metric} != {optimized_metric}"
        )
    if len(scatter_history) != len(optimized_history):
        raise ValueError("training histories have different epoch counts")

    metric = scatter_metric
    scatter_train = np.asarray([row["train"] for row in scatter_history])
    optimized_train = np.asarray([row["train"] for row in optimized_history])
    scatter_validation = np.asarray([row["validation"] for row in scatter_history])
    optimized_validation = np.asarray([row["validation"] for row in optimized_history])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = args.output_dir / "epoch_comparison.csv"
    with trajectory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                f"scatter_train_{metric}",
                f"optimized_train_{metric}",
                "train_difference",
                f"scatter_validation_{metric}",
                f"optimized_validation_{metric}",
                "validation_difference",
                "scatter_epoch_compute_seconds",
                "optimized_epoch_compute_seconds",
            ),
        )
        writer.writeheader()
        for scatter_row, optimized_row in zip(
            scatter_history,
            optimized_history,
            strict=True,
        ):
            writer.writerow(
                {
                    "epoch": int(scatter_row["epoch"]),
                    f"scatter_train_{metric}": scatter_row["train"],
                    f"optimized_train_{metric}": optimized_row["train"],
                    "train_difference": optimized_row["train"] - scatter_row["train"],
                    f"scatter_validation_{metric}": scatter_row["validation"],
                    f"optimized_validation_{metric}": optimized_row["validation"],
                    "validation_difference": (
                        optimized_row["validation"] - scatter_row["validation"]
                    ),
                    "scatter_epoch_compute_seconds": scatter_row[
                        "epoch_compute_seconds"
                    ],
                    "optimized_epoch_compute_seconds": optimized_row[
                        "epoch_compute_seconds"
                    ],
                }
            )

    scatter_best = float(scatter_summary[f"best_validation_{metric}"])
    optimized_best = float(optimized_summary[f"best_validation_{metric}"])
    scatter_test = float(scatter_summary[f"test_{metric}_at_best_validation"])
    optimized_test = float(optimized_summary[f"test_{metric}_at_best_validation"])

    payload = {
        "comparison": "m34_dinat_dit_scatter_vs_optimized_full_training",
        "metric": metric,
        "controlled_contract": {
            "same_resolved_config_except_backend_and_run_name": True,
            "same_train_validation_test_split": True,
            "same_training_standardizers": True,
            "seed": scatter_summary["seed"],
            "epochs": len(scatter_history),
        },
        "scatter": {
            "run_name": scatter_summary["run_name"],
            "best_epoch": int(scatter_summary["best_epoch"]),
            "best_validation": scatter_best,
            "test_at_best_validation": scatter_test,
            "fit_wall_seconds": float(scatter_summary["fit_wall_seconds"]),
            "mean_epoch_compute_seconds": float(
                scatter_summary["mean_epoch_compute_seconds"]
            ),
            "fit_peak_allocated_gib": float(
                scatter_summary["fit_peak_allocated_gib"]
            ),
            "fit_peak_reserved_gib": float(scatter_summary["fit_peak_reserved_gib"]),
            "final_train": float(scatter_train[-1]),
            "final_validation": float(scatter_validation[-1]),
        },
        "optimized": {
            "run_name": optimized_summary["run_name"],
            "best_epoch": int(optimized_summary["best_epoch"]),
            "best_validation": optimized_best,
            "test_at_best_validation": optimized_test,
            "fit_wall_seconds": float(optimized_summary["fit_wall_seconds"]),
            "mean_epoch_compute_seconds": float(
                optimized_summary["mean_epoch_compute_seconds"]
            ),
            "fit_peak_allocated_gib": float(
                optimized_summary["fit_peak_allocated_gib"]
            ),
            "fit_peak_reserved_gib": float(
                optimized_summary["fit_peak_reserved_gib"]
            ),
            "final_train": float(optimized_train[-1]),
            "final_validation": float(optimized_validation[-1]),
        },
        "optimized_relative_to_scatter": {
            "fit_wall_time": _ratio(
                float(optimized_summary["fit_wall_seconds"]),
                float(scatter_summary["fit_wall_seconds"]),
            ),
            "mean_epoch_compute_time": _ratio(
                float(optimized_summary["mean_epoch_compute_seconds"]),
                float(scatter_summary["mean_epoch_compute_seconds"]),
            ),
            "fit_peak_allocated_memory": _ratio(
                float(optimized_summary["fit_peak_allocated_gib"]),
                float(scatter_summary["fit_peak_allocated_gib"]),
            ),
            "best_validation": _ratio(optimized_best, scatter_best),
            "test_at_best_validation": _ratio(optimized_test, scatter_test),
        },
        "trajectory_difference": {
            "train_rmse": float(
                np.sqrt(np.mean((optimized_train - scatter_train) ** 2))
            ),
            "train_max_abs": float(np.max(np.abs(optimized_train - scatter_train))),
            "validation_rmse": float(
                np.sqrt(
                    np.mean((optimized_validation - scatter_validation) ** 2)
                )
            ),
            "validation_max_abs": float(
                np.max(np.abs(optimized_validation - scatter_validation))
            ),
        },
        "outputs": {
            "epoch_comparison": trajectory_path.name,
        },
    }

    output_json = args.output_dir / "full_training_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")

    print(json.dumps(payload, indent=2))
    print()
    print(f"JSON: {output_json}")
    print(f"CSV:  {trajectory_path}")


if __name__ == "__main__":
    main()
