"""Controlled training comparison for the M33 no-self vs M39 self-attention ablation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


_ALLOWED_CONFIG_DIFFERENCES = {
    "run_name",
    "model.include_self_attention",
}


def _load_history(path: Path) -> tuple[list[dict[str, float | None]], str]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty history: {path}")

    validation_columns = [name for name in rows[0] if name.startswith("validation_")]
    if len(validation_columns) != 1:
        raise ValueError(
            f"expected exactly one validation metric in {path}, got {validation_columns}"
        )
    validation_column = validation_columns[0]
    metric = validation_column.removeprefix("validation_")
    train_column = f"train_{metric}"
    required = {"epoch", train_column, validation_column}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"history {path} is missing columns: {sorted(missing)}")

    # Pre-M34 histories did not persist per-epoch timings. Keep the timing
    # unknown rather than fabricating it from an aggregate run summary.
    has_epoch_timing = "epoch_compute_seconds" in rows[0]
    parsed: list[dict[str, float | None]] = []
    for row in rows:
        raw_timing = row.get("epoch_compute_seconds") if has_epoch_timing else None
        parsed.append(
            {
                "epoch": float(row["epoch"]),
                "train": float(row[train_column]),
                "validation": float(row[validation_column]),
                "epoch_compute_seconds": (
                    float(raw_timing) if raw_timing not in (None, "") else None
                ),
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


def _to_plain_config(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping config: {path}")

    # Historical DiNAT runs predate these explicit config fields. Their runtime
    # defaults were scatter attention with no self token, so normalize missing
    # fields to those values before enforcing the controlled-ablation contract.
    model = payload.get("model")
    if not isinstance(model, dict):
        raise TypeError(f"expected model mapping in config: {path}")
    model.setdefault("sparse_attention_backend", "scatter")
    model.setdefault("include_self_attention", False)
    return payload


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flatten(value[key], path))
        return result
    return {prefix: value}


def _config_differences(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    left = _flatten(baseline)
    right = _flatten(candidate)
    differences: dict[str, dict[str, Any]] = {}
    for path in sorted(set(left).union(right)):
        left_value = left.get(path, "__MISSING__")
        right_value = right.get(path, "__MISSING__")
        if left_value != right_value:
            differences[path] = {
                "baseline": left_value,
                "candidate": right_value,
            }
    return differences


def _ratio(candidate: float, baseline: float) -> dict[str, float]:
    ratio = candidate / baseline
    return {
        "baseline": baseline,
        "candidate": candidate,
        "candidate_over_baseline": ratio,
        "change_percent": (ratio - 1.0) * 100.0,
    }


def _optional_float(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return float(value) if value is not None else None


def _optional_ratio(
    candidate: float | None,
    baseline: float | None,
) -> dict[str, float] | None:
    if candidate is None or baseline is None or baseline == 0.0:
        return None
    return _ratio(candidate, baseline)


def _plot_training(
    baseline_history: list[dict[str, float | None]],
    candidate_history: list[dict[str, float | None]],
    output_path: Path,
    metric: str,
) -> None:
    try:
        import matplotlib
    except ImportError as exc:
        raise RuntimeError("training comparison plotting requires matplotlib") from exc
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in baseline_history]
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 8.0), sharex=True)

    axes[0].plot(
        epochs,
        [row["train"] for row in baseline_history],
        linewidth=2.5,
        label="M33 no self",
    )
    axes[0].plot(
        epochs,
        [row["train"] for row in candidate_history],
        linewidth=2.5,
        label="M39 self",
    )
    axes[0].set_ylabel(f"Train {metric}")
    axes[0].set_yscale("log")
    axes[0].legend()

    axes[1].plot(
        epochs,
        [row["validation"] for row in baseline_history],
        linewidth=2.5,
        label="M33 no self",
    )
    axes[1].plot(
        epochs,
        [row["validation"] for row in candidate_history],
        linewidth=2.5,
        label="M39 self",
    )
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(f"Validation {metric}")
    axes[1].set_yscale("log")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    for label, run_dir in (
        ("baseline", args.baseline_run),
        ("candidate", args.candidate_run),
    ):
        if not run_dir.is_dir():
            raise NotADirectoryError(f"{label} run directory does not exist: {run_dir}")
        for name in (
            "summary.json",
            "resolved_config.yaml",
            "dataset_split_manifest.json",
            "standardizers.pt",
            "history.csv",
            "best.pt",
        ):
            if not (run_dir / name).is_file():
                raise FileNotFoundError(f"{label} is missing required artifact: {run_dir / name}")

    baseline_summary = json.loads((args.baseline_run / "summary.json").read_text())
    candidate_summary = json.loads((args.candidate_run / "summary.json").read_text())

    baseline_manifest = json.loads(
        (args.baseline_run / "dataset_split_manifest.json").read_text()
    )
    candidate_manifest = json.loads(
        (args.candidate_run / "dataset_split_manifest.json").read_text()
    )
    for key in ("train_ids", "validation_ids", "test_ids"):
        if baseline_manifest.get(key) != candidate_manifest.get(key):
            raise ValueError(f"dataset split differs at {key}")

    _assert_same_standardizers(
        _load_standardizers(args.baseline_run / "standardizers.pt"),
        _load_standardizers(args.candidate_run / "standardizers.pt"),
    )

    baseline_cfg = _to_plain_config(args.baseline_run / "resolved_config.yaml")
    candidate_cfg = _to_plain_config(args.candidate_run / "resolved_config.yaml")
    baseline_self = bool(
        baseline_cfg.get("model", {}).get("include_self_attention", False)
    )
    candidate_self = bool(
        candidate_cfg.get("model", {}).get("include_self_attention", False)
    )
    if baseline_self:
        raise ValueError("baseline run must have include_self_attention=false")
    if not candidate_self:
        raise ValueError("candidate run must have include_self_attention=true")

    differences = _config_differences(baseline_cfg, candidate_cfg)
    unexpected = {
        path: value
        for path, value in differences.items()
        if path not in _ALLOWED_CONFIG_DIFFERENCES
    }
    if unexpected:
        raise ValueError(
            "M33/M39 resolved configs differ beyond run_name and "
            f"model.include_self_attention: {json.dumps(unexpected, indent=2)}"
        )

    baseline_history, baseline_metric = _load_history(args.baseline_run / "history.csv")
    candidate_history, candidate_metric = _load_history(args.candidate_run / "history.csv")
    if baseline_metric != candidate_metric:
        raise ValueError(
            f"training metrics differ: {baseline_metric} != {candidate_metric}"
        )
    if len(baseline_history) != len(candidate_history):
        raise ValueError("training histories have different epoch counts")
    if [row["epoch"] for row in baseline_history] != [
        row["epoch"] for row in candidate_history
    ]:
        raise ValueError("training histories use different epoch indices")

    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"output directory already exists: {output_dir}; use --overwrite"
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    trajectory_path = output_dir / "training_epoch_comparison.csv"
    with trajectory_path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "epoch",
                f"m33_train_{baseline_metric}",
                f"m39_train_{baseline_metric}",
                "train_difference_m39_minus_m33",
                f"m33_validation_{baseline_metric}",
                f"m39_validation_{baseline_metric}",
                "validation_difference_m39_minus_m33",
                "m33_epoch_compute_seconds",
                "m39_epoch_compute_seconds",
            ),
        )
        writer.writeheader()
        for left, right in zip(baseline_history, candidate_history, strict=True):
            writer.writerow(
                {
                    "epoch": int(left["epoch"]),
                    f"m33_train_{baseline_metric}": left["train"],
                    f"m39_train_{baseline_metric}": right["train"],
                    "train_difference_m39_minus_m33": right["train"] - left["train"],
                    f"m33_validation_{baseline_metric}": left["validation"],
                    f"m39_validation_{baseline_metric}": right["validation"],
                    "validation_difference_m39_minus_m33": (
                        right["validation"] - left["validation"]
                    ),
                    "m33_epoch_compute_seconds": left["epoch_compute_seconds"],
                    "m39_epoch_compute_seconds": right["epoch_compute_seconds"],
                }
            )

    _plot_training(
        baseline_history,
        candidate_history,
        output_dir / "training_curves.png",
        baseline_metric,
    )

    baseline_train = np.asarray([row["train"] for row in baseline_history])
    candidate_train = np.asarray([row["train"] for row in candidate_history])
    baseline_val = np.asarray([row["validation"] for row in baseline_history])
    candidate_val = np.asarray([row["validation"] for row in candidate_history])

    metric = baseline_metric
    baseline_best = float(baseline_summary[f"best_validation_{metric}"])
    candidate_best = float(candidate_summary[f"best_validation_{metric}"])
    baseline_test = float(baseline_summary[f"test_{metric}_at_best_validation"])
    candidate_test = float(candidate_summary[f"test_{metric}_at_best_validation"])

    payload = {
        "comparison": "m33_no_self_vs_m39_self_attention",
        "metric": metric,
        "controlled_contract": {
            "only_resolved_config_differences": sorted(differences),
            "baseline_has_per_epoch_timing": baseline_history[0]["epoch_compute_seconds"] is not None,
            "candidate_has_per_epoch_timing": candidate_history[0]["epoch_compute_seconds"] is not None,
            "baseline_include_self_attention": baseline_self,
            "candidate_include_self_attention": candidate_self,
            "same_train_validation_test_split": True,
            "same_training_standardizers": True,
            "epochs": len(baseline_history),
            "seed": baseline_summary["seed"],
        },
        "m33_no_self": {
            "run_name": baseline_summary["run_name"],
            "best_epoch": int(baseline_summary["best_epoch"]),
            "best_validation": baseline_best,
            "test_at_best_validation": baseline_test,
            "fit_wall_seconds": _optional_float(baseline_summary, "fit_wall_seconds"),
            "mean_epoch_compute_seconds": _optional_float(
                baseline_summary, "mean_epoch_compute_seconds"
            ),
            "fit_peak_allocated_gib": _optional_float(
                baseline_summary, "fit_peak_allocated_gib"
            ),
            "fit_peak_reserved_gib": _optional_float(
                baseline_summary, "fit_peak_reserved_gib"
            ),
            "final_train": float(baseline_train[-1]),
            "final_validation": float(baseline_val[-1]),
        },
        "m39_self": {
            "run_name": candidate_summary["run_name"],
            "best_epoch": int(candidate_summary["best_epoch"]),
            "best_validation": candidate_best,
            "test_at_best_validation": candidate_test,
            "fit_wall_seconds": _optional_float(candidate_summary, "fit_wall_seconds"),
            "mean_epoch_compute_seconds": _optional_float(
                candidate_summary, "mean_epoch_compute_seconds"
            ),
            "fit_peak_allocated_gib": _optional_float(
                candidate_summary, "fit_peak_allocated_gib"
            ),
            "fit_peak_reserved_gib": _optional_float(
                candidate_summary, "fit_peak_reserved_gib"
            ),
            "final_train": float(candidate_train[-1]),
            "final_validation": float(candidate_val[-1]),
        },
        "m39_relative_to_m33": {
            "best_validation": _ratio(candidate_best, baseline_best),
            "test_at_best_validation": _ratio(candidate_test, baseline_test),
            "fit_wall_time": _optional_ratio(
                _optional_float(candidate_summary, "fit_wall_seconds"),
                _optional_float(baseline_summary, "fit_wall_seconds"),
            ),
            "mean_epoch_compute_time": _optional_ratio(
                _optional_float(candidate_summary, "mean_epoch_compute_seconds"),
                _optional_float(baseline_summary, "mean_epoch_compute_seconds"),
            ),
            "fit_peak_allocated_memory": _optional_ratio(
                _optional_float(candidate_summary, "fit_peak_allocated_gib"),
                _optional_float(baseline_summary, "fit_peak_allocated_gib"),
            ),
        },
        "trajectory_difference": {
            "train_rmse": float(np.sqrt(np.mean((candidate_train - baseline_train) ** 2))),
            "train_max_abs": float(np.max(np.abs(candidate_train - baseline_train))),
            "validation_rmse": float(
                np.sqrt(np.mean((candidate_val - baseline_val) ** 2))
            ),
            "validation_max_abs": float(np.max(np.abs(candidate_val - baseline_val))),
        },
        "outputs": {
            "training_epoch_comparison": trajectory_path.name,
            "training_curves": "training_curves.png",
        },
    }

    output_json = output_dir / "training_comparison.json"
    output_json.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
