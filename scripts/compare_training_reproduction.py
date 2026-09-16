"""Compare two short training runs for refactor-level numerical reproduction."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch


_IGNORED_MANIFEST_KEYS = {"runtime_provenance"}
_IGNORED_CHECKPOINT_KEYS = {"runtime_provenance"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    compare_runs(
        args.baseline.expanduser().resolve(),
        args.candidate.expanduser().resolve(),
        atol=args.atol,
        rtol=args.rtol,
    )


def compare_runs(
    baseline: Path,
    candidate: Path,
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    for path in (baseline, candidate):
        if not path.is_dir():
            raise NotADirectoryError(path)

    _compare_history(
        baseline / "history.csv",
        candidate / "history.csv",
        atol=atol,
        rtol=rtol,
    )
    _compare_json(
        baseline / "dataset_split_manifest.json",
        candidate / "dataset_split_manifest.json",
        ignored_keys=_IGNORED_MANIFEST_KEYS,
        atol=atol,
        rtol=rtol,
    )
    _compare_standardizers(
        baseline / "standardizers.pt",
        candidate / "standardizers.pt",
        atol=atol,
        rtol=rtol,
    )
    for checkpoint_name in ("best.pt", "last.pt"):
        _compare_checkpoint(
            baseline / checkpoint_name,
            candidate / checkpoint_name,
            atol=atol,
            rtol=rtol,
        )

    baseline_summary = json.loads((baseline / "summary.json").read_text())
    candidate_summary = json.loads((candidate / "summary.json").read_text())
    _assert_nested_equal(
        baseline_summary,
        candidate_summary,
        path="summary.json",
        atol=atol,
        rtol=rtol,
    )

    print("REPRODUCTION CHECK PASSED")
    print(f"baseline:  {baseline}")
    print(f"candidate: {candidate}")
    print(f"tensor tolerances: rtol={rtol:g}, atol={atol:g}")


def _compare_history(
    baseline_path: Path,
    candidate_path: Path,
    *,
    atol: float,
    rtol: float,
) -> None:
    baseline = _read_csv(baseline_path)
    candidate = _read_csv(candidate_path)
    if len(baseline) != len(candidate):
        raise AssertionError("history row counts differ")
    if not baseline:
        raise AssertionError("history is empty")
    if baseline[0].keys() != candidate[0].keys():
        raise AssertionError("history columns differ")

    for row_index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
        for key in left:
            if key == "epoch":
                if left[key] != right[key]:
                    raise AssertionError(f"history epoch differs at row {row_index}")
                continue
            left_value = torch.tensor(float(left[key]), dtype=torch.float64)
            right_value = torch.tensor(float(right[key]), dtype=torch.float64)
            if not torch.allclose(left_value, right_value, rtol=rtol, atol=atol):
                difference = abs(float(left_value - right_value))
                raise AssertionError(
                    f"history differs at row={row_index}, column={key}: "
                    f"baseline={left[key]}, candidate={right[key]}, abs_diff={difference:.17g}"
                )


def _compare_json(
    baseline_path: Path,
    candidate_path: Path,
    *,
    ignored_keys: set[str],
    atol: float,
    rtol: float,
) -> None:
    baseline = json.loads(baseline_path.read_text())
    candidate = json.loads(candidate_path.read_text())
    for key in ignored_keys:
        baseline.pop(key, None)
        candidate.pop(key, None)
    _assert_nested_equal(
        baseline,
        candidate,
        path=baseline_path.name,
        atol=atol,
        rtol=rtol,
    )


def _compare_standardizers(
    baseline_path: Path,
    candidate_path: Path,
    *,
    atol: float,
    rtol: float,
) -> None:
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    _assert_nested_equal(baseline, candidate, path="standardizers", atol=atol, rtol=rtol)


def _compare_checkpoint(
    baseline_path: Path,
    candidate_path: Path,
    *,
    atol: float,
    rtol: float,
) -> None:
    baseline = torch.load(baseline_path, map_location="cpu", weights_only=True)
    candidate = torch.load(candidate_path, map_location="cpu", weights_only=True)
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise TypeError("checkpoint payloads must be dictionaries")
    baseline = {
        key: value for key, value in baseline.items() if key not in _IGNORED_CHECKPOINT_KEYS
    }
    candidate = {
        key: value for key, value in candidate.items() if key not in _IGNORED_CHECKPOINT_KEYS
    }
    _assert_nested_equal(
        baseline,
        candidate,
        path=baseline_path.name,
        atol=atol,
        rtol=rtol,
    )


def _assert_nested_equal(
    baseline: Any,
    candidate: Any,
    *,
    path: str,
    atol: float,
    rtol: float,
) -> None:
    if isinstance(baseline, torch.Tensor):
        if not isinstance(candidate, torch.Tensor):
            raise AssertionError(f"type mismatch at {path}")
        if baseline.shape != candidate.shape or baseline.dtype != candidate.dtype:
            raise AssertionError(f"tensor metadata differs at {path}")
        if baseline.is_floating_point():
            if not torch.allclose(baseline, candidate, rtol=rtol, atol=atol):
                maximum = float((baseline - candidate).abs().max())
                raise AssertionError(f"tensor differs at {path}; max_abs_diff={maximum:.17g}")
        elif not torch.equal(baseline, candidate):
            raise AssertionError(f"tensor differs at {path}")
        return

    if isinstance(baseline, dict):
        if not isinstance(candidate, dict) or baseline.keys() != candidate.keys():
            raise AssertionError(f"mapping keys differ at {path}")
        for key in baseline:
            _assert_nested_equal(
                baseline[key],
                candidate[key],
                path=f"{path}.{key}",
                atol=atol,
                rtol=rtol,
            )
        return

    if isinstance(baseline, (list, tuple)):
        if not isinstance(candidate, type(baseline)) or len(baseline) != len(candidate):
            raise AssertionError(f"sequence differs at {path}")
        for index, (left, right) in enumerate(zip(baseline, candidate, strict=True)):
            _assert_nested_equal(
                left,
                right,
                path=f"{path}[{index}]",
                atol=atol,
                rtol=rtol,
            )
        return

    if isinstance(baseline, float) and isinstance(candidate, (int, float)):
        left = torch.tensor(baseline, dtype=torch.float64)
        right = torch.tensor(float(candidate), dtype=torch.float64)
        if not torch.allclose(left, right, rtol=rtol, atol=atol):
            difference = abs(baseline - float(candidate))
            raise AssertionError(
                f"float differs at {path}: {baseline!r} != {candidate!r}; "
                f"abs_diff={difference:.17g}"
            )
        return

    if baseline != candidate:
        raise AssertionError(f"value differs at {path}: {baseline!r} != {candidate!r}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    main()
