"""Case-resolved teacher normalization audit for the M48 CEDAR task.

No training or model changes. Reads and checks every selected MS_SR teacher
snapshot, with equal independent-snapshot weights. Outputs JSON and CSV to
a new GraphAttention-owned directory; never writes to MS_SR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch

from graph_attention.data.ms_sr_adapter import CEDAR_REVISION, FIELDS, MANIFEST_SHA256
from graph_attention.data.ms_sr_normalization_audit import audit_teacher_normalization
from graph_attention.data.ms_sr_training import select_role_records


MOMENT_FIELDS = (
    "coarse_mean", "coarse_std", "residual_mean", "residual_std", "residual_rms",
)
COMPARISON_FIELDS = (
    "coarse_mean_shift_in_global_std",
    "coarse_std_over_global_std",
    "residual_rms_over_global_rms",
    "residual_mean_over_global_rms",
    "residual_mean_over_case_rms",
)


def _write_csv(path: Path, headers: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("x", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def _case_rows(audit: dict) -> list[dict]:
    """One row per named conservative component and group."""
    rows = []
    categories = (
        ("pooled", {"pooled": {
            "source_count": audit["source_count"],
            "moments": audit["pooled"],
            "vs_pooled": {
                "coarse_mean_shift_in_global_std": [0.0] * len(FIELDS),
                "coarse_std_over_global_std": [1.0] * len(FIELDS),
                "residual_rms_over_global_rms": [1.0] * len(FIELDS),
                "residual_mean_over_global_rms": [
                    m / s for m, s in zip(
                        audit["pooled"]["residual_mean"],
                        audit["pooled"]["residual_rms"],
                    )
                ],
                "residual_mean_over_case_rms": [
                    m / s for m, s in zip(
                        audit["pooled"]["residual_mean"],
                        audit["pooled"]["residual_rms"],
                    )
                ],
            },
        }}),
        ("reynolds_group", audit["per_reynolds_group"]),
        ("case", audit["per_case"]),
    )
    for scope, groups in categories:
        for label, item in groups.items():
            for i, component in enumerate(FIELDS):
                row = {
                    "scope": scope, "name": label,
                    "source_count": item["source_count"], "field": component,
                }
                row.update({name: item["moments"][name][i] for name in MOMENT_FIELDS})
                row.update({
                    name: item["vs_pooled"][name][i] for name in COMPARISON_FIELDS
                })
                rows.append(row)
    return rows


def _snapshot_rows(audit: dict) -> list[dict]:
    rows = []
    for item in audit["per_snapshot"]:
        ident = item["source"]
        for i, component in enumerate(FIELDS):
            row = {
                "case": ident["case"],
                "snapshot_id": ident["snapshot_id"],
                "physical_time_s": item["physical_time_s"],
                "source_sha256": ident["source_sha256"],
                "field": component,
                "C_residual_max_component_scaled_linf":
                    item["C_residual_max_component_scaled_linf"],
            }
            row.update({name: item["moments"][name][i] for name in MOMENT_FIELDS})
            rows.append(row)
    return rows


def _extrema(audit: dict) -> dict:
    """Surface large regime changes without imposing heuristic pass thresholds."""
    result = {}
    for i, field in enumerate(FIELDS):
        groups = audit["per_case"]
        shifts = [
            (case, stats["vs_pooled"]["coarse_mean_shift_in_global_std"][i])
            for case, stats in groups.items()
        ]
        cstd = [
            (case, stats["vs_pooled"]["coarse_std_over_global_std"][i])
            for case, stats in groups.items()
        ]
        rrms = [
            (case, stats["vs_pooled"]["residual_rms_over_global_rms"][i])
            for case, stats in groups.items()
        ]
        result[field] = {
            "largest_absolute_mean_shift": dict(zip(
                ("case", "in_global_std"), max(shifts, key=lambda x: abs(x[1]))
            )),
            "coarse_std_ratio_min": dict(zip(
                ("case", "ratio"), min(cstd, key=lambda x: x[1])
            )),
            "coarse_std_ratio_max": dict(zip(
                ("case", "ratio"), max(cstd, key=lambda x: x[1])
            )),
            "residual_rms_ratio_min": dict(zip(
                ("case", "ratio"), min(rrms, key=lambda x: x[1])
            )),
            "residual_rms_ratio_max": dict(zip(
                ("case", "ratio"), max(rrms, key=lambda x: x[1])
            )),
        }
    return result


def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing audit output: {args.output_dir}")
    if args.max_sources < 0 or args.expected_sources < 0:
        raise ValueError("source limits must be nonnegative")
    records, selection_sha256 = select_role_records(
        args.ms_sr_root, role="teacher", filter_id=args.filter_id,
        case_prefix=args.case_prefix, max_sources=args.max_sources,
        manifest_sha256=args.manifest_sha256,
    )
    if args.expected_sources and len(records) != args.expected_sources:
        raise RuntimeError(
            f"teacher source count {len(records)} != expected {args.expected_sources}"
        )
    audit = audit_teacher_normalization(
        args.ms_sr_root, records, selection_sha256,
        manifest_sha256=args.manifest_sha256,
    )
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
    audit["provenance"] = {
        "graph_attention_commit": revision,
        "graph_attention_dirty": dirty,
        "ms_sr_campaign_data_sha256": hashlib.sha256(
            (args.ms_sr_root / "code" / "campaign_data.py").read_bytes()
        ).hexdigest(),
        "ms_sr_torchcedar_sha256": hashlib.sha256(
            (args.ms_sr_root / "code" / "models.py").read_bytes()
        ).hexdigest(),
        "ms_sr_cedar_revision": CEDAR_REVISION,
        "python_version": sys.version,
        "torch_version": torch.__version__,
    }
    audit["maxima_across_cases"] = _extrema(audit)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "normalization_audit.json").open("x", encoding="utf-8") as output:
        json.dump(audit, output, indent=2, allow_nan=False)
        output.write("\n")
    rows = _case_rows(audit)
    _write_csv(
        args.output_dir / "case_comparison.csv",
        ("scope", "name", "source_count", "field") + MOMENT_FIELDS + COMPARISON_FIELDS,
        rows,
    )
    _write_csv(
        args.output_dir / "snapshot_moments.csv",
        ("case", "snapshot_id", "physical_time_s", "source_sha256", "field",
         "C_residual_max_component_scaled_linf") + MOMENT_FIELDS,
        _snapshot_rows(audit),
    )
    print(
        f"PASS: {len(records)} teacher sources, "
        f"{audit['case_count']} cases, max C(R) defect "
        f"{audit['max_C_residual_component_scaled_linf']:.3e}",
        flush=True,
    )
    for field, maximum in audit["maxima_across_cases"].items():
        shift = maximum["largest_absolute_mean_shift"]
        print(f"{field}: largest |case mean-global mean|/global std = "
              f"{abs(shift['in_global_std']):.3g} ({shift['case']})", flush=True)
    print(f"Audit output: {args.output_dir}", flush=True)
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", default=MANIFEST_SHA256)
    parser.add_argument("--filter-id", choices=("w1p25_l1", "w1p5_l1"), default="w1p25_l1")
    parser.add_argument("--case-prefix", default="",
                        help="Empty selects all teacher cases, including ReT060 and ReT100")
    parser.add_argument("--max-sources", type=int, default=0,
                        help="Zero selects all eligible independent teacher sources")
    parser.add_argument("--expected-sources", type=int, default=44,
                        help="Fail if eligible source count differs; zero disables this check")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
