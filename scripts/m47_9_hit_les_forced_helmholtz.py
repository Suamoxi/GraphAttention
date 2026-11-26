"""Read-only M37/M38(/M40) HIT_LES_FORCED Helmholtz covariance audit.

Reuses saved generated_test.pt and held-out reference fields, *without*
model inference or retraining. Uses the same M47.7 Fourier longitudinal
fraction and wavenumber-weighted cross-term cancellation metrics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from omegaconf import OmegaConf

from graph_attention.evaluation.generation_benchmark import _fixed_mesh_samples
from graph_attention.evaluation.generation_benchmark_3d import _load_grid_3d
from scripts.m47_6_spatial_metrics import interior_divcurl, physical_velocity
from scripts.m47_7_covariance_metrics import (
    aggregate_spectral_tensor, spectral_tensor_audit,
)

PROTOCOL = "m47_9_legacy_hit_les_forced_helmholtz_v1"
FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _population(values, ids, channels, grid, u_ref):
    if len(values) != len(ids):
        raise ValueError("field count does not match reference/generation IDs")
    channels = tuple(c.split(".")[0] for c in channels)
    if len(channels) != len(FIELDS) or set(channels) != set(FIELDS):
        raise ValueError("expected exactly five conservative HIT channels")
    order = [channels.index(c) for c in FIELDS]
    records = []
    rows = []
    failures = []
    for sid, field in zip(ids, values, strict=True):
        ordered = field[:, order]
        rho = ordered[:, 0]
        if not np.isfinite(ordered).all() or np.any(rho <= 0):
            failures.append({
                "id": sid,
                "nonpositive_density_fraction": float(np.mean(rho <= 0)),
                "nonfinite_fraction": float(np.mean(~np.isfinite(ordered))),
            })
            continue
        grid_velocity = physical_velocity(ordered, grid, u_ref)
        report = spectral_tensor_audit(grid_velocity, grid)
        divcurl = interior_divcurl(grid_velocity, grid.spacing, trims=(0, 2, 4))
        centered = grid_velocity - grid_velocity.mean(
            axis=(0, 1, 2), keepdims=True,
        )
        tke = 0.5 * float(np.mean(np.sum(centered**2, axis=-1)))
        internal = ordered[:, 4] - 0.5 * np.sum(ordered[:, 1:4]**2, axis=1) / rho
        records.append(report)
        rows.append({
            "sample_id": sid,
            "longitudinal_energy_fraction": report["longitudinal_energy_fraction_integrated"],
            "cross_cancellation": report["cross_cancellation_integrated"],
            "tke_fluctuation_m2_s2": tke,
            "interior_divcurl_trim0": divcurl["0"]["divergence_to_vorticity_rms"],
            "interior_divcurl_trim4": divcurl["4"]["divergence_to_vorticity_rms"],
            "nonpositive_internal_energy_fraction": float(np.mean(internal <= 0)),
        })
    if not records:
        raise ValueError("no valid positive-density fields")
    aggregation = aggregate_spectral_tensor(records)
    names = ("longitudinal_energy_fraction", "cross_cancellation",
             "tke_fluctuation_m2_s2", "interior_divcurl_trim4")
    stats = {}
    for name in names:
        values = np.array([row[name] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        stats[name] = {
            "mean": float(np.mean(finite)) if len(finite) else None,
            "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
            "q10": float(np.quantile(finite, .1)) if len(finite) else None,
            "q90": float(np.quantile(finite, .9)) if len(finite) else None,
        }
    return {
        "requested_count": len(ids),
        "valid_count": len(rows),
        "invalid_count": len(failures),
        "invalid_fields": failures,
        "scalar_statistics": stats,
        "spectral_tensor": aggregation,
        "per_field": rows,
    }


def _generation(text):
    if "=" not in text:
        raise argparse.ArgumentTypeError("generation must be LABEL=/absolute/path")
    name, location = text.split("=", 1)
    if name not in {"M37", "M38", "M40"} or not location.startswith("/"):
        raise argparse.ArgumentTypeError("generation must be M37|M38|M40=/absolute/path")
    return name, Path(location)


def _load(generation, benchmark_cfg):
    for name in ("generated_test.pt", "resolved_config.yaml", "summary.json"):
        if not (generation / name).is_file():
            raise FileNotFoundError(f"required archived generation file absent: {generation / name}")
    source_cfg = OmegaConf.load(generation / "resolved_config.yaml")
    grid, mesh = _load_grid_3d(source_cfg, benchmark_cfg)
    artifact = torch.load(
        generation / "generated_test.pt", map_location="cpu", weights_only=True,
    )
    generated, reference, generation_ids, reference_ids, channels = _fixed_mesh_samples(artifact)
    if generated.shape[1] != int(np.prod(grid.source_shape)):
        raise ValueError("saved nodal field count differs from source mesh grid")
    if not len(generation_ids) == len(reference_ids) == generated.shape[0]:
        raise ValueError("generated and reference populations not aligned")
    return generated, reference, generation_ids, reference_ids, channels, grid, mesh


def _fingerprint(reference):
    arr = np.ascontiguousarray(reference)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


def _mesh_signature(grid, mesh):
    return {
        "mesh_file": str(mesh),
        "source_shape": list(grid.source_shape),
        "unique_fft_shape": list(grid.shape),
        "spacing_m": list(grid.spacing),
        "periodic_mode": grid.periodic_endpoint_mode,
        "node_order_sha256": hashlib.sha256(
            np.ascontiguousarray(grid.linear_indices).view(np.uint8)
        ).hexdigest(),
    }


def run(args):
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output {args.output_dir}")
    generations = dict(args.generation)
    if len(generations) != len(args.generation) or set(generations) < {"M37", "M38"}:
        raise ValueError("provide distinct M37 and M38 archived generations")
    benchmark_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs/benchmark_generation_3d.yaml"
    )
    if str(benchmark_cfg.spectra.periodic_endpoint_mode) != "drop_max":
        raise ValueError("expected historical drop_max periodic FFT convention")
    case_cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "cases/HIT_LES_FORCED.yaml"
    )
    if case_cfg.case_id != "HIT_LES_FORCED":
        raise ValueError("wrong case reference metadata")
    u_ref = float(case_cfg.references.U_ref.value)
    if not np.isfinite(u_ref) or u_ref <= 0:
        raise ValueError("invalid reference velocity m/s")

    output = {
        "protocol": PROTOCOL,
        "status": "complete",
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "case": "HIT_LES_FORCED",
        "U_ref_m_per_s": u_ref,
        "units": "physical velocity m/s and tke m2/s2",
        "metric_definition": "M47.7 spectral_tensor_audit on unique 32^3 periodic grid with k<=Nyquist; integrated per-sample ratios, then population mean",
        "warning": "This is the historical LES simulation reference, not a DNS claim. Cross-spectrum and periodic FFT assume periodicity; trim4 divcurl tests interior sensitivity. Generated/reference fields are unpaired.",
        "generation_paths": {k: str(v) for k, v in generations.items()},
        "populations": {},
    }
    reference_ids0 = None
    reference_hash0 = None
    mesh0 = None
    for label in ("M37", "M38", "M40"):
        if label not in generations:
            continue
        generated, reference, generation_ids, reference_ids, channels, grid, mesh = (
            _load(generations[label], benchmark_cfg)
        )
        signature = _mesh_signature(grid, mesh)
        reference_hash = _fingerprint(reference)
        if reference_ids0 is None:
            reference_ids0 = reference_ids
            reference_hash0 = reference_hash
            mesh0 = signature
            output["reference_field_sha256"] = reference_hash
            output["reference_ids"] = list(reference_ids)
            output["mesh"] = signature
            output["populations"]["reference"] = _population(
                reference, reference_ids, channels, grid, u_ref,
            )
        else:
            if reference_ids != reference_ids0 or reference_hash != reference_hash0:
                raise ValueError(f"{label} does not use identical held-out reference fields/IDs")
            if signature != mesh0:
                raise ValueError(f"{label} source mesh differs from baseline")
        output["populations"][label] = _population(
            generated, generation_ids, channels, grid, u_ref,
        )
        del generated, reference
        print(
            f"[M47.9] {label} valid={output['populations'][label]['valid_count']} "
            f"mean_fL={output['populations'][label]['scalar_statistics']['longitudinal_energy_fraction']['mean']:.6g}",
            flush=True,
        )

    baseline = output["populations"]["reference"]["scalar_statistics"]
    for label in ("M37", "M38", "M40"):
        if label not in output["populations"]:
            continue
        stats = output["populations"][label]["scalar_statistics"]
        output["populations"][label]["vs_reference"] = {
            "longitudinal_fraction_difference_percentage_points": 100 * (
                stats["longitudinal_energy_fraction"]["mean"] -
                baseline["longitudinal_energy_fraction"]["mean"]
            ),
            "cross_cancellation_difference": (
                stats["cross_cancellation"]["mean"] -
                baseline["cross_cancellation"]["mean"]
            ),
            "tke_ratio": (
                stats["tke_fluctuation_m2_s2"]["mean"] /
                baseline["tke_fluctuation_m2_s2"]["mean"]
            ),
        }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(output, indent=2, allow_nan=False) + "\n"
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation", type=_generation, action="append", required=True,
                   help="M37|M38|M40=/absolute/path/to/generation")
    p.add_argument("--output-dir", type=Path, required=True)
    result = run(p.parse_args())
    print("[M47.9] COMPLETE " + ", ".join(result["populations"]), flush=True)


if __name__ == "__main__":
    main()
