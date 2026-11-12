"""M45.8.1 validation: conditional energy, physical k-spectra and spatial statistics.

Metrics are computed per *held-out coarse field*, with repeated independent
target and generated residual draws. Neither pairwise sample errors nor
global mean-mode recovery is used as the primary distributional metric.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from graph_attention.data.conditional_hit_synthetic import (
    LocalHITConfig, make_local_pair, sample_local_residual,
)
from graph_attention.data.two_scale_synthetic import SyntheticQB
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import _generate_ensemble, _load_model, _write_csv
from scripts.m45_synthetic_pilot import _to_device
from scripts.m45_8_local_train import data_seed, residual_seed


def radial_spectrum(x: torch.Tensor, nf: int, length: float) -> torch.Tensor:
    """Shell-binned E(k) in physical k [rad / length], Parseval-normalized.

    A row of radial_spectrum is the energy in a shell, divided by delta-k.
    Hence sum(E_k * delta_k) = mean(0.5 * |u|^2).
    """
    if x.shape != (nf**3, 3):
        raise ValueError("expected [fine nodes, 3] velocity field")
    if length <= 0:
        raise ValueError("periodic length must be positive")
    k = torch.fft.fftfreq(nf, d=1.0 / nf)
    xyz = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), dim=-1)
    shell = xyz.square().sum(-1).sqrt().round().long()
    hat = torch.fft.fftn(x.reshape(nf, nf, nf, 3), dim=(0, 1, 2)) / nf**3
    energy = 0.5 * hat.abs().square().sum(-1)
    by_shell = torch.bincount(
        shell.flatten(), weights=energy.flatten(),
        minlength=int(shell.max()) + 1,
    )
    return by_shell / (2.0 * torch.pi / length)


def divergence_rms(x: torch.Tensor, nf: int, length: float) -> float:
    if x.shape != (nf**3, 3) or length <= 0:
        raise ValueError("invalid three-component periodic velocity field")
    k = torch.fft.fftfreq(nf, d=length / nf) * (2.0 * torch.pi)
    wv = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), dim=-1)
    hat = torch.fft.fftn(x.reshape(nf, nf, nf, 3), dim=(0, 1, 2))
    div_hat = 1j * (wv * hat).sum(-1)
    return float(torch.fft.ifftn(div_hat).real.square().mean().sqrt())


def conditional_bin_energy(
    qb: SyntheticQB, pair, samples: torch.Tensor, boundaries: torch.Tensor,
) -> list[dict]:
    """One conditional second moment per local coarse-strain bin.

    A strain bin contains many fine nodes; generated and reference must
    use identical boundaries and coarse states to be comparable.
    """
    if samples.ndim != 3 or samples.shape[1:] != (qb.nf**3, 3):
        raise ValueError("samples must be [draws, fine nodes, 3]")
    if boundaries.ndim != 1 or boundaries.numel() < 3:
        raise ValueError("need at least two strain bins")
    parent_strain = qb.reconstruct(pair.coarse_strain[:, None]).squeeze(-1)
    bins = torch.bucketize(parent_strain.contiguous(), boundaries[1:-1])
    squared = 0.5 * samples.square().sum(-1).mean(dim=0)
    rows = []
    for idx in range(boundaries.numel() - 1):
        mask = bins == idx
        nodes = int(mask.sum())
        rows.append({
            "bin": idx,
            "strain_lower": float(boundaries[idx]),
            "strain_upper": float(boundaries[idx + 1]),
            "nodes": nodes,
            "kinetic_energy_per_node": float(squared[mask].mean()) if nodes else 0.0,
        })
    return rows


def spatial_correlations(samples: torch.Tensor, nf: int,
                         max_lag: int = 4) -> list[dict]:
    if samples.ndim != 3 or samples.shape[1:] != (nf**3, 3):
        raise ValueError("invalid residual ensemble")
    if samples.shape[0] < 2:
        raise ValueError("need >= 2 draws for centered covariance")
    field = samples - samples.mean(dim=0, keepdim=True)
    field = field.reshape(samples.shape[0], nf, nf, nf, 3)
    variance = field.square().mean().clamp_min(1e-15)
    rows = []
    for axis, name in enumerate(("x", "y", "z"), start=1):
        for lag in range(min(max_lag, nf // 2) + 1):
            covariance = (field * torch.roll(field, -lag, dims=axis)).mean()
            rows.append({
                "axis": name, "lag": lag,
                "covariance": float(covariance * samples.shape[0] / (samples.shape[0] - 1)),
                "correlation": float(covariance / variance),
            })
    return rows


def summarize_conditional_bins(rows: list[dict]) -> list[dict]:
    """Pool by node count across fields, keeping generated and truth separate."""
    summary = []
    for distribution in ("generated", "reference"):
        for idx in range(5):
            subset = [r for r in rows if r["distribution"] == distribution and r["bin"] == idx]
            count = sum(r["nodes"] for r in subset)
            summary.append({
                "distribution": distribution, "bin": idx,
                "fine_nodes_per_draw": count,
                "mean_kinetic_energy": (
                    sum(r["nodes"] * r["kinetic_energy_per_node"] for r in subset) / count
                    if count else 0.0
                ),
            })
    return summary


def evaluate(*, checkpoint_root: Path, output_dir: Path, draws: int = 16,
             heun_steps: int = 50, eval_seed: int = 76001,
             val_fields: int | None = None, device_name: str = "cuda") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing directory: {output_dir}")
    if draws < 2 or heun_steps < 1 or eval_seed < 0:
        raise ValueError("draws >=2, Heun steps >=1, eval_seed >=0 required")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")

    report = json.loads((checkpoint_root / "summary.json").read_text())
    if report.get("schema") != "graph_attention.m45_8_local_train.v1":
        raise ValueError("expected M45.8 local-HIT training checkpoint")
    s = report["settings"]
    if (s["channels"] != 3 or s["case"] != "local_hit"
            or s["operator"] != "block_average"):
        raise ValueError("expected three-channel local HIT block_average checkpoint")
    num_fields = val_fields if val_fields is not None else s["val_fields"]
    if not 1 <= num_fields <= s["val_fields"]:
        raise ValueError("requested too many held-out fields")
    config = LocalHITConfig(**s["generator"])
    device = torch.device(device_name)
    mesh = build_hit_two_scale_geometry(
        coarse_side=s["coarse_side"], refinement=s["refinement"],
    )
    qb = SyntheticQB(mesh, "block_average")
    qb_gpu = SyntheticQB(_to_device(mesh, device), "block_average")
    model, scales = _load_model(
        checkpoint_root / "local_hit" / "block_average" / s["mode"] / "checkpoint.pt",
        s, device,
    )
    val = [
        make_local_pair(
            qb, config, sample_id=100000 + i,
            coarse_seed=data_seed(s["validation_seed"], i),
            residual_seed=residual_seed(s["validation_seed"] + 8_000_000, i),
        )
        for i in range(num_fields)
    ]
    # Descriptive bin edges derived only from held-out coarse features; no
    # target residual or generated residual enters the bin definitions.
    boundaries = torch.quantile(
        torch.cat([p.coarse_strain for p in val]),
        torch.linspace(0, 1, 6),
    )

    field_rows, bin_rows, spectrum_rows, corr_rows = [], [], [], []
    with torch.no_grad():
        for i, pair in enumerate(val):
            seeds = [
                eval_seed + 1_000_000 + 10_000_000 * i + 3277 * j
                for j in range(draws)
            ]
            generated = _generate_ensemble(
                model, qb_gpu.mesh, qb_gpu, pair, scales,
                device=device, seeds=seeds, steps=heun_steps,
                feedback_off=False,
            )
            reference = torch.stack([
                sample_local_residual(
                    qb, pair.coarse, config,
                    seed=eval_seed + 2_000_000 + 10_000_000 * i + 1009 * j,
                )[0] for j in range(draws)
            ])
            for dist, samples in (("generated", generated), ("reference", reference)):
                defect = max(float(qb.restrict(x).abs().max()) for x in samples)
                if not torch.isfinite(samples).all() or defect > 1e-5:
                    raise RuntimeError(f"{dist} nonfinite or coarse defect >1e-5")
                energy = 0.5 * samples.square().sum(-1).mean()
                full_energy = 0.5 * (samples + pair.baseline).square().sum(-1).mean()
                field_rows.append({
                    "mode": s["mode"], "validation_index": i,
                    "sample_id": pair.sample_id, "distribution": dist,
                    "draws": draws, "heun_steps": heun_steps,
                    "mean_residual_energy": float(energy),
                    "mean_total_energy": float(full_energy),
                    "rms_u": float(samples[..., 0].square().mean().sqrt()),
                    "rms_v": float(samples[..., 1].square().mean().sqrt()),
                    "rms_w": float(samples[..., 2].square().mean().sqrt()),
                    "coarse_defect": defect,
                    "fine_divergence_rms": mean(
                        divergence_rms(x + pair.baseline, qb.nf, mesh.periodic_length)
                        for x in samples
                    ),
                })
                for row in conditional_bin_energy(qb, pair, samples, boundaries):
                    bin_rows.append({
                        "mode": s["mode"], "validation_index": i,
                        "distribution": dist, **row,
                    })
                for row in spatial_correlations(samples, qb.nf):
                    corr_rows.append({
                        "mode": s["mode"], "validation_index": i,
                        "distribution": dist, **row,
                    })
                spectra = torch.stack([
                    radial_spectrum(x + pair.baseline, qb.nf, mesh.periodic_length)
                    for x in samples
                ]).mean(0)
                dk = 2 * torch.pi / mesh.periodic_length
                for shell, power in enumerate(spectra):
                    spectrum_rows.append({
                        "mode": s["mode"], "validation_index": i,
                        "distribution": dist, "shell_index": shell,
                        "k_physical": shell * dk,
                        "E_k": float(power),
                    })
            print(f"M45.8 evaluated {s['mode']} field={i + 1}/{num_fields}", flush=True)

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "per_field.csv", field_rows, list(field_rows[0]))
    _write_csv(output_dir / "conditional_energy.csv", bin_rows, list(bin_rows[0]))
    _write_csv(output_dir / "conditional_summary.csv",
               summarize_conditional_bins(bin_rows),
               ["distribution", "bin", "fine_nodes_per_draw", "mean_kinetic_energy"])
    _write_csv(output_dir / "spatial_correlations.csv", corr_rows, list(corr_rows[0]))
    _write_csv(output_dir / "spectra.csv", spectrum_rows, list(spectrum_rows[0]))
    results = {}
    for dist in ("reference", "generated"):
        subset = [row for row in field_rows if row["distribution"] == dist]
        results[dist] = {
            "mean_residual_energy": mean(row["mean_residual_energy"] for row in subset),
            "mean_total_energy": mean(row["mean_total_energy"] for row in subset),
            "mean_fine_divergence_rms": mean(row["fine_divergence_rms"] for row in subset),
            "max_coarse_defect": max(row["coarse_defect"] for row in subset),
        }
    result = {
        "schema": "graph_attention.m45_8_local_evaluate.v1",
        "scope": "synthetic, periodic HIT-like velocity, NOT DNS/LES",
        "checkpoint_root": str(checkpoint_root),
        "settings": {
            "mode": s["mode"], "draws": draws, "heun_steps": heun_steps,
            "eval_seed": eval_seed, "val_fields": num_fields,
            "validation_seed": s["validation_seed"],
            "dataset_seed": s["dataset_seed"],
            "init_seed": s["init_seed"],
            "schedule_seed": s["schedule_seed"],
            "residual_seed": s["residual_seed"],
            "generator": s["generator"],
            "bins_defined_by": "quantiles of coarse strain over held-out fields",
            "k_units": "physical angular wavenumber rad / periodic_length",
            "spectrum_normalization": "sum E(k)*delta_k = mean kinetic energy per mass",
        },
        "results": results,
        "conditional_bin_edges": boundaries.tolist(),
    }
    (output_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=16)
    parser.add_argument("--heun-steps", type=int, default=50)
    parser.add_argument("--eval-seed", type=int, default=76001)
    parser.add_argument("--val-fields", type=int, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    evaluate(
        checkpoint_root=args.checkpoint_root, output_dir=args.output_dir,
        draws=args.draws, heun_steps=args.heun_steps,
        eval_seed=args.eval_seed, val_fields=args.val_fields,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
