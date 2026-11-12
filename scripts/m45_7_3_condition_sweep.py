"""M45.7.3: evaluate existing static-oracle checkpoints on all held-out coarse fields.

Synthetic shared-mode/block-average Q/B only; NOT physical DNS/LES.
Reuses the existing checkpoint loader, Heun sampler and ensemble diagnostics.
No training, architecture changes, or per-checkpoint re-normalization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import torch

from graph_attention.data.two_scale_synthetic import (
    CHANNELS, SyntheticQB, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import (
    _analytic_mode_second_moment, _correlations, _ensemble_metrics,
    _fixed_coarse_references, _generate_ensemble, _load_model, _write_csv,
)
from scripts.m45_synthetic_pilot import _to_device


def _check_source(settings: dict, result: dict, *, seed: int, policy: str) -> None:
    expected = {
        "operator": "block_average", "source_subspace": "nullspace",
        "cases": ["shared_mode"], "train_samples": 24, "val_samples": 8,
        "steps": 10000, "coarse_side": 4, "refinement": 2,
        "width": 16, "heads": 4, "supervision": "oracle",
        "seed": seed, "val_seed": 7002026, "residual_policy": policy,
    }
    for key, value in expected.items():
        if settings.get(key) != value:
            raise ValueError(f"checkpoint {key}: expected {value}, got {settings.get(key)}")
    if result.get("mode") != "static" or result.get("training_steps") != 10000:
        raise ValueError("expected trained static 10000-update checkpoint")
    if result.get("supervision") != "oracle" or result.get("residual_policy") != policy:
        raise ValueError("checkpoint metrics and settings disagree")


def _generation_seeds(*, base: int, validation_index: int, count: int) -> list[int]:
    """Same initial noise across checkpoints, distinct streams across coarse fields.

    Index 0 reproduces the original M45.3 diagnostic noise seeds exactly.
    """
    if base < 0 or validation_index < 0 or count < 2:
        raise ValueError("base >= 0, index >= 0 and count >= 2 required")
    return [
        base + 1_000_000 + validation_index * 10_000_000 + 3277 * draw
        for draw in range(count)
    ]


def _field_records(
    qb: SyntheticQB, pair, generated: torch.Tensor, reference: torch.Tensor, *,
    seed: int, policy: str, validation_index: int, heun_steps: int,
) -> tuple[dict, list[dict], list[dict]]:
    """Summarize conditional ensembles without pooling distinct coarse fields."""
    template = pair.mode[:, 0]
    gen = _ensemble_metrics(generated, template, qb)
    ref = _ensemble_metrics(reference, template, qb)
    analytic_var = _analytic_mode_second_moment(qb, pair.coarse, "shared_mode")
    if not torch.isfinite(generated).all() or not torch.isfinite(reference).all():
        raise ValueError("non-finite generated or reference samples")
    if gen["max_Q_residual"] > 1e-5 or ref["max_Q_residual"] > 1e-5:
        raise ValueError("coarse restriction defect exceeds 1e-5")
    if analytic_var <= 0 or ref["mode_variance"] <= 0:
        raise ValueError("reference mode variance must be positive")
    gen_corr = _correlations(generated, qb.nf)
    ref_corr = _correlations(reference, qb.nf)
    prefix = {
        "seed": seed, "residual_policy": policy,
        "validation_index": validation_index, "sample_id": pair.sample_id,
    }
    row = {
        **prefix, "heun_steps": heun_steps, "generated_samples": gen["samples"],
        "analytic_mode_variance": analytic_var,
        "reference_mode_mean": ref["mode_mean"],
        "reference_mode_variance": ref["mode_variance"],
        "reference_mode_second_moment": ref["mode_second_moment"],
        "generated_mode_mean": gen["mode_mean"],
        "generated_mode_variance": gen["mode_variance"],
        "generated_mode_second_moment": gen["mode_second_moment"],
        "recovery_vs_analytic": gen["mode_variance"] / analytic_var,
        "recovery_vs_empirical": gen["mode_variance"] / ref["mode_variance"],
        "max_Q_generated": gen["max_Q_residual"],
        "max_Q_reference": ref["max_Q_residual"],
    }
    for i, channel in enumerate(CHANNELS):
        row[f"generated_rms_{channel}"] = gen["channel_rms"][i]
        row[f"reference_rms_{channel}"] = ref["channel_rms"][i]
    for dist, correlations in (("generated", gen_corr), ("reference", ref_corr)):
        for lag in (1, 4):
            vals = [r for r in correlations if r["axis"] == "x" and r["lag"] == lag]
            if len(vals) != 1:
                raise ValueError("expected one x-lag correlation for each requested lag")
            row[f"{dist}_corr_x_lag{lag}"] = vals[0]["correlation"]
            row[f"{dist}_cov_x_lag{lag}"] = vals[0]["covariance"]

    correlations = [
        {**prefix, "distribution": dist, **c}
        for dist, records in (("generated", gen_corr), ("reference", ref_corr))
        for c in records
    ]
    denominator = template.square().sum()
    amplitudes = [
        {**prefix, "distribution": dist, "draw": draw, "amplitude": float(value)}
        for dist, samples in (("generated", generated), ("reference", reference))
        for draw, value in enumerate(samples[:, :, 1] @ template / denominator)
    ]
    return row, correlations, amplitudes


def _summarize_fields(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("no validation fields")
    keys = ("recovery_vs_analytic", "recovery_vs_empirical",
            "generated_corr_x_lag1", "generated_corr_x_lag4",
            "generated_mode_mean", "max_Q_generated")
    out = {"validation_fields": len(rows)}
    for key in keys:
        values = [float(row[key]) for row in rows]
        out[key + "_mean"] = mean(values)
        out[key + "_std_across_fields"] = stdev(values) if len(values) > 1 else 0.0
        out[key + "_min"] = min(values)
        out[key + "_max"] = max(values)
    return out


def evaluate(*, checkpoint_root: Path, output_dir: Path, seed: int,
             policy: str, heun_steps: int = 200, samples: int = 128,
             eval_seed: int = 53177, max_fields: int = 8,
             device_name: str = "cuda") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {output_dir}")
    if policy not in ("fixed", "resample"):
        raise ValueError("residual policy must be fixed or resample")
    if heun_steps < 1 or samples < 2 or not 1 <= max_fields <= 8 or eval_seed < 0:
        raise ValueError("invalid integration, sample, evaluation seed, or field count")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    source = json.loads((checkpoint_root / "summary.json").read_text())
    settings = source["settings"]
    _check_source(settings, source["results"][0], seed=seed, policy=policy)
    checkpoint = checkpoint_root / "shared_mode" / "block_average" / "static" / "checkpoint.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    device = torch.device(device_name)
    mesh = build_hit_two_scale_geometry(
        coarse_side=settings["coarse_side"], refinement=settings["refinement"],
    )
    mesh_device = _to_device(mesh, device)
    qb = SyntheticQB(mesh, "block_average")
    qb_device = SyntheticQB(mesh_device, "block_average")
    model, scales = _load_model(checkpoint, settings, device)
    validation = [
        make_synthetic_pair(
            qb, case="shared_mode", sample_id=100000 + i,
            seed=settings["val_seed"] + 3001 * i,
        )
        for i in range(max_fields)
    ]
    field_rows, corr_rows, amplitude_rows = [], [], []
    with torch.no_grad():
        for i, pair in enumerate(validation):
            gen_seeds = _generation_seeds(
                base=eval_seed, validation_index=i, count=samples,
            )
            generated = _generate_ensemble(
                model, mesh_device, qb_device, pair, scales,
                device=device, seeds=gen_seeds, steps=heun_steps,
                feedback_off=False,
            )
            # Independent conditional target samples for exactly this U_c.
            reference = _fixed_coarse_references(
                qb, pair.coarse, "shared_mode",
                seed=eval_seed + 700_000 + i * 10_000_000,
                count=samples,
            )
            row, corr, amplitudes = _field_records(
                qb, pair, generated, reference, seed=seed, policy=policy,
                validation_index=i, heun_steps=heun_steps,
            )
            field_rows.append(row)
            corr_rows.extend(corr)
            amplitude_rows.extend(amplitudes)
            print(
                f"M45.7.3 {seed}/{policy} val={i + 1}/{max_fields} "
                f"recovery_analytic={100 * row['recovery_vs_analytic']:.2f}% "
                f"mean={row['generated_mode_mean']:.5g}",
                flush=True,
            )

    output_dir.mkdir(parents=True)
    _write_csv(output_dir / "per_field.csv", field_rows, list(field_rows[0]))
    _write_csv(output_dir / "spatial_correlations.csv", corr_rows, list(corr_rows[0]))
    _write_csv(output_dir / "mode_amplitudes.csv", amplitude_rows, list(amplitude_rows[0]))
    report = {
        "schema": "graph_attention.m45_7_3_condition_sweep.v1",
        "scope": "synthetic shared_mode block-average Q/B, NOT physical DNS/LES",
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_settings": settings,
        "evaluation": {
            "training_seed": seed, "residual_policy": policy,
            "heun_steps": heun_steps, "generated_samples": samples,
            "eval_seed": eval_seed, "validation_fields": max_fields,
            "validation_indices": list(range(max_fields)),
            "source_seed_rule": "eval_seed + 1000000 + val_index * 10000000 + draw * 3277",
            "reference_seed_rule": "eval_seed + 700000 + val_index * 10000000 + draw * 1009",
            "source_projected_to_Q_nullspace": True,
            "reference_policy": "independent_conditional_samples",
            "primary_recovery_denominator": "analytic conditional mode variance",
        },
        "statistics": _summarize_fields(field_rows),
    }
    (output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"M45.7.3 evaluation complete: {output_dir}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(2026, 2027, 2028), required=True)
    parser.add_argument("--residual-policy", choices=("fixed", "resample"), required=True)
    parser.add_argument("--heun-steps", type=int, default=200)
    parser.add_argument("--generated-samples", type=int, default=128)
    parser.add_argument("--eval-seed", type=int, default=53177)
    parser.add_argument("--max-fields", type=int, default=8)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    evaluate(
        checkpoint_root=args.checkpoint_root,
        output_dir=args.output_dir,
        seed=args.seed,
        policy=args.residual_policy,
        heun_steps=args.heun_steps,
        samples=args.generated_samples,
        eval_seed=args.eval_seed,
        max_fields=args.max_fields,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
