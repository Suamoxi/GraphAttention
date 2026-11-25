"""M47.8B: compare frozen FM response on DNS and Heun paths sharing x0."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from graph_attention.tasks.flow_matching import (
    _deterministic_sampling_source, _model_velocity,
)
from scripts.m47_5_regime_diagnostics import ANCHOR_CASE
from scripts.m47_6_spatial_metrics import scatter_unique
from scripts.m47_8_response_audit import _mode_arg, _response, _to_momentum_rate_velocity
from scripts.m47_8b_context import load_context
from scripts.m47_8b_trajectory_math import (
    compare_response, exact_steps, field_summary, paired_state_metrics, spectrum_summary,
)
from scripts.m47_fm_diagnostics import heun_snapshots

PROTOCOL = "m47_8b_matched_teacher_generated_response_v1"


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("M47.8B requires CUDA in a Grace sbatch job")
    slots = exact_steps(tuple(args.times), args.steps)
    if args.epoch not in (10, 30) or args.validation_index not in range(5) or args.seed < 0:
        raise ValueError("epoch=10|30, validation_index=0..4, seed>=0 required")
    if not args.modes or len(set(args.modes)) != len(args.modes):
        raise ValueError("unique, nonempty Fourier modes required")
    if (not args.amplitudes or len(set(args.amplitudes)) != len(args.amplitudes)
        or any(not np.isfinite(e) or not 0 < e < 1 for e in args.amplitudes)):
        raise ValueError("relative amplitudes must be distinct and in (0,1)")
    ctx = load_context(args)
    model, batch, grid = ctx.model, ctx.batch, ctx.grid
    scale, uref, rho_dns = ctx.scalers.inputs.scale, ctx.u_ref, ctx.dns_rho
    key = f"M47.2/{ANCHOR_CASE}/replicate_0000"
    source = _deterministic_sampling_source(batch, args.seed, (key,))
    samples = heun_snapshots(
        model, ctx.task, batch, steps=args.steps, seed=args.seed,
        sampling_key=key, observe_steps={0, args.steps, *slots},
    )
    if not torch.equal(source, samples[0]):
        raise ValueError("teacher and generated paths did not share exact x0")
    if any(not bool(torch.isfinite(z).all()) for z in samples.values()):
        raise ValueError("nonfinite Heun state")

    measurements, response_pairs = [], []
    for t, step in zip(args.times, slots, strict=True):
        teacher = (1 - t) * source + t * batch.inputs
        generated = samples[step]
        condition = torch.full(
            (batch.num_graphs,), t, device=ctx.device, dtype=source.dtype,
        )
        v_teacher = _model_velocity(
            model, batch, teacher, condition,
            time_embedding_scale=ctx.task.time_embedding_scale,
        )
        v_generated = _model_velocity(
            model, batch, generated, condition,
            time_embedding_scale=ctx.task.time_embedding_scale,
        )
        teacher_rate = scatter_unique(
            _to_momentum_rate_velocity(v_teacher, rho_dns, scale, uref), grid,
        )
        generated_rate = scatter_unique(
            _to_momentum_rate_velocity(v_generated, rho_dns, scale, uref), grid,
        )
        row = {
            "time": t, "heun_step": step,
            "standardized_state_difference": paired_state_metrics(teacher, generated),
            "teacher_state": field_summary(
                teacher, ctx.scalers, batch, grid, uref, ctx.fixed_density,
            ),
            "generated_state": field_summary(
                generated, ctx.scalers, batch, grid, uref, ctx.fixed_density,
            ),
            "teacher_fm_momentum_rate_proxy": spectrum_summary(teacher_rate, grid),
            "generated_fm_momentum_rate_proxy": spectrum_summary(generated_rate, grid),
        }
        measurements.append(row)
        print(
            f"[M47.8B] val={args.validation_index} t={t:.2f} "
            f"standardized_rmse={row['standardized_state_difference']['paired_rmse_standardized']:.5g} "
            f"teacher_proxy_fL={row['teacher_state']['fixed_dns_density_momentum_proxy']['longitudinal_fraction']:.5g} "
            f"generated_proxy_fL={row['generated_state']['fixed_dns_density_momentum_proxy']['longitudinal_fraction']:.5g}",
            flush=True,
        )
        for mode in args.modes:
            for eps in args.amplitudes:
                a = _response(
                    model, batch, teacher, t, rho_dns, scale, uref,
                    grid, mode, eps, ctx.reference_rms,
                )
                b = _response(
                    model, batch, generated, t, rho_dns, scale, uref,
                    grid, mode, eps, ctx.reference_rms,
                )
                if max(a["input_mode_leakage_fraction_max"],
                       b["input_mode_leakage_fraction_max"]) > 1e-6:
                    raise ValueError("incorrect injected Fourier mode")
                paired = compare_response(a, b)
                response_pairs.append({
                    "time": t, "mode": list(mode), "relative_amplitude": eps,
                    "teacher": a, "generated": b, "paired_comparison": paired,
                })
                print(
                    f"[M47.8B] t={t:.2f} mode={mode} eps={eps:.3g} "
                    f"Jdelta={paired['matrix_delta_relative']:.4g} "
                    f"coherence={paired['matrix_real_cosine_similarity']}",
                    flush=True,
                )

    result = {
        "protocol": PROTOCOL, "status": "complete",
        "training_signature_sha256": ctx.signature_sha256,
        "training_revision": ctx.training_revision,
        "audit_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
        ).strip(),
        "checkpoint": str(args.train_dir / f"epoch{args.epoch:02d}.pt"),
        "evaluated_epoch": args.epoch,
        "validation_sample": ctx.sample_id,
        "validation_index": args.validation_index,
        "test_data_accessed": False, "matched_initial_gaussian": True,
        "sampling_seed": args.seed, "sampling_key": key, "heun_steps": args.steps,
        "times": list(args.times), "modes": [list(m) for m in args.modes],
        "relative_amplitudes": list(args.amplitudes),
        "dns_component_velocity_rms_m_per_s": ctx.reference_rms,
        "dns_reference": ctx.dns_spectrum,
        "generated_endpoint": field_summary(
            samples[args.steps], ctx.scalers, batch, grid, uref, ctx.fixed_density,
        ),
        "path_definitions": {
            "teacher": "x_t=(1-t)*x0+t*xDNS, same x0 as generation",
            "generated": "Heun dx/dt=v_theta(x,t), initialized from same x0",
        },
        "response_definition": "M47.8A-v2 fixed-DNS-density momentum-rate Jacobian; identical carrier, amplitude, rho_dns and U_ref on both paths",
        "interpretation_warning": "momentum-proxy spectra use fixed rho_DNS, not rho_state; true velocity only where rho_state>0 and internal energy admissibility reported separately",
        "time_metrics": measurements,
        "response_pairs": response_pairs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n",
    )
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--archive-root", type=Path, default=Path(
        "/archivenfs/phlusim/HIT_DNS_COMP",
    ))
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references",
    ))
    p.add_argument("--epoch", type=int, choices=(10, 30), default=30)
    p.add_argument("--validation-index", type=int, default=0)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--times", type=float, nargs="+", default=(0.25, 0.5, 0.75))
    p.add_argument("--modes", type=_mode_arg, nargs="+", default=(
        (2, 2, 0), (4, 3, 1), (8, 6, 5),
    ))
    p.add_argument("--amplitudes", type=float, nargs="+", default=(0.04, 0.08))
    p.add_argument("--seed", type=int, default=20261116)
    outcome = run(p.parse_args())
    print(
        f"[M47.8B] COMPLETE {outcome['validation_sample']} "
        f"times={len(outcome['time_metrics'])} Fourier_pairs={len(outcome['response_pairs'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
