"""M45.3: evaluate saved two-scale FM checkpoints on known synthetic distributions.

This is a diagnostic, not DNS/LES validation. Inference uses one fixed coarse
field, many independent noise draws, and the *original* cached coarse encoder.
Only block-average Q/B is supported by the exact Gaussian velocity oracle.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from unittest.mock import patch

import torch

from graph_attention.data.two_scale_synthetic import (
    CASES, SyntheticQB, SyntheticScales, fit_synthetic_scales,
    make_synthetic_pair, sample_conditional_residual,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from graph_attention.models.two_scale_dinat import TwoScaleDiNAT
from scripts.m45_synthetic_pilot import _generate, _state, _to_device, _velocity

CHANNEL_SCALE = (0.018, 0.023, 0.023, 0.023, 0.050)


def _as_blocks(x: torch.Tensor, n: int, r: int) -> torch.Tensor:
    """[n_f**3, channels] -> [n_c**3, channels, r**3]."""
    channels = x.shape[1]
    return x.reshape(n, r, n, r, n, r, channels).permute(0, 2, 4, 6, 1, 3, 5).reshape(n**3, channels, r**3)


def _from_blocks(x: torch.Tensor, n: int, r: int) -> torch.Tensor:
    """Inverse of _as_blocks."""
    channels = x.shape[1]
    return x.reshape(n, n, n, channels, r, r, r).permute(0, 4, 1, 5, 2, 6, 3).reshape((n * r)**3, channels)


class BlockGaussianOracle:
    """Exact conditional-mean FM velocity for synthetic block-average Q/B.

    In each (coarse block, channel), projected local Gaussian noise induces a
    small covariance L. The shared case adds one global rank-one factor gg^T,
    coupled across all blocks *and all five channels*. Solve the conditional
    Gaussian velocity by batched small solves and a rank-one Woodbury update.
    """

    def __init__(self, qb: SyntheticQB, coarse: torch.Tensor,
                 scales: SyntheticScales, case: str, device: torch.device):
        if qb.kind != "block_average":
            raise ValueError("exact Gaussian oracle currently requires block_average Q/B")
        if case not in CASES:            raise ValueError(f"unknown synthetic case: {case}")
        self.nc, self.r, self.case = qb.nc, qb.r, case
        m = qb.r**3
        p = torch.eye(m, device=device, dtype=torch.float64) - torch.ones((m, m), device=device, dtype=torch.float64) / m
        j = torch.ones((m, m), device=device, dtype=torch.float64) / m
        baseline = qb.reconstruct(coarse)
        strength = 0.75 + 0.25 * torch.tanh(20 * (baseline[:, :1] - 1.0))
        strength = _as_blocks(strength.to(device=device, dtype=torch.float64), qb.nc, qb.r)
        channel_scale = torch.tensor(CHANNEL_SCALE, device=device, dtype=torch.float64)
        rms = scales.rms_residual.to(device=device, dtype=torch.float64)
        variances = (strength * (channel_scale / rms)[None, :, None]).square()
        self.local_cov = p @ torch.diag_embed(variances) @ p
        self.p, self.j = p, j
        if case == "shared_mode":
            from graph_attention.data.two_scale_synthetic import shared_mode
            template = shared_mode(qb).to(device=device, dtype=torch.float64)
            self.g = _as_blocks(template * (1.7 * channel_scale / rms)[None, :], qb.nc, qb.r)
        else:
            self.g = None

    def velocity(self, x: torch.Tensor, t: float) -> torch.Tensor:
        if not 0.0 <= t <= 1.0:
            raise ValueError("FM time must be in [0, 1]")
        blocks = _as_blocks(x.to(dtype=torch.float64), self.nc, self.r)
        # Complement J ensures invertibility on the discarded block-mean axis;
        # x, g and the returned velocity all lie in the orthogonal Q-nullspace.
        a = (1.0 - t)**2 * self.p + t**2 * self.local_cov + self.j
        if self.g is None:
            y = torch.linalg.solve(a, blocks.unsqueeze(-1)).squeeze(-1)
        else:
            solutions = torch.linalg.solve(a, torch.stack((blocks, self.g), dim=-1))
            u, z = solutions.unbind(dim=-1)
            numer = (self.g * u).sum()
            denom = 1.0 + t**2 * (self.g * z).sum()
            y = u - z * (t**2 * numer / denom)
        k = t * self.local_cov - (1.0 - t) * self.p
        v = (k @ y.unsqueeze(-1)).squeeze(-1)
        if self.g is not None:
            v = v + t * self.g * (self.g * y).sum()
        return _from_blocks(v, self.nc, self.r).to(dtype=x.dtype)


def _analytic_mode_second_moment(qb: SyntheticQB, coarse: torch.Tensor, case: str) -> float:
    """Exact conditional E[a_phi**2] in physical rhou units for block Q/B."""
    if qb.kind != "block_average":
        raise ValueError("analytic modal statistic requires block_average")
    from graph_attention.data.two_scale_synthetic import shared_mode
    phi = shared_mode(qb)[:, 0]
    denom = phi.square().sum()
    baseline = qb.reconstruct(coarse)
    strength = 0.75 + 0.25 * torch.tanh(20 * (baseline[:, 0] - 1.0))
    # P^T phi = phi for the orthogonal block-average projection.
    local_var = ((phi * strength * CHANNEL_SCALE[1]).square().sum() / denom.square()).item()
    return float(local_var + ((1.7 * CHANNEL_SCALE[1])**2 if case == "shared_mode" else 0.0))


def _ensemble_metrics(samples: torch.Tensor, template: torch.Tensor, qb: SyntheticQB) -> dict:
    """Compare identical pooled moment estimators for reference and generation."""
    if samples.ndim != 3 or samples.shape[1:] != (qb.nf**3, 5):
        raise ValueError("expected [n_samples, fine_nodes, 5]")
    denominator = template.square().sum()
    amplitudes = samples[:, :, 1] @ template / denominator
    defects = [qb.restrict(z).abs().max().item() for z in samples]
    pooled_rms = samples.square().mean(dim=(0, 1)).sqrt()
    return {
        "samples": samples.shape[0],
        "mode_mean": amplitudes.mean().item(),
        "mode_second_moment": amplitudes.square().mean().item(),
        "mode_variance": amplitudes.var(unbiased=samples.shape[0] > 1).item(),
        "channel_rms": pooled_rms.tolist(),
        "max_Q_residual": max(defects),
    }


def _correlations(samples: torch.Tensor, nf: int) -> list[dict]:
    """Conditional covariance of rhou at axis-aligned periodic spatial lags."""
    centered = samples[:, :, 1] - samples[:, :, 1].mean(dim=0, keepdim=True)
    field = centered.reshape(samples.shape[0], nf, nf, nf)
    correction = samples.shape[0] / (samples.shape[0] - 1) if samples.shape[0] > 1 else 1.0
    zero = float(field.square().mean() * correction)
    rows = []
    for axis, name in enumerate(("x", "y", "z"), start=1):
        for lag in range(nf // 2 + 1):
            covariance = float((field * field.roll(shifts=-lag, dims=axis)).mean() * correction)
            rows.append({"axis": name, "lag": lag, "distance_fraction_L": lag / nf,
                         "covariance": covariance,
                         "correlation": covariance / zero if zero else 0.0})
    return rows


def _write_csv(path: Path, records: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _aggregate_time(records: list[dict]) -> list[dict]:
    """Mean and coarse-realization-clustered SEM for each case/mode/FM time."""
    from collections import defaultdict
    import math
    from statistics import mean, stdev

    keys = ("fm_mse_raw", "fm_mse_projected", "oracle_fm_mse",
            "excess_mse_vs_oracle", "mode_velocity_mse", "oracle_mode_velocity_mse")
    grouped = defaultdict(list)
    for row in records:
        grouped[(row["case"], row["mode"], row["t"], row["sample_id"])].append(row)
    per_sample = defaultdict(list)
    for (case, mode, t, _), rows in grouped.items():
        per_sample[(case, mode, t)].append({k: mean(r[k] for r in rows) for k in keys})
    result = []
    for (case, mode, t), rows in sorted(per_sample.items()):
        out = {"case": case, "mode": mode, "t": t, "validation_coarse_fields": len(rows)}
        for key in keys:
            vals = [r[key] for r in rows]
            out[key + "_mean"] = mean(vals)
            out[key + "_sem"] = stdev(vals) / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        result.append(out)
    return result


def _load_model(path: Path, settings: dict, device: torch.device) -> tuple[TwoScaleDiNAT, SyntheticScales]:
    # Only load locally produced, trusted checkpoints. The dataclass is safely
    # allowlisted while using the restricted torch.load weights-only unpickler.
    with torch.serialization.safe_globals([SyntheticScales]):
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = checkpoint["model_config"]
    model = TwoScaleDiNAT(
        mode=config["mode"], coarse_width=settings["width"],
        fine_width=settings["width"], num_heads=settings["heads"],
        fine_pre_layers=config["fine_pre_layers"],
        fine_post_layers=config["fine_post_layers"],
        coarse_static_layers=config["coarse_static_layers"],
        coarse_dynamic_layers=max(1, config["coarse_dynamic_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    return model, checkpoint["scales"]


def _fixed_coarse_references(qb: SyntheticQB, coarse: torch.Tensor, case: str,
                             *, seed: int, count: int) -> torch.Tensor:
    return torch.stack([
        sample_conditional_residual(qb, coarse, case=case,
            generator=torch.Generator(device="cpu").manual_seed(seed + 1009 * i))[0]
        for i in range(count)
    ])


def _time_metrics(model: TwoScaleDiNAT, qb: SyntheticQB, qb_gpu: SyntheticQB,
                  mesh_gpu, val_pairs, scales: SyntheticScales,
                  *, case: str, times: list[float], draws: int, seed: int,
                  device: torch.device) -> list[dict]:
    rows = []
    for pair in val_pairs:
        oracle = BlockGaussianOracle(qb, pair.coarse, scales, case, device)
        c, b, x1 = _state(pair, scales)
        c, b, x1 = c.to(device), b.to(device), x1.to(device)
        phi = pair.mode[:, 0].to(device)
        denom = phi.square().sum()
        for draw in range(draws):
            gen = torch.Generator(device="cpu").manual_seed(seed + pair.sample_id * 103 + draw * 179)
            eps = qb.nullspace(torch.randn(pair.residual.shape, generator=gen)).to(device)
            target = x1 - eps
            for t in times:
                xt = (1 - t) * eps + t * x1
                prediction = _velocity(model, xt, b, c, mesh_gpu, xt.new_tensor([t]))
                constrained = qb_gpu.nullspace(prediction)
                optimal = oracle.velocity(xt, t)
                raw_error = prediction - target
                projected_error = constrained - target
                excess_error = constrained - optimal
                optimal_error = optimal - target
                rows.append({
                    "case": case, "mode": model.mode,
                    "sample_id": pair.sample_id, "noise_draw": draw, "t": t,
                    "fm_mse_raw": raw_error.square().mean().item(),
                    "fm_mse_projected": projected_error.square().mean().item(),
                    "oracle_fm_mse": optimal_error.square().mean().item(),
                    "excess_mse_vs_oracle": excess_error.square().mean().item(),
                    "mode_velocity_mse": ((projected_error[:, 1] * phi).sum() / denom).square().item(),
                    "oracle_mode_velocity_mse": ((optimal_error[:, 1] * phi).sum() / denom).square().item(),
                })
    return rows


@torch.no_grad()
def _generate_oracle(oracle: BlockGaussianOracle, qb: SyntheticQB,
                     scales: SyntheticScales, *, device: torch.device,
                     seed: int, steps: int) -> torch.Tensor:
    """Positive-control FM sampler: integrate the *exact* Gaussian velocity.

    Uses the same projected Gaussian source and Heun formula as neural
    inference, then returns the residual in original synthetic units.
    """
    if steps < 1:
        raise ValueError("Heun steps must be positive")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = qb.nullspace(torch.randn((qb.nf**3, 5), generator=gen)).to(device)
    dt = 1.0 / steps
    for n in range(steps):
        t = n * dt
        k1 = oracle.velocity(x, t)
        xp = x + dt * k1
        k2 = oracle.velocity(xp, (n + 1) * dt)
        x = x + 0.5 * dt * (k1 + k2)
    return (x.cpu() * scales.rms_residual).contiguous()


def _generate_ensemble(model: TwoScaleDiNAT, mesh_gpu, qb_gpu, pair, scales,
                       *, device, seeds: list[int], steps: int, feedback_off: bool) -> torch.Tensor:
    def _run() -> torch.Tensor:
        return torch.stack([
            _generate(model, mesh_gpu, qb_gpu, pair, scales, device=device,
                      seed=seed, heun_steps=steps, projected=True)
            for seed in seeds
        ])
    if not feedback_off:
        return _run()
    if model.mode != "dynamic":
        raise ValueError("fine-to-coarse ablation is available only for dynamic")
    # Keep the trained dynamic coarse processor and its time conditioning.
    # Only remove the *incoming fine feedback* at evaluation time.
    with patch.object(model.down, "forward", side_effect=lambda queries, keys, **kwargs: torch.zeros_like(queries)):
        return _run()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--modes", nargs="+", choices=("static", "dynamic"), default=["static", "dynamic"])
    parser.add_argument("--val-index", type=int, default=0)
    parser.add_argument("--generated-samples", type=int, default=128)
    parser.add_argument("--heun-steps", type=int, nargs="+", default=[8, 25, 50])
    parser.add_argument("--eval-times", type=float, nargs="+", default=[0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    parser.add_argument("--velocity-noise-draws", type=int, default=4)
    parser.add_argument("--oracle-sampling", action="store_true",
                        help="Heun-integrate exact conditional Gaussian FM velocity as positive control")
    parser.add_argument("--oracle-heun-steps", type=int, nargs="+", default=None,
                        help="Separate step list for exact-oracle Heun sampling")
    parser.add_argument("--seed", type=int, default=53177)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("refusing to overwrite output directory")
    if args.generated_samples < 2 or args.velocity_noise_draws < 1 or args.seed < 0:
        parser.error("generated-samples >= 2, velocity-noise-draws >= 1, and seed >= 0 required")
    if (any(x < 1 for x in args.heun_steps)
            or (args.oracle_heun_steps is not None and any(x < 1 for x in args.oracle_heun_steps))
            or any(not 0 <= x <= 1 for x in args.eval_times)):
        parser.error("Heun steps must be positive and evaluation times must be in [0,1]")
    summary = json.loads((args.checkpoint_root / "summary.json").read_text())
    settings = summary["settings"]
    if settings["operator"] != "block_average" or settings["source_subspace"] != "nullspace":
        parser.error("M45.3 exact Gaussian oracle currently requires block_average and projected source")
    if not 0 <= args.val_index < settings["val_samples"]:
        parser.error("val-index outside the saved validation split")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = torch.device(args.device)
    mesh = build_hit_two_scale_geometry(coarse_side=settings["coarse_side"],
                                       refinement=settings["refinement"])
    mesh_gpu = _to_device(mesh, device)
    qb = SyntheticQB(mesh, settings["operator"])
    qb_gpu = SyntheticQB(mesh_gpu, settings["operator"])
    train_seed = settings["seed"]
    val_seed = settings.get("val_seed")
    if val_seed is None:
        val_seed = train_seed + 7_000_000
    args.output_dir.mkdir(parents=True)
    timing, generation, correlations, amplitudes, references = [], [], [], [], []
    for case in args.cases:
        pairs = [make_synthetic_pair(qb, case=case, sample_id=i + 100000,
                 seed=val_seed + 3001 * i) for i in range(settings["val_samples"])]
        fixed = pairs[args.val_index]
        template = fixed.mode[:, 0]
        reference_samples = _fixed_coarse_references(qb, fixed.coarse, case, seed=args.seed + 700000,
                                                     count=args.generated_samples)
        ref_metrics = _ensemble_metrics(reference_samples, template, qb)
        ref_metrics.update({"case": case, "mode": "reference", "variant": "synthetic",
                            "heun_steps": 0,
                            "analytic_mode_second_moment": _analytic_mode_second_moment(qb, fixed.coarse, case)})
        references.append(ref_metrics)
        correlations.extend({"case": case, "mode": "reference", "variant": "synthetic", "heun_steps": 0, **r}
                            for r in _correlations(reference_samples, qb.nf))
        for draw, sample in enumerate(reference_samples):
            amplitudes.append({"case": case, "mode": "reference", "variant": "synthetic",
                               "heun_steps": 0, "draw": draw,
                               "amplitude": float(sample[:, 1] @ template / template.square().sum())})
        print(f"M45.3 reference case={case} analytic_M2={ref_metrics['analytic_mode_second_moment']:.6g}", flush=True)
        for mode in args.modes:
            checkpoint = args.checkpoint_root / case / settings["operator"] / mode / "checkpoint.pt"
            model, scales = _load_model(checkpoint, settings, device)
            # Compare against exactly the same fixed validation paths for both models.
            with torch.no_grad():
                timing.extend(_time_metrics(model, qb, qb_gpu, mesh_gpu, pairs, scales,
                              case=case, times=args.eval_times, draws=args.velocity_noise_draws,
                              seed=args.seed, device=device))
                seeds = [args.seed + 1_000_000 + 3277 * i for i in range(args.generated_samples)]
                if args.oracle_sampling and mode == args.modes[0]:
                    oracle = BlockGaussianOracle(qb, fixed.coarse, scales, case, device)
                    for steps in (args.oracle_heun_steps or args.heun_steps):
                        ensemble = torch.stack([
                            _generate_oracle(oracle, qb, scales, device=device,
                                             seed=seed, steps=steps) for seed in seeds
                        ])
                        metrics = _ensemble_metrics(ensemble, template, qb)
                        metrics.update({"case": case, "mode": "oracle", "variant": "exact_velocity",
                                        "heun_steps": steps,
                                        "analytic_mode_second_moment": ref_metrics["analytic_mode_second_moment"]})
                        generation.append(metrics)
                        correlations.extend({"case": case, "mode": "oracle", "variant": "exact_velocity",
                                             "heun_steps": steps, **r}
                                            for r in _correlations(ensemble, qb.nf))
                        denominator = template.square().sum()
                        for draw, sample in enumerate(ensemble):
                            amplitudes.append({"case": case, "mode": "oracle", "variant": "exact_velocity",
                                               "heun_steps": steps, "draw": draw,
                                               "amplitude": float(sample[:, 1] @ template / denominator)})
                        print(f"case={case} oracle Heun={steps}: "
                              f"mode_M2={metrics['mode_second_moment']:.6g}", flush=True)
                for steps in args.heun_steps:
                    variants = ["normal"]
                    if mode == "dynamic" and steps == max(args.heun_steps):
                        variants.append("feedback_off")
                    for variant in variants:
                        ensemble = _generate_ensemble(model, mesh_gpu, qb_gpu, fixed, scales,
                                     device=device, seeds=seeds, steps=steps,
                                     feedback_off=variant == "feedback_off")
                        metrics = _ensemble_metrics(ensemble, template, qb)
                        metrics.update({"case": case, "mode": mode, "variant": variant,
                                        "heun_steps": steps,
                                        "analytic_mode_second_moment": ref_metrics["analytic_mode_second_moment"]})
                        generation.append(metrics)
                        correlations.extend({"case": case, "mode": mode, "variant": variant,
                                             "heun_steps": steps, **r}
                                            for r in _correlations(ensemble, qb.nf))
                        denominator = template.square().sum()
                        for draw, sample in enumerate(ensemble):
                            amplitudes.append({"case": case, "mode": mode, "variant": variant,
                                               "heun_steps": steps, "draw": draw,
                                               "amplitude": float(sample[:, 1] @ template / denominator)})
                        print(f"case={case} mode={mode} steps={steps} {variant}: "
                              f"mode_M2={metrics['mode_second_moment']:.6g}", flush=True)
    _write_csv(args.output_dir / "time_resolved.csv", timing,
               ["case", "mode", "sample_id", "noise_draw", "t", "fm_mse_raw",
                "fm_mse_projected", "oracle_fm_mse", "excess_mse_vs_oracle",
                "mode_velocity_mse", "oracle_mode_velocity_mse"])
    time_aggregate = _aggregate_time(timing)
    _write_csv(args.output_dir / "time_summary.csv", time_aggregate,
               ["case", "mode", "t", "validation_coarse_fields"] +
               [key + suffix for key in ("fm_mse_raw", "fm_mse_projected",
                  "oracle_fm_mse", "excess_mse_vs_oracle", "mode_velocity_mse",
                  "oracle_mode_velocity_mse") for suffix in ("_mean", "_sem")])
    _write_csv(args.output_dir / "generation_metrics.csv", generation + references,
               ["case", "mode", "variant", "heun_steps", "samples", "mode_mean",
                "mode_second_moment", "mode_variance", "analytic_mode_second_moment",
                "channel_rms", "max_Q_residual"])
    _write_csv(args.output_dir / "spatial_correlations.csv", correlations,
               ["case", "mode", "variant", "heun_steps", "axis", "lag",
                "distance_fraction_L", "covariance", "correlation"])
    _write_csv(args.output_dir / "mode_amplitudes.csv", amplitudes,
               ["case", "mode", "variant", "heun_steps", "draw", "amplitude"])
    report = {"schema": "graph_attention.m45_checkpoint_diagnostics.v1",
              "scope": "synthetic block-average conditional generation, NOT physical LES/DNS",
              "checkpoint_root": str(args.checkpoint_root), "settings": {**vars(args),
              "checkpoint_root": str(args.checkpoint_root), "output_dir": str(args.output_dir)},
              "source_settings": settings, "reference_metrics": references,
              "generation_metrics": generation,
              "time_records": len(timing), "correlation_records": len(correlations)}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"M45.3 diagnostics saved to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
