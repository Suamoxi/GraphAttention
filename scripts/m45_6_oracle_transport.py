"""M45.6 analytical Gaussian transport reference and Heun-convergence audit.

Synthetic block-average Q/B only. The direct map is the canonical symmetric
square-root transport from projected N(0, I) to the exact conditional Gaussian;
it uses the same initial Gaussian as the Heun-integrated optimal FM velocity.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from graph_attention.data.two_scale_synthetic import (
    CASES, SyntheticQB, fit_synthetic_scales, make_synthetic_pair,
)
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_checkpoint_diagnostics import (
    BlockGaussianOracle, _analytic_mode_second_moment, _as_blocks,
    _from_blocks, _generate_oracle,
)


class CanonicalGaussianTransport:
    """Apply Sigma_target**(1/2) on ker(Q) without a dense fine-grid matrix.

    For block averaging, local covariance equals lambda_b P in each spatial
    block/channel b. The shared contribution is one rank-one vector g g^T.
    Only one direction per block interacts with g. Thus a symmetric
    (Ncoarse*5) square root suffices; all orthogonal directions scale by
    sqrt(lambda_b). This is the exact FM ODE endpoint, not merely another
    map with the same final covariance.
    """

    def __init__(self, oracle: BlockGaussianOracle):
        n, r = oracle.nc, oracle.r
        self.nc, self.r = n, r
        m = r**3
        cov = oracle.local_cov.to(dtype=torch.float64)
        lam = cov.diagonal(dim1=-2, dim2=-1).sum(-1) / (m - 1)
        if not torch.all(lam > 0):
            raise ValueError("nonpositive local nullspace covariance")
        if not torch.allclose(cov, lam[..., None, None] * oracle.p,
                              rtol=2e-10, atol=2e-10):
            raise ValueError("direct transport needs block-isotropic local covariance")
        self.sqrt_local = lam.sqrt()
        if oracle.g is None:
            self.direction = None
            self.sqrt_coupled = None
        else:
            g = oracle.g.to(dtype=torch.float64)
            strength = torch.linalg.vector_norm(g, dim=-1)
            direction = g / strength.clamp_min(1e-30)[..., None]
            fallback = torch.zeros_like(direction)
            fallback[..., 0] = 2**-0.5
            fallback[..., 1] = -(2**-0.5)
            self.direction = torch.where(strength[..., None] > 1e-15,
                                         direction, fallback)
            d = lam.flatten()
            h = strength.flatten()
            covariance = torch.diag(d) + torch.outer(h, h)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
            if eigenvalues.min() <= 0:
                raise ValueError("target covariance is not positive on ker(Q)")
            self.sqrt_coupled = (eigenvectors * eigenvalues.sqrt()) @ eigenvectors.T

    @torch.no_grad()
    def apply(self, source: torch.Tensor) -> torch.Tensor:
        """source: normalized, projected fine Gaussian [Nf,5]."""
        blocks = _as_blocks(source.to(device=self.sqrt_local.device, dtype=torch.float64),
                            self.nc, self.r)
        if self.direction is None:
            result = self.sqrt_local[..., None] * blocks
        else:
            z = (blocks * self.direction).sum(-1)
            updated = (self.sqrt_coupled @ z.flatten()).reshape_as(z)
            result = (self.sqrt_local[..., None] *
                      (blocks - self.direction * z[..., None]) +
                      self.direction * updated[..., None])
        return _from_blocks(result, self.nc, self.r)


def _source(qb: SyntheticQB, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return qb.nullspace(torch.randn((qb.nf**3, 5), generator=generator))


def _moment(samples: torch.Tensor, template: torch.Tensor, qb: SyntheticQB) -> dict:
    coefficients = (samples[:, :, 1] @ template) / template.square().sum()
    return {
        "mode_second_moment": float(coefficients.square().mean()),
        "rhou_rms": float(samples[:, :, 1].square().mean().sqrt()),
        "max_Q_residual": max(float(qb.restrict(x).abs().max()) for x in samples),
    }


def run_oracle(*, output_dir: Path, coarse_side: int, refinement: int,
               train_samples: int, seed: int, validation_seed: int, val_index: int,
               cases: tuple[str, ...], heun_steps: tuple[int, ...],
               draws: int, direct_draws: int, device: torch.device) -> list[dict]:
    if output_dir.exists():
        raise FileExistsError(f"refusing existing output: {output_dir}")
    if not 0 <= val_index < 8 or draws < 2 or direct_draws < draws:
        raise ValueError("val-index must be 0..7; require 2 <= draws <= direct-draws")
    mesh = build_hit_two_scale_geometry(coarse_side=coarse_side, refinement=refinement)
    qb = SyntheticQB(mesh, "block_average")
    output_dir.mkdir(parents=True)
    all_rows = []
    for case in cases:
        train = [make_synthetic_pair(qb, case=case, sample_id=i,
                                    seed=seed + i * 3001) for i in range(train_samples)]
        scales = fit_synthetic_scales(train)
        pair = make_synthetic_pair(qb, case=case, sample_id=100000 + val_index,
                                   seed=validation_seed + val_index * 3001)
        oracle = BlockGaussianOracle(qb, pair.coarse, scales, case, device)
        direct_map = CanonicalGaussianTransport(oracle)
        template = pair.mode[:, 0]
        analytic = _analytic_mode_second_moment(qb, pair.coarse, case)
        seeds = [53177 + 1_000_000 + i * 3277 for i in range(direct_draws)]
        direct = torch.stack([            (direct_map.apply(_source(qb, s)).cpu() * scales.rms_residual).float()
            for s in seeds
        ])
        direct_paired = direct[:draws]
        base = {"case": case, "analytic_mode_second_moment": analytic}
        for kind, sample in (("direct", direct), ("direct_paired", direct_paired)):
            all_rows.append({**base, "method": kind, "heun_steps": 0,
                             "samples": len(sample), **_moment(sample, template, qb),
                             "paired_rmse_relative": 0.0,
                             "paired_mode_rmse": 0.0})
        for count in heun_steps:
            outputs = torch.stack([
                _generate_oracle(oracle, qb, scales, device=device,
                                 seed=s, steps=count) for s in seeds[:draws]
            ])
            denom = direct_paired.square().mean().sqrt().clamp_min(1e-12)
            relative_rmse = float((outputs - direct_paired).square().mean().sqrt() / denom)
            amps = (outputs[:, :, 1] - direct_paired[:, :, 1]) @ template / template.square().sum()
            all_rows.append({**base, "method": "exact_velocity_heun", "heun_steps": count,
                             "samples": draws, **_moment(outputs, template, qb),
                             "paired_rmse_relative": relative_rmse,
                             "paired_mode_rmse": float(amps.square().mean().sqrt())})
            print(f"{case}: Heun={count} paired relative RMSE={relative_rmse:.6g}", flush=True)
    keys = list(all_rows[0])
    with (output_dir / "oracle_transport.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(all_rows)
    (output_dir / "summary.json").write_text(json.dumps({
        "schema": "graph_attention.m45_6_oracle_transport.v1",
        "scope": "synthetic block-average Gaussian reference, not CFD validation",
        "settings": {"coarse_side": coarse_side, "refinement": refinement,
                     "train_samples": train_samples, "seed": seed,
                     "validation_seed": validation_seed, "val_index": val_index,
                     "cases": cases, "heun_steps": heun_steps, "draws": draws,
                     "direct_draws": direct_draws, "device": str(device)},
        "records": all_rows,
    }, indent=2) + "\n")
    return all_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--coarse-side", type=int, default=4)
    parser.add_argument("--refinement", type=int, default=2)
    parser.add_argument("--train-samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--validation-seed", type=int, default=7002026)
    parser.add_argument("--val-index", type=int, default=0)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--heun-steps", nargs="+", type=int, default=[25, 50, 100, 200, 400, 800])
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--direct-draws", type=int, default=512)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (min(args.coarse_side, args.refinement, args.train_samples,
            args.draws, args.direct_draws) < 1 or any(x < 1 for x in args.heun_steps)):
        parser.error("positive sizes, draw counts, and Heun steps required")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    run_oracle(output_dir=args.output_dir, coarse_side=args.coarse_side,
               refinement=args.refinement, train_samples=args.train_samples,
               seed=args.seed, validation_seed=args.validation_seed,
               val_index=args.val_index, cases=tuple(args.cases),
               heun_steps=tuple(args.heun_steps), draws=args.draws,
               direct_draws=args.direct_draws, device=torch.device(args.device))


if __name__ == "__main__":
    main()
