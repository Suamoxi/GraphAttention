"""M47.8 Gaussian full-field positive control; CPU NumPy, no DNS or model weights.

The target has a prescribed longitudinal spectral fraction. Independent unit
Gaussian source and the exact conditional straight-path FM velocity are used.
This tests the formulation and the numerical Heun integrator, NOT DiNAT learning.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def operators(side: int) -> tuple[np.ndarray, np.ndarray]:
    if side < 5 or side % 2 != 1:
        raise ValueError("side must be odd and at least 5 (avoid Nyquist ambiguity)")
    k = np.meshgrid(*(np.fft.fftfreq(side) for _ in range(3)), indexing="ij")
    k2 = sum(a * a for a in k)
    khat = np.stack(k, axis=-1) / np.sqrt(np.maximum(k2, 1e-40))[..., None]
    active = k2 > 0
    return khat, active


def split_fourier(vhat: np.ndarray, khat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    long = np.sum(vhat * khat, axis=-1, keepdims=True) * khat
    return vhat - long, long


def target_spectrum_fraction(f_long: float) -> tuple[float, float]:
    if not 0 < f_long < 1:
        raise ValueError("target fraction must lie strictly between zero and one")
    # 2 transverse eigenvalues and one longitudinal; total trace = 3.
    return 1.5 * (1.0 - f_long), 3.0 * f_long


def direct_transport(source: np.ndarray, khat: np.ndarray, eigenvalues: tuple[float, float]) -> np.ndarray:
    spectrum = np.fft.fftn(source, axes=(0, 1, 2), norm="ortho")
    transverse, longitudinal = split_fourier(spectrum, khat)
    a_s, a_d = eigenvalues
    out = np.fft.ifftn(np.sqrt(a_s) * transverse + np.sqrt(a_d) * longitudinal, axes=(0, 1, 2), norm="ortho")
    return out.real


def exact_fm_velocity(state: np.ndarray, t: float, khat: np.ndarray, eigenvalues: tuple[float, float]) -> np.ndarray:
    spectrum = np.fft.fftn(state, axes=(0, 1, 2), norm="ortho")
    transverse, longitudinal = split_fourier(spectrum, khat)
    a_s, a_d = eigenvalues
    gain_s = (t * a_s - (1 - t)) / (t * t * a_s + (1 - t) ** 2)
    gain_d = (t * a_d - (1 - t)) / (t * t * a_d + (1 - t) ** 2)
    out = np.fft.ifftn(gain_s * transverse + gain_d * longitudinal, axes=(0, 1, 2), norm="ortho")
    return out.real


def integrate(source: np.ndarray, steps: int, khat: np.ndarray, eigenvalues: tuple[float, float]) -> np.ndarray:
    if steps < 1:
        raise ValueError("steps must be positive")
    dt = 1.0 / steps
    x = source.copy()
    for j in range(steps):
        t = j * dt
        a = exact_fm_velocity(x, t, khat, eigenvalues)
        b = exact_fm_velocity(x + dt * a, t + dt, khat, eigenvalues)
        x += 0.5 * dt * (a + b)
    return x


def longitudinal_fraction(state: np.ndarray, khat: np.ndarray, active: np.ndarray) -> float:
    spectrum = np.fft.fftn(state, axes=(0, 1, 2), norm="ortho")
    _, longitudinal = split_fourier(spectrum, khat)
    return float(np.sum(np.abs(longitudinal[active]) ** 2) / np.sum(np.abs(spectrum[active]) ** 2))


def run(*, side: int = 33, f_long: float = 0.018731345684936694,
        steps: tuple[int, ...] = (25, 50, 100), samples: int = 3,
        seed: int = 47) -> dict:
    if samples < 1:
        raise ValueError("samples must be positive")
    khat, active = operators(side)
    eigenvalues = target_spectrum_fraction(f_long)
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(samples):
        source = rng.standard_normal((side, side, side, 3))
        direct = direct_transport(source, khat, eigenvalues)
        source_fraction = longitudinal_fraction(source, khat, active)
        direct_fraction = longitudinal_fraction(direct, khat, active)
        for count in steps:
            numerical = integrate(source, count, khat, eigenvalues)
            error = np.sqrt(np.mean((numerical - direct) ** 2) / np.mean(direct ** 2))
            rows.append({"sample": idx, "heun_steps": count,
                         "source_longitudinal_fraction": source_fraction,
                         "direct_longitudinal_fraction": direct_fraction,
                         "heun_longitudinal_fraction": longitudinal_fraction(numerical, khat, active),
                         "paired_relative_rmse": float(error)})
    return {"protocol": "m47_8_full_field_gaussian_oracle_v1",
            "description": "3D periodic odd-grid, flat-spectrum isotropic Gaussian with prescribed longitudinal/transverse covariance; exact unconditional straight-path FM velocity; Heun ODE",
            "side": side, "seed": seed, "samples": samples,
            "target_longitudinal_fraction": f_long,
            "covariance_eigenvalues": {"transverse": eigenvalues[0], "longitudinal": eigenvalues[1]},
            "records": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=33)
    parser.add_argument("--fraction", type=float, default=0.018731345684936694)
    parser.add_argument("--heun-steps", type=int, nargs="+", default=(25, 50, 100))
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = run(side=args.side, f_long=args.fraction, steps=tuple(args.heun_steps),
                 samples=args.samples, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for count in args.heun_steps:
        rows = [r for r in report["records"] if r["heun_steps"] == count]
        print(f"Heun {count}: source fL={np.mean([r['source_longitudinal_fraction'] for r in rows]):.5f}, "
              f"oracle fL={np.mean([r['direct_longitudinal_fraction'] for r in rows]):.5f}, "
              f"generated fL={np.mean([r['heun_longitudinal_fraction'] for r in rows]):.5f}, "
              f"paired RMSE={np.mean([r['paired_relative_rmse'] for r in rows]):.5g}")


if __name__ == "__main__":
    main()
