"""M47.8B: mathematical diagnostics for matched teacher and generated FM paths."""
from __future__ import annotations

import math

import numpy as np
import torch

from scripts.m47_8_response_audit import CHANNELS
from scripts.m47_6_spatial_metrics import scatter_unique
from scripts.m47_7_covariance_metrics import spectral_tensor_audit


def exact_steps(times: tuple[float, ...], steps: int) -> tuple[int, ...]:
    if type(steps) is not int or steps <= 0 or not times or len(set(times)) != len(times):
        raise ValueError("positive step count and unique nonempty times required")
    out = []
    for t in times:
        if not math.isfinite(t) or not 0 < t < 1:
            raise ValueError("FM times must lie strictly inside (0,1)")
        index = round(t * steps)
        if not math.isclose(index / steps, t, abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"t={t} is not on the {steps}-step Heun grid")
        out.append(index)
    return tuple(out)


def paired_state_metrics(teacher: torch.Tensor, generated: torch.Tensor) -> dict:
    if teacher.shape != generated.shape or teacher.ndim != 2 or teacher.shape[1] != len(CHANNELS):
        raise ValueError("states must share shape [N,5]")
    if not bool(torch.isfinite(teacher).all() and torch.isfinite(generated).all()):
        raise ValueError("nonfinite path state")
    a, b = teacher.double(), generated.double()
    difference = b - a
    rms = float(a.square().mean().sqrt().item())
    error = float(difference.square().mean().sqrt().item())
    return {
        "teacher_rms_standardized": rms,
        "generated_rms_standardized": float(b.square().mean().sqrt().item()),
        "paired_rmse_standardized": error,
        "paired_rmse_over_teacher_rms": error / max(rms, 1e-30),
        "paired_mse_by_channel": dict(zip(CHANNELS, difference.square().mean(0).cpu().tolist())),
    }


def compare_response(a: dict, b: dict) -> dict:
    """Complex Fourier response differences; a=teacher, b=generated."""
    for key in ("mode", "time", "relative_amplitude"):
        if a[key] != b[key]:
            raise ValueError("response parameters must match")
    a_complex = np.asarray(a["matrix_real"]) + 1j * np.asarray(a["matrix_imag"])
    b_complex = np.asarray(b["matrix_real"]) + 1j * np.asarray(b["matrix_imag"])
    if (a_complex.shape != (3, 3) or b_complex.shape != (3, 3)
        or not np.isfinite(a_complex).all() or not np.isfinite(b_complex).all()):
        raise ValueError("invalid complex 3x3 responses")
    norm_a, norm_b = float(np.linalg.norm(a_complex)), float(np.linalg.norm(b_complex))
    delta = float(np.linalg.norm(b_complex - a_complex))
    cosine = (float(np.real(np.vdot(a_complex, b_complex)) / (norm_a * norm_b))
              if norm_a * norm_b > 1e-28 else None)
    gain = lambda x, k: complex(x[f"gain_{k}_real"], x[f"gain_{k}_imag"])
    return {
        "teacher_matrix_frobenius": norm_a,
        "generated_matrix_frobenius": norm_b,
        "matrix_delta_frobenius": delta,
        "matrix_delta_relative": delta / max(norm_a, norm_b, 1e-30),
        "matrix_real_cosine_similarity": cosine,
        "longitudinal_complex_gain_difference_abs": float(abs(gain(b,"long") - gain(a,"long"))),
        "transverse_complex_gain_difference_abs": float(abs(gain(b,"trans") - gain(a,"trans"))),
        "response_leakage_difference": (b["response_leakage_fraction_mean"] -
                                        a["response_leakage_fraction_mean"]),
    }


def spectrum_summary(velocity: np.ndarray, grid) -> dict:
    report = spectral_tensor_audit(velocity, grid)
    e = np.asarray(report["E_total_m3_per_s2"])
    l = np.asarray(report["E_longitudinal_m3_per_s2"])
    return {
        "longitudinal_fraction": report["longitudinal_energy_fraction_integrated"],
        "cross_cancellation": report["cross_cancellation_integrated"],
        "sum_shell_spectral_density_m3_per_s2": float(e.sum()),
        "first_two_shells_longitudinal_fraction": (
            float(l[:2].sum() / e[:2].sum()) if e[:2].sum() > 0 else None
        ),
    }


def field_summary(state: torch.Tensor, scalers, prepared, grid,
                  u_ref: float, fixed_density: np.ndarray) -> dict:
    """True velocity only if rho>0; fixed-density momentum proxy in all cases."""
    q = scalers.inputs.inverse(state, prepared.input_channels).detach().cpu().numpy()
    unique = scatter_unique(q, grid)
    if not np.isfinite(unique).all():
        raise ValueError("nonfinite inverse-standardized field")
    rho = unique[..., 0]
    invalid_density = float(np.mean(rho <= 0))
    proxy = u_ref * unique[..., 1:4] / fixed_density[..., None]
    result = {
        "nonpositive_density_fraction": invalid_density,
        "nonpositive_internal_energy_fraction": None,
        "fixed_dns_density_momentum_proxy": spectrum_summary(proxy, grid),
        "actual_velocity_if_density_positive": None,
    }
    if invalid_density == 0:
        internal = unique[..., 4] - 0.5 * (unique[..., 1:4] ** 2).sum(-1) / rho
        result["nonpositive_internal_energy_fraction"] = float(np.mean(internal <= 0))
        result["actual_velocity_if_density_positive"] = spectrum_summary(
            u_ref * unique[..., 1:4] / rho[..., None], grid
        )
    return result
