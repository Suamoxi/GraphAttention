"""Fourier-mode linear-response estimators used by M47.8; NumPy-only."""
from __future__ import annotations

import numpy as np


def transfer_column(delta_cos: np.ndarray, delta_sin: np.ndarray,
                    mode: tuple[int, int, int], amplitude: float) -> np.ndarray:
    """Return response to positive complex Fourier mode, output rows xyz.

    delta_cos/sin are central-difference output differences for real inputs
    epsilon*cos(k.x) and epsilon*sin(k.x), shaped [nx,ny,nz,3].
    The combination isolates response to exp(+i*k.x), even if the linearized
    model is not translation-invariant. It is NOT a diagonal Fourier operator
    assumption; off-mode leakage is measured separately.
    """
    if not (np.isfinite(amplitude) and amplitude > 0):
        raise ValueError("amplitude must be positive and finite")
    a = np.asarray(delta_cos, dtype=np.float64)
    b = np.asarray(delta_sin, dtype=np.float64)
    if a.ndim != 4 or a.shape != b.shape or a.shape[-1] != 3:
        raise ValueError("responses must have identical [nx,ny,nz,3] shapes")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("responses must be finite")
    if len(mode) != 3 or all(int(k) == 0 for k in mode):
        raise ValueError("mode must be nonzero three-dimensional index")
    index = tuple(int(m) % n for m, n in zip(mode, a.shape[:3]))
    fa = np.fft.fftn(a, axes=(0, 1, 2), norm="forward")[index]
    fb = np.fft.fftn(b, axes=(0, 1, 2), norm="forward")[index]
    return (fa + 1j * fb) / amplitude


def response_geometry(matrix: np.ndarray, mode: tuple[int, int, int],
                      box_lengths: tuple[float, float, float]) -> dict:
    """Evaluate closeness to longitudinal/transverse projector response."""
    j = np.asarray(matrix, dtype=np.complex128)
    if j.shape != (3, 3) or not np.isfinite(j).all():
        raise ValueError("matrix must have finite shape [3,3]")
    lengths = np.asarray(box_lengths, dtype=float)
    if lengths.shape != (3,) or not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("box lengths must be finite positive three-vector")
    k = np.asarray(mode, dtype=float) / lengths
    if not np.isfinite(k).all() or np.linalg.norm(k) == 0:
        raise ValueError("invalid nonzero physical wavevector")
    k /= np.linalg.norm(k)
    pl = np.outer(k, k)
    pt = np.eye(3) - pl
    gain_long = np.trace(pl @ j)
    gain_trans = np.trace(pt @ j) / 2
    fit = gain_trans * pt + gain_long * pl
    norm = float(np.linalg.norm(j))
    offdiag = j - np.diag(np.diag(j))
    mixing = pt @ j @ pl + pl @ j @ pt
    return {
        "gain_long_real": float(gain_long.real),
        "gain_long_imag": float(gain_long.imag),
        "gain_trans_real": float(gain_trans.real),
        "gain_trans_imag": float(gain_trans.imag),
        "long_trans_gap_relative": float(abs(gain_long - gain_trans) / max(norm, 1e-30)),
        "offdiagonal_relative_frobenius": float(np.linalg.norm(offdiag) / max(norm, 1e-30)),
        "projector_fit_relative_error": float(np.linalg.norm(j - fit) / max(norm, 1e-30)),
        "long_trans_mixing_relative": float(np.linalg.norm(mixing) / max(norm, 1e-30)),
        "matrix_real": j.real.tolist(),
        "matrix_imag": j.imag.tolist(),
    }


def mode_leakage(delta: np.ndarray, mode: tuple[int, int, int]) -> float:
    """Fraction of total output response Fourier energy outside input +/- k."""
    u = np.asarray(delta, dtype=float)
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError("delta must be a vector grid")
    f = np.fft.fftn(u, axes=(0, 1, 2), norm="ortho")
    power = np.sum(np.abs(f) ** 2, axis=-1)
    total = float(power.sum())
    if total == 0:
        return 0.0
    plus = tuple(int(k) % n for k, n in zip(mode, u.shape[:3]))
    minus = tuple(-int(k) % n for k, n in zip(mode, u.shape[:3]))
    retained = float(power[plus]) + (float(power[minus]) if plus != minus else 0.0)
    return float(np.clip((total - retained) / total, 0.0, 1.0))
