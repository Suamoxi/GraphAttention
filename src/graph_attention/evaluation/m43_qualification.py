"""Conservative, review-gated statistical stationarity diagnostics for M43 HIT.

All indices are zero-based. The output is *evidence for review*, never a
mathematical proof of stationarity or an automatic training-data filter.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


STATIONARITY_METRICS = (
    "turbulent_k",
    "enstrophy",
    "std_rho",
    "f4_spectral_unresolved_energy_fraction",
    "spectrum_low_fraction",
    "spectrum_mid_fraction",
    "spectrum_high_fraction",
)


@dataclass(frozen=True)
class QualificationConfig:
    window: int = 12
    anchor_windows: int = 3
    minimum_late_windows: int = 3
    relative_tolerance: float = 0.22
    maximum_relative_drift: float = 0.25
    transition_width: int = 8
    bootstrap_repeats: int = 256
    seed: int = 43

    def __post_init__(self) -> None:
        if min(self.window, self.anchor_windows, self.minimum_late_windows) < 2:
            raise ValueError("window, anchor_windows and minimum_late_windows must be >=2")
        if not 0 < self.relative_tolerance < 1:
            raise ValueError("relative_tolerance must be in (0,1)")
        if not 0 < self.maximum_relative_drift < 1:
            raise ValueError("maximum_relative_drift must be in (0,1)")
        if self.transition_width < 0 or self.bootstrap_repeats < 16:
            raise ValueError("invalid transition width or bootstrap count")


def spectral_energy_fractions(k: np.ndarray, e: np.ndarray) -> dict[str, float]:
    """Spectral energy fractions by integer multiples of fundamental wavenumber.

    Reuses cached shell energy density E(k). All three bins together cover the
    available nonzero shells; integral ratios are invariant to velocity units.
    """
    wave = np.asarray(k, dtype=float)
    energy = np.asarray(e, dtype=float)
    if (wave.ndim != 1 or energy.shape != wave.shape or wave.size < 16
        or not np.isfinite(wave).all() or not np.isfinite(energy).all()
        or np.any(np.diff(wave) <= 0) or np.any(energy < 0)
        or wave[0] <= 0):
        raise ValueError("invalid physical energy spectrum")
    # The existing radial_velocity_spectrum uses uniform shells and E(k) / dk.
    dk = float(np.median(np.diff(wave)))
    if not np.allclose(np.diff(wave), dk, rtol=1e-6, atol=1e-12 * dk):
        raise ValueError("expected uniform radial spectral shells")
    relative = wave / wave[0]
    bands = (("low", relative < 4),
             ("mid", (relative >= 4) & (relative < 16)),
             ("high", relative >= 16))
    total = float(np.sum(energy) * dk)
    if total <= 0:
        raise ValueError("spectrum has zero turbulent energy")
    return {f"spectrum_{name}_fraction": float(np.sum(energy[mask]) * dk / total)
            for name, mask in bands}


def effective_block_length(values: np.ndarray, *, maximum: int = 12) -> int:
    """Autocorrelation-based block length (index units, not independent samples)."""
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 8 or not np.isfinite(x).all():
        raise ValueError("block-length estimate needs >=8 finite observations")
    centered = x - x.mean()
    variance = float(centered @ centered)
    if variance <= 1e-28:
        return 2
    cap = min(maximum, max(2, x.size // 4))
    positive = []
    for lag in range(1, cap + 1):
        rho = float(centered[:-lag] @ centered[lag:] / variance)
        if rho <= 0:
            break
        positive.append(rho)
    tau_int = 1.0 + 2.0 * sum(positive)
    return max(2, min(cap, int(np.ceil(tau_int))))


def block_bootstrap_mean(
    values: np.ndarray, *, length: int, repeats: int, seed: int,
) -> tuple[float, float, float]:
    """Circular moving-block percentile CI for the temporal mean.

    This is a descriptive time-dependence correction, not an independence
    certificate; estimates can be weak for a single 100-snapshot trajectory.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 8 or length < 1 or repeats < 16:
        raise ValueError("invalid block-bootstrap arguments")
    rng = np.random.default_rng(seed)
    nblocks = int(np.ceil(x.size / length))
    offsets = np.arange(length)
    means = np.empty(repeats)
    for i in range(repeats):
        starts = rng.integers(x.size, size=nblocks)
        selection = ((starts[:, None] + offsets[None, :]) % x.size).ravel()[:x.size]
        means[i] = np.mean(x[selection])
    low, high = np.quantile(means, [0.025, 0.975])
    return float(np.mean(x)), float(low), float(high)


def _normalized_slope(values: np.ndarray, anchor: float) -> float:
    """End-to-end linear drift relative to the tail reference level."""
    y = np.asarray(values, dtype=float)
    indices = np.arange(y.size, dtype=float)
    centered = indices - indices.mean()
    denominator = float(centered @ centered)
    if denominator <= 0:
        return 0.0
    slope = float(centered @ (y - y.mean()) / denominator)
    return slope * (y.size - 1) / max(abs(anchor), 1e-15)


def _evaluate_candidate(
    signals: Mapping[str, np.ndarray], config: QualificationConfig,
    *, tolerance: float, window: int,
) -> tuple[int | None, dict[str, dict[str, float]]]:
    n = len(next(iter(signals.values())))
    latest = n - config.anchor_windows * window
    minimum_remainder = config.minimum_late_windows * window
    if latest < window or n < minimum_remainder + window:
        return None, {}
    info: dict[str, dict[str, float]] = {}
    normalized_window_means = []
    for metric, value in signals.items():
        a = np.asarray(value, dtype=float)
        anchor_values = a[-config.anchor_windows * window:]
        anchor = float(np.mean(anchor_values))
        if abs(anchor) < 1e-14:
            # Spectral band is absent for a small test grid; ignore it explicitly.
            continue
        length = effective_block_length(anchor_values)
        average, low, high = block_bootstrap_mean(
            anchor_values, length=length, repeats=config.bootstrap_repeats,
            seed=config.seed + len(info),
        )
        uncertainty = (high - low) / (2 * abs(anchor))
        allowed = max(tolerance, 2.0 * uncertainty)
        # A noisy terminal reference must not make an arbitrary early transient pass.
        allowed = min(0.35, allowed)
        means = np.convolve(a, np.ones(window) / window, mode="valid")
        normalized_window_means.append((metric, means, anchor, allowed, a))
        info[metric] = {
            "late_mean": average,
            "late_ci95_low": low,
            "late_ci95_high": high,
            "bootstrap_block_length_snapshots": length,
            "relative_tolerance_used": allowed,
            "first_window_relative_error": abs(float(means[0]) - anchor) / abs(anchor),
        }
    if not normalized_window_means:
        return None, info
    candidate = None
    for start in range(0, n - minimum_remainder + 1):
        good = True
        for _, means, anchor, allowed, array in normalized_window_means:
            last_mean_start = n - window
            if np.max(np.abs(means[start:last_mean_start + 1] - anchor) / abs(anchor)) > allowed:
                good = False
                break
            if abs(_normalized_slope(array[start:], anchor)) > config.maximum_relative_drift:
                good = False
                break
        if good:
            # A passing window starts at `start` but includes earlier states.
            # Place the candidate onset at its END, avoiding spin-up leakage.
            candidate = start + window - 1
            break
    if candidate is not None:
        for name, _, anchor, _, arr in normalized_window_means:
            info[name]["candidate_remaining_relative_drift"] = _normalized_slope(
                arr[candidate:], anchor
            )
    return candidate, info


def qualify_case(
    time_series: Mapping[str, np.ndarray],
    config: QualificationConfig | None = None,
) -> dict:
    """Qualify one case, preserving uncertainty and parameter sensitivity.

    Inputs include spectral fractions computed from *each* cached snapshot,
    not spectra normalized against independently fitted case means.
    """
    conf = config or QualificationConfig()
    missing = set(STATIONARITY_METRICS) - set(time_series)
    if missing:
        raise ValueError(f"missing required stationarity metrics: {sorted(missing)}")
    metrics = {name: np.asarray(time_series[name], dtype=float)
               for name in STATIONARITY_METRICS}
    length = len(next(iter(metrics.values())))
    if (length < max(50, (conf.anchor_windows + conf.minimum_late_windows + 1) * conf.window)
        or any(a.ndim != 1 or a.size != length or not np.isfinite(a).all()
               or np.any(a < 0) for a in metrics.values())):
        raise ValueError("insufficient, inconsistent or invalid stationarity time histories")
    candidate, evidence = _evaluate_candidate(
        metrics, conf, tolerance=conf.relative_tolerance, window=conf.window
    )
    sensitivity = {}
    for w in sorted({max(8, conf.window - 2), conf.window, conf.window + 4}):
        for tol in (0.18, conf.relative_tolerance, 0.30):
            proposed, _ = _evaluate_candidate(metrics, conf, tolerance=tol, window=w)
            sensitivity[f"window_{w}_relative_tolerance_{tol:.2f}"] = proposed
    numerical = [v for v in sensitivity.values() if v is not None]
    # Separate candidate existence from robustness. The data are a single run.
    spread = max(numerical) - min(numerical) if len(numerical) >= 2 else None
    stable = (candidate is not None and len(numerical) >= 6 and spread is not None
              and spread <= 2 * conf.window)
    transition_start = max(0, candidate - conf.transition_width) if candidate is not None else None
    first_window_differs = any(v["first_window_relative_error"] > v["relative_tolerance_used"]
                               for v in evidence.values())
    if candidate is None:
        outcome = "inconclusive_no_sustained_plateau"
    elif not stable:
        outcome = "provisional_sensitive_to_parameters"
    elif not first_window_differs:
        outcome = "provisional_no_clear_initial_transient"
    else:
        outcome = "provisional_candidate_for_review"
    return {
        "status": outcome,
        "candidate_stationary_start_index": candidate,
        "candidate_transition_start_index": transition_start,
        "sensitivity_range_indices": (
            [min(numerical), max(numerical)] if numerical else None
        ),
        "sensitivity": sensitivity,
        "diagnostics": evidence,
        "snapshot_count": length,
        "physical_time_authoritative": False,
        "reviewed": False,
        "warning": (
            "Statistical stationarity is NOT automatically certified. "
            "Candidate index is 0-based; assess spectra, time correlations, "
            "forcing physics, and cutoff sensitivity before approving."
        ),
    }


def validate_reviewed_cutoffs(bounds: Mapping[str, int], n: int) -> tuple[int, int]:
    if (set(bounds) != {"transient_end_index", "stationary_start_index"}
        or any(type(bounds[k]) is not int for k in bounds)):
        raise ValueError("reviewed bounds need exactly two integer index fields")
    end, start = bounds["transient_end_index"], bounds["stationary_start_index"]
    if not 0 <= end < start < n or start - end < 2 or n - start < 16:
        raise ValueError(
            "cutoff indices must leave a transition gap and at least 16 stationary snapshots"
        )
    return end, start


def snapshot_labels(
    count: int, result: Mapping[str, object],
    reviewed: Mapping[str, int] | None = None,
) -> list[str]:
    if reviewed is not None:
        end, start = validate_reviewed_cutoffs(reviewed, count)
        return ["transient" if i <= end else "stationary" if i >= start
                else "transition" for i in range(count)]
    start = result.get("candidate_stationary_start_index")
    transition = result.get("candidate_transition_start_index")
    if start is None or transition is None:
        return ["unresolved"] * count
    return ["transient_candidate" if i < transition
            else "transition_uncertain" if i < start
            else "stationary_candidate" for i in range(count)]


def wasserstein_quantile_1d(a: np.ndarray, b: np.ndarray, *, bins: int = 512) -> float:
    """Discrete-quantile W1 approximation, requiring 1D finite sample arrays."""
    left, right = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if (left.ndim != 1 or right.ndim != 1 or not len(left) or not len(right)
        or not np.isfinite(left).all() or not np.isfinite(right).all()
        or bins < 8):
        raise ValueError("Wasserstein samples must be nonempty finite vectors")
    quantiles = (np.arange(bins) + 0.5) / bins
    return float(np.mean(np.abs(np.quantile(left, quantiles) -
                                np.quantile(right, quantiles))))
