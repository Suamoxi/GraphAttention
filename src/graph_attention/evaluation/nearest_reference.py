"""Unpaired nearest-reference diagnostics for generated CFD snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .spectra import CartesianGrid2D, scatter_to_grid


@dataclass(frozen=True)
class NearestReferenceDiagnostics:
    """Nearest-neighbour results in normalized snapshot-descriptor space."""

    rows: tuple[dict[str, float | int | str], ...]
    summary: dict[str, Any]
    generated_to_reference_indices: np.ndarray
    generated_to_reference_distances: np.ndarray


def nearest_neighbor_spatial_metrics(
    values: np.ndarray,
    grid: CartesianGrid2D,
) -> tuple[float, float]:
    """Return nearest-grid-neighbour correlation and first-difference RMS."""

    field = scatter_to_grid(values, grid)
    first = np.concatenate((field[:-1, :].reshape(-1), field[:, :-1].reshape(-1)))
    second = np.concatenate((field[1:, :].reshape(-1), field[:, 1:].reshape(-1)))
    correlation = _correlation(first, second)
    difference_rms = float(np.sqrt(np.mean((second - first) ** 2)))
    return correlation, difference_rms


def build_snapshot_descriptors(
    samples: np.ndarray,
    grid: CartesianGrid2D,
    channel_names: tuple[str, ...],
    sample_power: np.ndarray,
    k_centers: np.ndarray,
    *,
    k_nyquist: float,
    bands: dict[str, tuple[float, float]],
    include_channel_correlations: bool,
) -> tuple[tuple[str, ...], np.ndarray]:
    """Build one interpretable statistical/spatial descriptor per snapshot.

    Each channel contributes spatial mean, spatial standard deviation,
    nearest-neighbour correlation, first-difference RMS, and integrated spectral
    power in every configured band. Optional within-snapshot cross-channel
    correlations add one feature per unique channel pair.
    """

    values = np.asarray(samples, dtype=np.float64)
    power = np.asarray(sample_power, dtype=np.float64)
    centers = np.asarray(k_centers, dtype=np.float64)
    if values.ndim != 3 or values.shape[2] != len(channel_names):
        raise ValueError("snapshot descriptor samples must have shape [S, N, C]")
    if power.ndim != 3 or power.shape[:2] != (values.shape[0], values.shape[2]):
        raise ValueError("sample spectral power must have shape [S, C, K]")
    if power.shape[2] != centers.size:
        raise ValueError("sample spectral power does not match k centers")
    if k_nyquist <= 0.0:
        raise ValueError("k_nyquist must be positive")

    band_masks = _spectral_band_masks(centers, k_nyquist, bands)
    feature_names: list[str] = []
    columns: list[np.ndarray] = []

    for channel, name in enumerate(channel_names):
        channel_values = values[:, :, channel]
        columns.append(np.mean(channel_values, axis=1))
        feature_names.append(f"{name}:spatial_mean")
        columns.append(np.std(channel_values, axis=1, ddof=0))
        feature_names.append(f"{name}:spatial_std")

        correlations = np.empty(values.shape[0], dtype=np.float64)
        difference_rms = np.empty(values.shape[0], dtype=np.float64)
        for sample_index in range(values.shape[0]):
            correlations[sample_index], difference_rms[sample_index] = (
                nearest_neighbor_spatial_metrics(channel_values[sample_index], grid)
            )
        columns.append(correlations)
        feature_names.append(f"{name}:nearest_neighbor_correlation")
        columns.append(difference_rms)
        feature_names.append(f"{name}:first_difference_rms")

        for band_name, mask in band_masks.items():
            columns.append(np.sum(power[:, channel, :][:, mask], axis=1))
            feature_names.append(f"{name}:spectral_power:{band_name}")

    if include_channel_correlations:
        for first_channel in range(len(channel_names)):
            for second_channel in range(first_channel + 1, len(channel_names)):
                correlations = np.asarray(
                    [
                        _correlation(
                            values[sample_index, :, first_channel],
                            values[sample_index, :, second_channel],
                        )
                        for sample_index in range(values.shape[0])
                    ],
                    dtype=np.float64,
                )
                columns.append(correlations)
                feature_names.append(
                    "channel_correlation:"
                    f"{channel_names[first_channel]}|{channel_names[second_channel]}"
                )

    return tuple(feature_names), np.column_stack(columns)


def nearest_reference_diagnostics(
    generated_features: np.ndarray,
    reference_features: np.ndarray,
    generated_ids: tuple[str, ...],
    reference_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    *,
    normalization_eps: float,
) -> NearestReferenceDiagnostics:
    """Compute fidelity, coverage, and real-data leave-one-out NN distances.

    Features are standardized only with statistics from the test reference
    population. Distances are root-mean-square Euclidean distances over active
    standardized features, so one unit has the interpretation of roughly one
    reference-population standard deviation per descriptor component.
    """

    generated = np.asarray(generated_features, dtype=np.float64)
    reference = np.asarray(reference_features, dtype=np.float64)
    if generated.ndim != 2 or reference.ndim != 2:
        raise ValueError("nearest-reference features must be two-dimensional")
    if generated.shape[1] != reference.shape[1] or generated.shape[1] != len(feature_names):
        raise ValueError("generated/reference feature dimensions must match feature names")
    if generated.shape[0] != len(generated_ids):
        raise ValueError("generated feature rows must match generated identifiers")
    if reference.shape[0] != len(reference_ids):
        raise ValueError("reference feature rows must match reference identifiers")
    if reference.shape[0] < 2:
        raise ValueError("test-to-test calibration requires at least two reference snapshots")
    eps = float(normalization_eps)
    if eps <= 0.0:
        raise ValueError("normalization_eps must be positive")

    reference_mean = np.nanmean(reference, axis=0)
    reference_std = np.nanstd(reference, axis=0, ddof=0)
    finite_reference = np.all(np.isfinite(reference), axis=0)
    active = finite_reference & np.isfinite(reference_mean) & (reference_std > eps)
    if not np.any(active):
        raise ValueError("nearest-reference descriptor has no varying finite test features")
    if not np.all(np.isfinite(generated[:, active])):
        raise ValueError("generated snapshots contain non-finite active descriptor features")

    generated_normalized = (generated[:, active] - reference_mean[active]) / reference_std[active]
    reference_normalized = (reference[:, active] - reference_mean[active]) / reference_std[active]

    generated_to_reference = _pairwise_rms_distance(
        generated_normalized,
        reference_normalized,
    )
    generated_nearest_indices = np.argmin(generated_to_reference, axis=1)
    generated_nearest_distances = generated_to_reference[
        np.arange(generated.shape[0]), generated_nearest_indices
    ]

    reference_nearest_generated_indices = np.argmin(generated_to_reference, axis=0)
    reference_nearest_generated_distances = generated_to_reference[
        reference_nearest_generated_indices, np.arange(reference.shape[0])
    ]

    reference_to_reference = _pairwise_rms_distance(
        reference_normalized,
        reference_normalized,
    )
    np.fill_diagonal(reference_to_reference, np.inf)
    reference_nearest_reference_indices = np.argmin(reference_to_reference, axis=1)
    reference_nearest_reference_distances = reference_to_reference[
        np.arange(reference.shape[0]), reference_nearest_reference_indices
    ]

    rows: list[dict[str, float | int | str]] = []
    rows.extend(
        _direction_rows(
            "generated_to_test",
            generated_ids,
            reference_ids,
            generated_nearest_indices,
            generated_nearest_distances,
            source_role="sampling_key",
            nearest_role="test_sample_id",
        )
    )
    rows.extend(
        _direction_rows(
            "test_to_generated",
            reference_ids,
            generated_ids,
            reference_nearest_generated_indices,
            reference_nearest_generated_distances,
            source_role="test_sample_id",
            nearest_role="sampling_key",
        )
    )
    rows.extend(
        _direction_rows(
            "test_to_test_leave_one_out",
            reference_ids,
            reference_ids,
            reference_nearest_reference_indices,
            reference_nearest_reference_distances,
            source_role="test_sample_id",
            nearest_role="test_sample_id",
        )
    )

    generated_summary = _distance_summary(generated_nearest_distances)
    coverage_summary = _distance_summary(reference_nearest_generated_distances)
    calibration_summary = _distance_summary(reference_nearest_reference_distances)
    calibration_mean = calibration_summary["mean"]
    summary = {
        "distance": "rms_euclidean_in_test_standardized_descriptor_space",
        "feature_normalization": "test_population_mean_and_std",
        "num_features_total": len(feature_names),
        "num_features_active": int(np.sum(active)),
        "active_features": [name for name, keep in zip(feature_names, active, strict=True) if keep],
        "omitted_nonvarying_or_nonfinite_test_features": [
            name for name, keep in zip(feature_names, active, strict=True) if not keep
        ],
        "generated_to_test": generated_summary,
        "test_to_generated": coverage_summary,
        "test_to_test_leave_one_out": calibration_summary,
        "generated_to_test_mean_over_test_to_test_mean": (
            generated_summary["mean"] / calibration_mean if calibration_mean > eps else None
        ),
        "test_to_generated_mean_over_test_to_test_mean": (
            coverage_summary["mean"] / calibration_mean if calibration_mean > eps else None
        ),
        "generated_ids_are_sampling_keys_not_target_pairings": True,
    }
    return NearestReferenceDiagnostics(
        rows=tuple(rows),
        summary=summary,
        generated_to_reference_indices=generated_nearest_indices,
        generated_to_reference_distances=generated_nearest_distances,
    )


def _spectral_band_masks(
    k_centers: np.ndarray,
    k_nyquist: float,
    bands: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    fraction = np.asarray(k_centers, dtype=np.float64) / float(k_nyquist)
    masks: dict[str, np.ndarray] = {}
    for name, (lower, upper) in bands.items():
        if lower < 0.0 or upper <= lower or upper > 1.0:
            raise ValueError(f"invalid normalized spectral band '{name}': ({lower}, {upper})")
        mask = (fraction >= lower) & (fraction < upper)
        if upper == 1.0:
            mask = (fraction >= lower) & (fraction <= upper)
        if not np.any(mask):
            raise ValueError(f"spectral band '{name}' contains no radial bins")
        masks[name] = mask
    if not masks:
        raise ValueError("nearest-reference descriptor requires at least one spectral band")
    return masks


def _pairwise_rms_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    if left.shape[1] != right.shape[1] or left.shape[1] == 0:
        raise ValueError("pairwise distance requires a shared non-empty feature dimension")
    left_norm = np.sum(left * left, axis=1)[:, None]
    right_norm = np.sum(right * right, axis=1)[None, :]
    squared = left_norm + right_norm - 2.0 * left @ right.T
    squared = np.maximum(squared / left.shape[1], 0.0)
    return np.sqrt(squared)


def _direction_rows(
    direction: str,
    source_ids: tuple[str, ...],
    nearest_ids: tuple[str, ...],
    nearest_indices: np.ndarray,
    distances: np.ndarray,
    *,
    source_role: str,
    nearest_role: str,
) -> list[dict[str, float | int | str]]:
    return [
        {
            "direction": direction,
            "source_index": source_index,
            "source_id": source_ids[source_index],
            "source_id_role": source_role,
            "nearest_index": int(nearest_indices[source_index]),
            "nearest_id": nearest_ids[int(nearest_indices[source_index])],
            "nearest_id_role": nearest_role,
            "distance": float(distances[source_index]),
        }
        for source_index in range(len(source_ids))
    ]


def _distance_summary(distances: np.ndarray) -> dict[str, float]:
    values = np.asarray(distances, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    left = np.asarray(first, dtype=np.float64).reshape(-1)
    right = np.asarray(second, dtype=np.float64).reshape(-1)
    if left.shape != right.shape or left.size == 0:
        raise ValueError("correlation inputs must have the same non-empty shape")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = np.sqrt(float(np.sum(left_centered**2)) * float(np.sum(right_centered**2)))
    if denominator <= 0.0:
        return float("nan")
    return float(np.sum(left_centered * right_centered) / denominator)
