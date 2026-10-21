"""Coarse-feature-matched DNS comparisons with snapshot-block uncertainty.

A finite descriptor vector is NOT a sufficient statistic for the full LES state.
Rows sampled from the same snapshot are clustered, never counted as independent
replicates for confidence intervals.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

DEFAULT_FEATURES = ("log_coarse_energy", "log_strain")
DEFAULT_TARGETS = ("unresolved_k", "within_s2", "cross_s2")


def _validated_rows(
    table: Mapping[str, np.ndarray], keys: tuple[str, ...]
) -> dict[str, np.ndarray]:
    required = (*keys, "snapshot_index")
    data = {key: np.asarray(table[key]).reshape(-1) for key in required}
    n = len(data["snapshot_index"])
    if n == 0 or any(len(v) != n for v in data.values()):
        raise ValueError("empty or inconsistent matched feature table")
    keep = np.ones(n, dtype=bool)
    for key in keys:
        keep &= np.isfinite(data[key].astype(np.float64))
    if not keep.any():
        raise ValueError("no finite matched rows")
    return {key: v[keep] for key, v in data.items()}


def _assign_bins(
    matrix: np.ndarray, edges: list[np.ndarray], nbins: int
) -> np.ndarray:
    assigned = np.zeros(matrix.shape[0], dtype=np.int64)
    for axis, cuts in enumerate(edges):
        digit = np.searchsorted(cuts[1:-1], matrix[:, axis], side="right")
        assigned = assigned * nbins + digit
    return assigned


def _one_side_summary(
    table: dict[str, np.ndarray],
    bin_id: np.ndarray,
    supported: np.ndarray,
    count_bins: int,
    metrics: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = table["snapshot_index"].astype(np.int64)
    snapshots, inverse = np.unique(labels, return_inverse=True)
    counts = np.zeros((len(snapshots), count_bins), dtype=np.float64)
    sums = np.zeros((len(snapshots), count_bins, len(metrics)), dtype=np.float64)
    flat = inverse * count_bins + bin_id
    counts[:] = np.bincount(
        flat, weights=supported.astype(float), minlength=counts.size
    ).reshape(counts.shape)
    for k, metric in enumerate(metrics):
        sums[:, :, k] = np.bincount(
            flat,
            weights=np.where(supported, table[metric].astype(float), 0.0),
            minlength=counts.size,
        ).reshape(counts.shape)
    return snapshots, counts, sums


def matched_regime_comparison(
    transient: Mapping[str, np.ndarray],
    stationary: Mapping[str, np.ndarray],
    *,
    features: tuple[str, ...] = DEFAULT_FEATURES,
    targets: tuple[str, ...] = DEFAULT_TARGETS,
    bins_per_feature: int = 4,
    min_rows_per_bin: int = 20,
    min_snapshots_per_bin: int = 2,
    bootstrap_repeats: int = 200,
    seed: int = 42,
) -> dict:
    """Histogram overlap standardization with independently resampled snapshots.

    Shared matched-bin weights are proportional to min(P_trans(bin), P_stat(bin)).
    The interpretation is conditional comparison over common descriptor support,
    not comparison at identical full coarse states.
    """
    if not 1 <= len(features) <= 3 or bins_per_feature < 2:
        raise ValueError("choose one to three features and at least two bins")
    if min_rows_per_bin < 1 or min_snapshots_per_bin < 1:
        raise ValueError("minimum support must be positive")
    keys = tuple(dict.fromkeys((*features, *targets)))
    t = _validated_rows(transient, keys)
    s = _validated_rows(stationary, keys)
    arrays = [
        np.column_stack([table[key].astype(np.float64) for key in features])
        for table in (t, s)
    ]
    min_bound = np.maximum(arrays[0].min(axis=0), arrays[1].min(axis=0))
    max_bound = np.minimum(arrays[0].max(axis=0), arrays[1].max(axis=0))
    if np.any(min_bound >= max_bound):
        return {
            "status": "inconclusive_no_common_support",
            "reason": "descriptor ranges do not overlap",
            "features": list(features),
            "support_bounds": {"lower": min_bound.tolist(), "upper": max_bound.tolist()},
        }
    valid_range = [
        np.all((a >= min_bound) & (a <= max_bound), axis=1)
        for a in arrays
    ]
    if not all(np.any(v) for v in valid_range):
        return {"status": "inconclusive_no_common_support",
                "reason": "no joint descriptor overlap", "features": list(features)}
    pooled = np.concatenate([a[mask] for a, mask in zip(arrays, valid_range)])
    edges = [np.quantile(pooled[:, i], np.linspace(0, 1, bins_per_feature + 1))
             for i in range(len(features))]
    if any(np.unique(cuts).size < bins_per_feature + 1 for cuts in edges):
        return {"status": "inconclusive_no_common_support",
                "reason": "degenerate descriptor quantile bins",
                "features": list(features)}
    n_bins = bins_per_feature**len(features)
    ids = [_assign_bins(a, edges, bins_per_feature) for a in arrays]
    hist = [np.bincount(i[mask], minlength=n_bins)
            for i, mask in zip(ids, valid_range)]
    unique_snapshot_count = []
    supported = (hist[0] >= min_rows_per_bin) & (hist[1] >= min_rows_per_bin)
    for table, which in zip((t, s), ids):
        seen = np.zeros(n_bins, dtype=np.int64)
        for b in np.flatnonzero(supported):
            seen[b] = len(np.unique(table["snapshot_index"][which == b]))
        unique_snapshot_count.append(seen)
        supported &= seen >= min_snapshots_per_bin
    if not supported.any():
        return {
            "status": "inconclusive_no_common_support",
            "reason": "insufficient shared within-bin/snapshot support",
            "features": list(features),
            "rows": {"transient": len(t["snapshot_index"]),
                     "stationary": len(s["snapshot_index"])},
        }
    masks = [mask & supported[which] for mask, which in zip(valid_range, ids)]
    counts = [np.bincount(which[mask], minlength=n_bins).astype(float)
              for which, mask in zip(ids, masks)]
    weights = np.minimum(counts[0] / counts[0].sum(), counts[1] / counts[1].sum())
    weights /= weights.sum()
    metric_names = (*features, *targets)
    group_results = []
    group_accumulators = []
    for table, which, mask in zip((t, s), ids, masks):
        snapshots, per_snap_count, per_snap_sum = _one_side_summary(
            table, which, mask, n_bins, metric_names
        )
        sums = per_snap_sum.sum(axis=0)
        total = per_snap_count.sum(axis=0)
        mean = np.divide(sums, total[:, None], out=np.zeros_like(sums),
                         where=total[:, None] > 0)
        matched = weights @ mean
        unconditional = np.asarray(
            [float(np.mean(table[m].astype(float))) for m in metric_names]
        )
        group_results.append({
            "snapshots": int(len(snapshots)),
            "rows": int(len(table["snapshot_index"])),
            "rows_matched": int(mask.sum()),
            "fraction_retained": float(mask.mean()),
            "unconditional": dict(zip(metric_names, unconditional.tolist())),
            "matched": dict(zip(metric_names, matched.tolist())),
        })
        group_accumulators.append((per_snap_count, per_snap_sum))
    result = {
        "status": "ok",
        "interpretation": "descriptor-matched association; no causal or full-field conditional claim",
        "features": list(features),
        "targets": list(targets),
        "bins_per_feature": bins_per_feature,
        "common_bins": int(supported.sum()),
        "total_bins": n_bins,
        "support_bounds": {
            "lower": min_bound.tolist(),
            "upper": max_bound.tolist(),
        },
        "quantile_edges": [v.tolist() for v in edges],
        "transient": group_results[0],
        "stationary": group_results[1],
        "matched_difference_stationary_minus_transient": {},
        "confidence_intervals_95_snapshot_bootstrap": {},
    }
    rng = np.random.default_rng(seed)
    for k, target in enumerate(targets):
        j = len(features) + k
        difference = (
            group_results[1]["matched"][target] - group_results[0]["matched"][target]
        )
        result["matched_difference_stationary_minus_transient"][target] = float(difference)
        if bootstrap_repeats < 20:
            continue
        draws = []
        for _ in range(bootstrap_repeats):
            sampled = []
            for per_snap_count, per_snap_sum in group_accumulators:
                resampled = rng.integers(len(per_snap_count), size=len(per_snap_count))
                cnt = per_snap_count[resampled].sum(axis=0)
                if np.any(cnt[supported] == 0):
                    break
                totals = per_snap_sum[resampled, :, j].sum(axis=0)
                sampled.append(float(np.sum(weights[supported] *
                    (totals[supported] / cnt[supported]))))
            if len(sampled) == 2:
                draws.append(sampled[1] - sampled[0])
        if len(draws) >= max(20, bootstrap_repeats // 2):
            result["confidence_intervals_95_snapshot_bootstrap"][target] = [
                float(v) for v in np.quantile(draws, [0.025, 0.975])
            ]
        else:
            result["confidence_intervals_95_snapshot_bootstrap"][target] = None
    for key in features:
        result.setdefault("descriptor_balance", {})[key] = {
            "before": float(group_results[1]["unconditional"][key] -
                            group_results[0]["unconditional"][key]),
            "after": float(group_results[1]["matched"][key] -
                           group_results[0]["matched"][key]),
        }
    return result
