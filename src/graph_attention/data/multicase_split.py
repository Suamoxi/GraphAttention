"""Chronological case-stratified splits and balanced sampling for M43-B.

Never split 600 snapshots by randomly mixing neighboring times. Each
simulation trajectory receives separate contiguous train/val/test intervals
with an index-space exclusion window at both boundaries.

Index-space embargo is NOT proof of physical decorrelation: snapshot times
and turbulent turnover times need additional scientific auditing.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
import random
from typing import Mapping, Sequence

from .splits import SplitManifest


@dataclass(frozen=True, slots=True)
class MultiCaseSplit:
    manifest: SplitManifest
    excluded_ids: tuple[str, ...]
    by_case: dict[str, dict[str, object]]
    temporal_embargo: int
    train_ratio: float
    validation_ratio: float


def temporal_case_split(
    case_sample_ids: Mapping[str, Sequence[str]],
    *,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
    temporal_embargo: int = 2,
) -> MultiCaseSplit:
    """Contiguous per-trajectory holdouts, with an explicit sample index embargo.

    Remove temporal_embargo samples on EACH side of each
    train/validation and validation/test boundary.
    """
    if not isinstance(temporal_embargo, int) or isinstance(temporal_embargo, bool):
        raise TypeError("temporal_embargo must be an integer")
    if temporal_embargo < 0:
        raise ValueError("temporal_embargo must be >= 0")
    if (not isfinite(float(train_ratio)) or not isfinite(float(validation_ratio))
        or not 0 < train_ratio < 1
        or not 0 < validation_ratio < 1
        or train_ratio + validation_ratio >= 1):
        raise ValueError("invalid train/validation ratios")

    entries = tuple(case_sample_ids.items())
    if not entries:
        raise ValueError("at least one physical case is required")
    splits: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    excluded: list[str] = []
    case_summary: dict[str, dict[str, object]] = {}
    all_ids: list[str] = []
    for case, raw_ids in entries:
        ids = tuple(raw_ids)
        if not case or not ids or len(set(ids)) != len(ids):
            raise ValueError(f"invalid or duplicate sample IDs in {case}")
        if tuple(sorted(ids)) != ids:
            raise ValueError(
                f"{case}: snapshot paths must be in chronological filename order"
            )
        n = len(ids)
        train_stop = int(n * train_ratio)
        validation_stop = int(n * (train_ratio + validation_ratio))
        if (train_stop <= temporal_embargo
            or validation_stop - train_stop <= 2 * temporal_embargo
            or n - validation_stop <= temporal_embargo):
            raise ValueError(f"{case}: not enough snapshots for requested embargo")
        case_parts = {
            "train": ids[:train_stop-temporal_embargo],
            "validation": ids[train_stop+temporal_embargo:validation_stop-temporal_embargo],
            "test": ids[validation_stop+temporal_embargo:],
        }
        blocked = (
            ids[train_stop-temporal_embargo:train_stop+temporal_embargo]
            + ids[validation_stop-temporal_embargo:validation_stop+temporal_embargo]
        )
        for kind in ("train", "validation", "test"):
            splits[kind].extend(case_parts[kind])
        excluded.extend(blocked)
        all_ids.extend(ids)
        case_summary[case] = {
            "total": n,
            "train": len(case_parts["train"]),
            "validation": len(case_parts["validation"]),
            "test": len(case_parts["test"]),
            "excluded": len(blocked),
            "train_interval": [0, train_stop-temporal_embargo],
            "validation_interval": [
                train_stop+temporal_embargo, validation_stop-temporal_embargo,
            ],
            "test_interval": [validation_stop+temporal_embargo, n],
            "boundary_indices": [train_stop, validation_stop],
        }
    manifest = SplitManifest(
        train_ids=tuple(splits["train"]),
        validation_ids=tuple(splits["validation"]),
        test_ids=tuple(splits["test"]),
    )
    if (len(set(all_ids)) != len(all_ids)
        or set(manifest.all_ids).intersection(excluded)
        or set(manifest.all_ids).union(excluded) != set(all_ids)):
        raise AssertionError("case split is not disjoint and complete")
    return MultiCaseSplit(
        manifest=manifest, excluded_ids=tuple(excluded),
        by_case=case_summary,
        temporal_embargo=temporal_embargo,
        train_ratio=float(train_ratio),
        validation_ratio=float(validation_ratio),
    )


def balanced_case_interleave(
    case_indices: Mapping[str, Sequence[int]],
    *,
    seed: int,
    epoch: int,
) -> tuple[int, ...]:
    """Sample each case once per round, independently of node count.

    Caller orders cases to interleave resolutions. No samples
    are duplicated or omitted.
    """
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a nonnegative integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    randomizer = random.Random(seed + epoch * 1000003)
    queues: list[list[int]] = []
    for case, original in case_indices.items():
        indices = list(original)
        if not case or not indices or any(not isinstance(x, int) for x in indices):
            raise ValueError(f"case {case!r} has no valid indices")
        if len(set(indices)) != len(indices):
            raise ValueError(f"case {case}: repeated sample index")
        randomizer.shuffle(indices)
        queues.append(indices)

    if not queues:
        raise ValueError("at least one case is required")
    all_indices = [value for values in queues for value in values]
    if len(set(all_indices)) != len(all_indices):
        raise ValueError("an index appears in multiple cases")
    ordered: list[int] = []
    for row in range(max(len(values) for values in queues)):
        for queue in queues:
            if row < len(queue):
                ordered.append(queue[row])
    assert len(ordered) == len(all_indices)
    return tuple(ordered)
