"""Explicit train/validation/test sample split manifests."""

from __future__ import annotations

import random
from dataclasses import dataclass
from math import isfinite
from typing import Literal

SplitName = Literal["train", "validation", "test"]


@dataclass(frozen=True, slots=True)
class SplitManifest:
    """Immutable sample-ID partition for one training experiment.

    The manifest records membership only. It does not discover files, shuffle
    samples, or generate a split from data-dependent statistics.
    """

    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...] = ()
    test_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for attribute in ("train_ids", "validation_ids", "test_ids"):
            raw_values = getattr(self, attribute)
            if isinstance(raw_values, str):
                raise TypeError(f"{attribute} must be a sequence of sample IDs, not one string")
            values = tuple(raw_values)
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{attribute} must contain only non-empty sample IDs")
            object.__setattr__(self, attribute, values)

        if not self.train_ids:
            raise ValueError("SplitManifest.train_ids must contain at least one sample")

        all_ids = self.train_ids + self.validation_ids + self.test_ids
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("sample IDs must be unique across train/validation/test splits")

    @property
    def all_ids(self) -> tuple[str, ...]:
        return self.train_ids + self.validation_ids + self.test_ids

    def split_for(self, sample_id: str) -> SplitName:
        """Return the declared split for ``sample_id``."""

        if sample_id in self.train_ids:
            return "train"
        if sample_id in self.validation_ids:
            return "validation"
        if sample_id in self.test_ids:
            return "test"
        raise KeyError(f"sample '{sample_id}' is not present in the split manifest")


def make_grouped_split_manifest(
    sample_ids: tuple[str, ...] | list[str],
    group_ids: tuple[str, ...] | list[str],
    *,
    seed: int,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> SplitManifest:
    """Split complete metadata groups with the diffusion4avbp grouping convention.

    Unique groups are collected in first-occurrence order, deterministically
    shuffled with ``random.Random(seed)``, then partitioned by integer-truncated
    group counts. Every sample belonging to one group remains in one split.
    """

    samples = tuple(sample_ids)
    groups_for_samples = tuple(group_ids)
    if not samples:
        raise ValueError("sample_ids must contain at least one sample")
    if len(samples) != len(groups_for_samples):
        raise ValueError("sample_ids and group_ids must have identical lengths")
    if len(set(samples)) != len(samples):
        raise ValueError("sample_ids must be unique")
    if any(not isinstance(value, str) or not value.strip() for value in samples):
        raise ValueError("sample_ids must contain non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in groups_for_samples):
        raise ValueError("group_ids must contain non-empty strings")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    _validate_split_ratios(train_ratio, validation_ratio)

    grouped_samples: dict[str, list[str]] = {}
    for sample_id, group_id in zip(samples, groups_for_samples, strict=True):
        grouped_samples.setdefault(group_id, []).append(sample_id)

    shuffled_groups = list(grouped_samples)
    random.Random(seed).shuffle(shuffled_groups)
    num_groups = len(shuffled_groups)
    num_train = int(num_groups * train_ratio)
    num_validation = int(num_groups * validation_ratio)

    train_groups = shuffled_groups[:num_train]
    validation_groups = shuffled_groups[num_train : num_train + num_validation]
    test_groups = shuffled_groups[num_train + num_validation :]

    manifest = SplitManifest(
        train_ids=tuple(sample for group in train_groups for sample in grouped_samples[group]),
        validation_ids=tuple(
            sample for group in validation_groups for sample in grouped_samples[group]
        ),
        test_ids=tuple(sample for group in test_groups for sample in grouped_samples[group]),
    )
    _assert_group_disjoint(manifest, samples, groups_for_samples)
    return manifest


def _validate_split_ratios(train_ratio: float, validation_ratio: float) -> None:
    for value, name in ((train_ratio, "train_ratio"), (validation_ratio, "validation_ratio")):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real scalar")
        if not isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
    if not 0.0 < float(train_ratio) < 1.0:
        raise ValueError("train_ratio must lie strictly between zero and one")
    if not 0.0 <= float(validation_ratio) < 1.0:
        raise ValueError("validation_ratio must lie in [0, 1)")
    if float(train_ratio) + float(validation_ratio) >= 1.0:
        raise ValueError("train_ratio + validation_ratio must be strictly smaller than one")


def _assert_group_disjoint(
    manifest: SplitManifest,
    sample_ids: tuple[str, ...],
    group_ids: tuple[str, ...],
) -> None:
    group_by_sample = dict(zip(sample_ids, group_ids, strict=True))
    by_split = {
        "train": {group_by_sample[sample_id] for sample_id in manifest.train_ids},
        "validation": {group_by_sample[sample_id] for sample_id in manifest.validation_ids},
        "test": {group_by_sample[sample_id] for sample_id in manifest.test_ids},
    }
    pairs = (("train", "validation"), ("train", "test"), ("validation", "test"))
    for left, right in pairs:
        overlap = by_split[left] & by_split[right]
        if overlap:
            raise ValueError(
                f"group leakage detected between {left} and {right}: {sorted(overlap)}"
            )
