"""Explicit train/validation/test sample split manifests."""

from __future__ import annotations

from dataclasses import dataclass
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
