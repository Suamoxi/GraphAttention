"""Deterministic M44 SR-0 snapshot/patch experiment design (no CFD loading).

These are *sampling and split* utilities, not DNS-to-LES physical operators.
A snapshot and all its spatial patches always belong to a single split.
"""

from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class SR0SnapshotSplit:
    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    embargo: tuple[int, ...]

    def validate(self, selected: tuple[int, ...]) -> None:
        sets = [set(self.train), set(self.validation), set(self.test), set(self.embargo)]
        if any(not part for part in (self.train, self.validation, self.test)):
            raise ValueError("SR-0 requires nonempty train, validation and test snapshots")
        if any(sets[i] & sets[j] for i in range(4) for j in range(i + 1, 4)):
            raise ValueError("snapshot leakage: overlapping train/validation/test/embargo IDs")
        if set.union(*sets) != set(selected):
            raise ValueError("snapshot split must partition every selected snapshot")


def select_chronological_indices(total: int, maximum: int) -> tuple[int, ...]:
    """Evenly choose numbered snapshots without a random or spatial split."""
    if isinstance(total, bool) or isinstance(maximum, bool) or total < 3 or maximum < 3:
        raise ValueError("total and maximum must both be integers >=3")
    if maximum >= total:
        return tuple(range(total))
    # Middle of each disjoint chronological bin. Deterministic and distinct.
    chosen = tuple((2 * k + 1) * total // (2 * maximum) for k in range(maximum))
    if len(set(chosen)) != maximum or chosen != tuple(sorted(chosen)):
        raise RuntimeError("chronological down-selection unexpectedly duplicated a snapshot")
    return chosen


def split_chronological_indices(
    selected: tuple[int, ...],
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    embargo_snapshots: int = 1,
) -> SR0SnapshotSplit:
    """Contiguous chronological blocks separated by an optional index embargo.

    The embargo excludes selected source snapshots, NOT a guaranteed physical
    eddy-turnover-time interval. Physical temporal decorrelation is unverified.
    """
    if len(selected) != len(set(selected)) or tuple(sorted(selected)) != selected:
        raise ValueError("selected snapshot indices must be unique and chronological")
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1
            and train_fraction + validation_fraction < 1):
        raise ValueError("train/validation fractions must be positive with sum <1")
    if isinstance(embargo_snapshots, bool) or not isinstance(embargo_snapshots, int) or embargo_snapshots < 0:
        raise ValueError("embargo_snapshots must be a nonnegative integer")
    usable = len(selected) - 2 * embargo_snapshots
    if usable < 3:
        raise ValueError("too few snapshots for train/validation/test with this embargo")

    n_train = max(1, min(usable - 2, round(train_fraction * usable)))
    n_val = max(1, min(usable - n_train - 1, round(validation_fraction * usable)))
    n_test = usable - n_train - n_val
    k = 0
    train = selected[k:k + n_train]
    k += n_train
    first_gap = selected[k:k + embargo_snapshots]
    k += embargo_snapshots
    validation = selected[k:k + n_val]
    k += n_val
    second_gap = selected[k:k + embargo_snapshots]
    k += embargo_snapshots
    test = selected[k:k + n_test]
    result = SR0SnapshotSplit(train, validation, test, first_gap + second_gap)
    result.validate(selected)
    return result


def choose_patch_origins(
    *,
    side: int,
    patch_cells: int,
    refinement: int,
    patches_per_snapshot: int,
    seed: int,
    boundary_margin: int = 1,
) -> tuple[tuple[int, int, int], ...]:
    """Deterministically choose nonoverlapping Cartesian patches.

    Fixed origins are used at *every* snapshot, so source-snapshot splitting
    tests temporal generalization without confounding a different spatial
    patch sample between splits. No periodic wrap is assumed at patch edges.
    """
    if any(isinstance(v, bool) or not isinstance(v, int) for v in (
        side, patch_cells, refinement, patches_per_snapshot, seed, boundary_margin
    )):
        raise TypeError("SR-0 patch selection parameters must be integers")
    if side < 3 or patch_cells < 2 or refinement < 2 or patch_cells % refinement:
        raise ValueError("invalid Cartesian patch size/refinement")
    if patches_per_snapshot < 1 or seed < 0 or boundary_margin < 0:
        raise ValueError("patch count, random seed or boundary margin invalid")
    max_origin = (side - 1) - patch_cells - boundary_margin
    if max_origin < boundary_margin:
        raise ValueError("no valid interior patch fits; decrease boundary_margin")
    positions = list(range(boundary_margin, max_origin + 1, patch_cells))
    candidates = [(i, j, k) for i in positions for j in positions for k in positions]
    if patches_per_snapshot > len(candidates):
        raise ValueError(f"requested {patches_per_snapshot} patches; only {len(candidates)} available")
    rng = random.Random(seed)
    rng.shuffle(candidates)
    return tuple(sorted(candidates[:patches_per_snapshot]))


def choose_epoch_patch_indices(
    *,
    pool_size: int,
    patches_per_epoch: int,
    seed: int,
    snapshot_index: int,
    epoch: int,
) -> tuple[int, ...]:
    """Fixed-budget, deterministic, snapshot-specific cyclic pool schedule.

    Every source snapshot sees each pool origin exactly once per complete
    cycle, and only `patches_per_epoch` origins per epoch. No global RNG is
    consumed, so interrupted/resumed runs reconstruct the identical epoch
    selection. Indexing is into the sorted train patch-origin pool.

    This is a spatial augmentation, not a temporal split or LES restriction.
    """
    for name, value in (
        ("pool_size", pool_size),
        ("patches_per_epoch", patches_per_epoch),
        ("seed", seed),
        ("snapshot_index", snapshot_index),
        ("epoch", epoch),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if pool_size < 1 or patches_per_epoch < 1 or seed < 0 or snapshot_index < 0 or epoch < 0:
        raise ValueError("patch schedule requires positive pool/budget and nonnegative indices")
    if pool_size % patches_per_epoch:
        raise ValueError("pool_size must be a multiple of patches_per_epoch for exact coverage")

    permutation = list(range(pool_size))
    random.Random((seed + 1) * 1000003 + snapshot_index * 1009).shuffle(permutation)
    cycle_length = pool_size // patches_per_epoch
    slot = epoch % cycle_length
    selected = permutation[slot * patches_per_epoch:(slot + 1) * patches_per_epoch]
    return tuple(sorted(selected))
