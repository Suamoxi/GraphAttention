"""Train-only sample-balanced statistical scaling for task channels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from math import isfinite

import torch

from graph_attention.data import FieldCatalog, Sample, SplitManifest
from graph_attention.tasks import NodeRegressionBatch, NodeRegressionTask


_DEFAULT_MINIMUM_STD = 1.0e-12


@dataclass(frozen=True, slots=True)
class ChannelStandardizer:
    """Frozen affine standardizer for an explicit ordered channel set."""

    channel_names: tuple[str, ...]
    mean: torch.Tensor
    scale: torch.Tensor
    minimum_std: float = _DEFAULT_MINIMUM_STD

    def __post_init__(self) -> None:
        names = tuple(self.channel_names)
        if not names or any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("channel_names must contain non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("channel_names must be unique")
        if not isinstance(self.mean, torch.Tensor) or not isinstance(self.scale, torch.Tensor):
            raise TypeError("mean and scale must be torch.Tensor values")
        if self.mean.ndim != 1 or self.scale.ndim != 1:
            raise ValueError("mean and scale must have shape [C]")
        if self.mean.shape != self.scale.shape or self.mean.shape[0] != len(names):
            raise ValueError("mean/scale shapes must match the declared channel count")
        if not self.mean.is_floating_point() or not self.scale.is_floating_point():
            raise TypeError("mean and scale must use floating-point dtypes")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.scale).all():
            raise ValueError("mean and scale must contain only finite values")
        if not isinstance(self.minimum_std, (int, float)) or isinstance(self.minimum_std, bool):
            raise TypeError("minimum_std must be a real scalar")
        minimum_std = float(self.minimum_std)
        if not isfinite(minimum_std) or minimum_std <= 0.0:
            raise ValueError("minimum_std must be finite and strictly positive")
        if torch.any(self.scale <= minimum_std):
            raise ValueError("scale values must be strictly greater than minimum_std")

        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "minimum_std", minimum_std)
        object.__setattr__(self, "mean", self.mean.detach().clone())
        object.__setattr__(self, "scale", self.scale.detach().clone())

    def transform(
        self,
        values: torch.Tensor,
        channel_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Standardize values after validating exact channel semantics."""

        self._validate_values(values, channel_names)
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return (values - mean) / scale

    def inverse(
        self,
        values: torch.Tensor,
        channel_names: tuple[str, ...],
    ) -> torch.Tensor:
        """Invert standardization for the same ordered semantic channels."""

        self._validate_values(values, channel_names)
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        scale = self.scale.to(device=values.device, dtype=values.dtype)
        return values * scale + mean

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ChannelStandardizer:
        """Return a copy with scaler tensors moved/cast for repeated use."""

        target_dtype = dtype or self.mean.dtype
        if not target_dtype.is_floating_point:
            raise TypeError("standardizer dtype must be floating-point")
        return ChannelStandardizer(
            channel_names=self.channel_names,
            mean=self.mean.to(device=device, dtype=target_dtype),
            scale=self.scale.to(device=device, dtype=target_dtype),
            minimum_std=self.minimum_std,
        )

    def _validate_values(
        self,
        values: torch.Tensor,
        channel_names: tuple[str, ...],
    ) -> None:
        if tuple(channel_names) != self.channel_names:
            raise ValueError(
                "channel order does not match fitted standardizer: "
                f"got {tuple(channel_names)}, expected {self.channel_names}"
            )
        if values.ndim != 2 or values.shape[1] != len(self.channel_names):
            raise ValueError(
                f"values must have shape [N, {len(self.channel_names)}], got {tuple(values.shape)}"
            )
        if not values.is_floating_point():
            raise TypeError("values must use a floating-point dtype")
        if not torch.isfinite(values).all():
            raise ValueError("values contain NaN or Inf")


@dataclass(frozen=True, slots=True)
class TaskStandardizers:
    """Frozen input/target scalers tied to one training split and preprocessing convention."""

    inputs: ChannelStandardizer
    targets: ChannelStandardizer
    train_sample_ids: tuple[str, ...]
    physical_nondimensionalization: bool
    weighting: str = "sample_balanced"

    def __post_init__(self) -> None:
        train_ids = tuple(self.train_sample_ids)
        if not train_ids or any(
            not isinstance(value, str) or not value.strip() for value in train_ids
        ):
            raise ValueError("train_sample_ids must contain non-empty sample IDs")
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train_sample_ids must be unique")
        if not isinstance(self.physical_nondimensionalization, bool):
            raise TypeError("physical_nondimensionalization must be a bool")
        if self.weighting != "sample_balanced":
            raise ValueError("M6 supports only sample_balanced statistical scaling")
        object.__setattr__(self, "train_sample_ids", train_ids)

    def transform(self, batch: NodeRegressionBatch) -> NodeRegressionBatch:
        """Return a statistically scaled copy of one prepared task batch."""

        return replace(
            batch,
            inputs=self.inputs.transform(batch.inputs, batch.input_channels),
            targets=self.targets.transform(batch.targets, batch.target_channels),
        )

    def inverse_targets(
        self,
        values: torch.Tensor,
        target_channels: tuple[str, ...],
    ) -> torch.Tensor:
        """Map standardized model outputs back to the task's pre-statistical scale."""

        return self.targets.inverse(values, target_channels)

    def to(
        self,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> TaskStandardizers:
        """Return a copy with scaler tensors moved/cast for repeated training use."""

        return TaskStandardizers(
            inputs=self.inputs.to(device=device, dtype=dtype),
            targets=self.targets.to(device=device, dtype=dtype),
            train_sample_ids=self.train_sample_ids,
            physical_nondimensionalization=self.physical_nondimensionalization,
            weighting=self.weighting,
        )


def fit_train_standardizers(
    task: NodeRegressionTask,
    training_samples: Iterable[Sample],
    catalog: FieldCatalog,
    split_manifest: SplitManifest,
    *,
    minimum_std: float = _DEFAULT_MINIMUM_STD,
) -> TaskStandardizers:
    """Fit equal-sample input/target statistics from exactly the declared train split.

    Every physical sample contributes total statistical weight one regardless of
    its node count. Samples outside ``split_manifest.train_ids`` are rejected
    rather than ignored so validation/test leakage cannot occur silently.
    """

    if not isinstance(minimum_std, (int, float)) or isinstance(minimum_std, bool):
        raise TypeError("minimum_std must be a real scalar")
    minimum_std = float(minimum_std)
    if not isfinite(minimum_std) or minimum_std <= 0.0:
        raise ValueError("minimum_std must be finite and strictly positive")

    expected_ids = set(split_manifest.train_ids)
    seen_ids: set[str] = set()
    input_moments: _SampleBalancedMoments | None = None
    target_moments: _SampleBalancedMoments | None = None
    input_channels: tuple[str, ...] | None = None
    target_channels: tuple[str, ...] | None = None

    for sample in training_samples:
        if sample.sample_id not in expected_ids:
            raise ValueError(
                f"sample '{sample.sample_id}' is not declared in SplitManifest.train_ids"
            )
        if sample.sample_id in seen_ids:
            raise ValueError(f"training sample '{sample.sample_id}' was supplied more than once")

        batch = task.pack_and_prepare([sample], catalog)
        if input_channels is None:
            input_channels = batch.input_channels
            target_channels = batch.target_channels
            input_moments = _SampleBalancedMoments(len(input_channels))
            target_moments = _SampleBalancedMoments(len(target_channels))
        elif batch.input_channels != input_channels or batch.target_channels != target_channels:
            raise ValueError("training samples produced inconsistent task channel semantics")

        assert input_moments is not None
        assert target_moments is not None
        input_moments.update(batch.inputs)
        target_moments.update(batch.targets)
        seen_ids.add(sample.sample_id)

    missing_ids = expected_ids - seen_ids
    if missing_ids:
        raise ValueError(
            "training_samples did not contain every declared training sample: "
            f"{sorted(missing_ids)}"
        )
    if input_channels is None or target_channels is None:
        raise ValueError("training_samples must contain at least one sample")

    assert input_moments is not None
    assert target_moments is not None
    input_mean, input_std = input_moments.finalize(input_channels, minimum_std)
    target_mean, target_std = target_moments.finalize(target_channels, minimum_std)

    return TaskStandardizers(
        inputs=ChannelStandardizer(
            channel_names=input_channels,
            mean=input_mean,
            scale=input_std,
            minimum_std=minimum_std,
        ),
        targets=ChannelStandardizer(
            channel_names=target_channels,
            mean=target_mean,
            scale=target_std,
            minimum_std=minimum_std,
        ),
        train_sample_ids=split_manifest.train_ids,
        physical_nondimensionalization=task.physical_nondimensionalization,
    )


class _SampleBalancedMoments:
    """Stable online moments for an equal-weight mixture of physical samples."""

    def __init__(self, channels: int) -> None:
        self.channels = channels
        self.count = 0
        self.mean: torch.Tensor | None = None
        self.m2: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        if values.ndim != 2 or values.shape[1] != self.channels:
            raise ValueError(f"values must have shape [N, {self.channels}]")
        if values.shape[0] <= 0:
            raise ValueError("each training sample must contain at least one node")
        if not values.is_floating_point():
            raise TypeError("statistical scaling requires floating-point values")
        if not torch.isfinite(values).all():
            raise ValueError("statistical scaling values contain NaN or Inf")

        values64 = values.detach().to(dtype=torch.float64)
        sample_mean = values64.mean(dim=0)
        centered = values64 - sample_mean
        sample_variance = centered.square().mean(dim=0)

        if self.count == 0:
            self.count = 1
            self.mean = sample_mean
            self.m2 = sample_variance
            return

        assert self.mean is not None
        assert self.m2 is not None
        new_count = self.count + 1
        delta = sample_mean - self.mean
        self.m2 = self.m2 + sample_variance + delta.square() * (self.count / new_count)
        self.mean = self.mean + delta / new_count
        self.count = new_count

    def finalize(
        self,
        channel_names: tuple[str, ...],
        minimum_std: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count <= 0 or self.mean is None or self.m2 is None:
            raise ValueError("cannot finalize statistical scaling without samples")

        variance = torch.clamp(self.m2 / self.count, min=0.0)
        std = variance.sqrt()
        small = std <= minimum_std
        if torch.any(small):
            bad_channels = [
                channel_names[index]
                for index in torch.nonzero(small, as_tuple=False).flatten().tolist()
            ]
            raise ValueError(
                "cannot standardize zero/near-zero variance channels at "
                f"minimum_std={minimum_std}: {bad_channels}"
            )
        return self.mean, std
