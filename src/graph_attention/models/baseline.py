"""Minimal task-plumbing baseline models."""

from __future__ import annotations

from operator import index as operator_index

import torch
from torch import nn


class NodeLinearBaseline(nn.Module):
    """Independent affine map at every node, with optional global conditioning.

    This model intentionally ignores coordinates and graph connectivity. It is a
    null geometric baseline for validating task semantics, batching equivalence,
    and node-renumbering equivariance before sparse message passing is added.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conditioning_channels: int = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = _channel_count(in_channels, "in_channels", positive=True)
        self.out_channels = _channel_count(out_channels, "out_channels", positive=True)
        self.conditioning_channels = _channel_count(
            conditioning_channels,
            "conditioning_channels",
            positive=False,
        )
        if not isinstance(bias, bool):
            raise TypeError("bias must be a bool")

        self.linear = nn.Linear(
            self.in_channels + self.conditioning_channels,
            self.out_channels,
            bias=bias,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        batch_index: torch.Tensor | None = None,
        conditioning: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if inputs.ndim != 2:
            raise ValueError("inputs must have shape [N, C]")
        if not inputs.is_floating_point():
            raise TypeError("inputs must use a floating-point dtype")
        if inputs.shape[1] != self.in_channels:
            raise ValueError(
                f"inputs have {inputs.shape[1]} channels, expected {self.in_channels}"
            )

        if self.conditioning_channels == 0:
            if conditioning is not None:
                if conditioning.ndim != 2 or conditioning.shape[1] != 0:
                    raise ValueError(
                        "conditioning must have zero columns when conditioning_channels=0"
                    )
            features = inputs
        else:
            if batch_index is None or conditioning is None:
                raise ValueError(
                    "batch_index and conditioning are required when conditioning_channels > 0"
                )
            _validate_conditioning(
                inputs,
                batch_index,
                conditioning,
                expected_channels=self.conditioning_channels,
            )
            features = torch.cat((inputs, conditioning[batch_index]), dim=1)

        return self.linear(features)


def _channel_count(value: int, name: str, *, positive: bool) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer channel count")
    try:
        count = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer channel count") from exc
    if positive and count <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and count < 0:
        raise ValueError(f"{name} must be non-negative")
    return count


def _validate_conditioning(
    inputs: torch.Tensor,
    batch_index: torch.Tensor,
    conditioning: torch.Tensor,
    *,
    expected_channels: int,
) -> None:
    if batch_index.ndim != 1 or batch_index.shape[0] != inputs.shape[0]:
        raise ValueError("batch_index must have shape [N]")
    if batch_index.dtype != torch.long:
        raise TypeError("batch_index must use torch.long indices")
    if batch_index.device != inputs.device:
        raise ValueError("batch_index and inputs must be on the same device")

    if conditioning.ndim != 2 or conditioning.shape[1] != expected_channels:
        raise ValueError(f"conditioning must have shape [B, {expected_channels}]")
    if not conditioning.is_floating_point():
        raise TypeError("conditioning must use a floating-point dtype")
    if conditioning.dtype != inputs.dtype:
        raise TypeError("conditioning and inputs must share one dtype")
    if conditioning.device != inputs.device:
        raise ValueError("conditioning and inputs must be on the same device")

    if batch_index.numel() > 0:
        if int(batch_index.min()) < 0:
            raise ValueError("batch_index contains a negative graph index")
        if int(batch_index.max()) >= conditioning.shape[0]:
            raise ValueError("batch_index references a graph outside conditioning")
