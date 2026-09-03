"""Per-sample loss reductions for variable-size packed graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class SampleLossAggregate:
    """Per-sample losses plus their equal-sample numerator."""

    per_sample: torch.Tensor
    loss_sum: torch.Tensor
    sample_count: int

    @property
    def mean(self) -> torch.Tensor:
        return self.loss_sum / self.sample_count


def sample_reduced_mse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    ptr: torch.Tensor,
    *,
    node_weights: torch.Tensor | None = None,
) -> SampleLossAggregate:
    """Reduce MSE inside each physical sample before combining samples.

    Channel errors are averaged equally at each node. Prediction/target dtypes
    are promoted explicitly for the loss calculation so autocast outputs can be
    compared against full-precision targets. Spatial reduction then uses supplied
    non-negative node weights when present, otherwise an unweighted node mean.
    The returned ``loss_sum`` is the sum of physical-sample losses, not a
    node-weighted global mean.
    """

    _validate_prediction_target(predictions, targets)
    sample_count = _validate_ptr(ptr, predictions.shape[0])

    loss_dtype = torch.promote_types(predictions.dtype, targets.dtype)
    difference = predictions.to(loss_dtype) - targets.to(loss_dtype)
    node_loss = difference.square().mean(dim=1)
    weights = _validate_weights(node_weights, predictions.shape[0], predictions.device)

    per_sample: list[torch.Tensor] = []
    for graph_index in range(sample_count):
        start = int(ptr[graph_index])
        stop = int(ptr[graph_index + 1])
        graph_loss = node_loss[start:stop]

        if weights is None:
            per_sample.append(graph_loss.mean())
            continue

        graph_weights = weights[start:stop]
        denominator = graph_weights.sum()
        if denominator <= 0:
            raise ValueError(f"node_weights sum to zero for graph {graph_index}")
        per_sample.append((graph_weights * graph_loss).sum() / denominator)

    per_sample_tensor = torch.stack(per_sample)
    if not torch.isfinite(per_sample_tensor).all():
        raise ValueError("per-sample loss contains NaN or Inf")

    return SampleLossAggregate(
        per_sample=per_sample_tensor,
        loss_sum=per_sample_tensor.sum(),
        sample_count=sample_count,
    )


def _validate_prediction_target(
    predictions: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    if not isinstance(predictions, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("predictions and targets must be torch.Tensor values")
    if predictions.ndim != 2 or targets.ndim != 2:
        raise ValueError("predictions and targets must have shape [N, C]")
    if predictions.shape != targets.shape:
        raise ValueError(
            f"predictions and targets must share one shape, got "
            f"{tuple(predictions.shape)} and {tuple(targets.shape)}"
        )
    if predictions.shape[0] <= 0 or predictions.shape[1] <= 0:
        raise ValueError("predictions and targets must contain at least one node and channel")
    if not predictions.is_floating_point() or not targets.is_floating_point():
        raise TypeError("predictions and targets must use floating-point dtypes")
    if predictions.device != targets.device:
        raise ValueError("predictions and targets must be on the same device")
    if not torch.isfinite(predictions).all() or not torch.isfinite(targets).all():
        raise ValueError("predictions and targets must contain only finite values")


def _validate_ptr(ptr: torch.Tensor, num_nodes: int) -> int:
    if not isinstance(ptr, torch.Tensor):
        raise TypeError("ptr must be a torch.Tensor")
    if ptr.ndim != 1 or ptr.numel() < 2:
        raise ValueError("ptr must have shape [B+1] with at least one graph")
    if ptr.dtype != torch.long:
        raise TypeError("ptr must use torch.long indices")
    if int(ptr[0]) != 0 or int(ptr[-1]) != num_nodes:
        raise ValueError("ptr must start at zero and end at the total node count")
    counts = ptr[1:] - ptr[:-1]
    if torch.any(counts <= 0):
        raise ValueError("ptr must describe non-empty graphs with strictly increasing offsets")
    return ptr.numel() - 1


def _validate_weights(
    node_weights: torch.Tensor | None,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor | None:
    if node_weights is None:
        return None
    if not isinstance(node_weights, torch.Tensor):
        raise TypeError("node_weights must be a torch.Tensor when supplied")
    if node_weights.ndim != 1 or node_weights.shape[0] != num_nodes:
        raise ValueError(f"node_weights must have shape [{num_nodes}]")
    if not node_weights.is_floating_point():
        raise TypeError("node_weights must use a floating-point dtype")
    if node_weights.device != device:
        raise ValueError("node_weights must be on the prediction device")
    if not torch.isfinite(node_weights).all():
        raise ValueError("node_weights contain NaN or Inf")
    if torch.any(node_weights < 0):
        raise ValueError("node_weights must be non-negative")
    return node_weights
