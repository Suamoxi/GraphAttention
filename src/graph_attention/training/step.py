"""Reference-correct optimizer stepping across variable computational microbatches."""

from __future__ import annotations

from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import dataclass
from operator import index as operator_index

import torch
import torch.distributed as dist
from torch import nn
from torch.optim import Optimizer

from graph_attention.tasks import NodeRegressionBatch

from .losses import sample_reduced_mse
from .scaling import TaskStandardizers


@dataclass(frozen=True, slots=True)
class OptimizerStepResult:
    """Detached diagnostics for one equal-sample optimizer step."""

    objective: torch.Tensor
    local_sample_count: int
    global_sample_count: int
    microbatch_count: int
    world_size: int


def equal_sample_ddp_backward_scale(
    global_sample_count: int,
    world_size: int,
) -> float:
    """Scale local summed sample losses before DDP's gradient averaging.

    PyTorch DDP averages gradients over ``world_size`` ranks. Multiplying each
    rank-local summed sample loss by ``world_size / global_sample_count`` makes
    the post-DDP gradient equal the gradient of the global equal-sample mean.
    """

    global_count = _positive_count(global_sample_count, "global_sample_count")
    world = _positive_count(world_size, "world_size")
    return world / global_count


def train_equal_sample_optimizer_step(
    model: nn.Module,
    optimizer: Optimizer,
    microbatches: Iterable[NodeRegressionBatch],
    *,
    local_sample_count: int,
    standardizers: TaskStandardizers | None = None,
    autocast_dtype: torch.dtype | None = None,
    grad_scaler: torch.amp.GradScaler | None = None,
) -> OptimizerStepResult:
    """Execute one optimizer step without making microbatch size a sample weight.

    ``local_sample_count`` is the statistical number of physical samples assigned
    to this rank for the optimizer step. It must be known before backward starts.
    In DDP, the function sums that count across ranks once and compensates for
    DDP's gradient averaging. Rank-local microbatch counts may differ: all local
    microbatches except the last run under ``DDP.no_sync()`` so each rank performs
    exactly one gradient synchronization for the optimizer step.
    """

    expected_local_count = _positive_count(local_sample_count, "local_sample_count")
    parameter = _first_trainable_parameter(model)
    device = parameter.device

    if grad_scaler is not None and autocast_dtype is None:
        raise ValueError("grad_scaler requires autocast_dtype to be enabled")
    if autocast_dtype is not None and not autocast_dtype.is_floating_point:
        raise TypeError("autocast_dtype must be floating-point")
    if autocast_dtype is torch.float16 and device.type == "cuda" and grad_scaler is None:
        raise ValueError("CUDA float16 training requires an explicit GradScaler")

    iterator = iter(microbatches)
    try:
        current = next(iterator)
        local_has_batch = 1
    except StopIteration:
        current = None
        local_has_batch = 0

    world_size, global_sample_count = _distributed_counts(
        expected_local_count,
        local_has_batch=local_has_batch,
        device=device,
    )
    if local_has_batch == 0:
        raise ValueError("microbatches must contain at least one local batch")

    if world_size > 1 and not callable(getattr(model, "no_sync", None)):
        raise TypeError("distributed training requires a DistributedDataParallel model")

    backward_scale = equal_sample_ddp_backward_scale(global_sample_count, world_size)
    optimizer.zero_grad(set_to_none=True)

    local_loss_sum = torch.zeros((), dtype=torch.float64, device=device)
    seen_samples = 0
    microbatch_count = 0

    try:
        assert current is not None
        while True:
            try:
                following = next(iterator)
                is_last = False
            except StopIteration:
                following = None
                is_last = True

            seen_samples += current.num_graphs
            prepared = standardizers.transform(current) if standardizers is not None else current
            sync_context = model.no_sync() if world_size > 1 and not is_last else nullcontext()

            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=autocast_dtype,
                    enabled=autocast_dtype is not None,
                ):
                    predictions = model(
                        prepared.inputs,
                        edge_index=prepared.edge_index,
                        batch_index=prepared.batch_index,
                        conditioning=prepared.conditioning,
                    )
                    aggregate = sample_reduced_mse(
                        predictions,
                        prepared.targets,
                        prepared.ptr,
                        node_weights=prepared.node_weights,
                    )
                    backward_loss = aggregate.loss_sum * backward_scale

                if grad_scaler is None:
                    backward_loss.backward()
                else:
                    grad_scaler.scale(backward_loss).backward()

            local_loss_sum = local_loss_sum + aggregate.loss_sum.detach().to(torch.float64)
            microbatch_count += 1

            if is_last:
                break
            assert following is not None
            current = following

        _validate_observed_sample_count(
            observed=seen_samples,
            expected=expected_local_count,
            world_size=world_size,
            device=device,
        )

        if grad_scaler is None:
            optimizer.step()
        else:
            grad_scaler.step(optimizer)
            grad_scaler.update()
    except Exception:
        optimizer.zero_grad(set_to_none=True)
        raise

    global_loss_sum = local_loss_sum.clone()
    if world_size > 1:
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)

    return OptimizerStepResult(
        objective=(global_loss_sum / global_sample_count).detach(),
        local_sample_count=expected_local_count,
        global_sample_count=global_sample_count,
        microbatch_count=microbatch_count,
        world_size=world_size,
    )


def _distributed_counts(
    local_sample_count: int,
    *,
    local_has_batch: int,
    device: torch.device,
) -> tuple[int, int]:
    if not dist.is_available() or not dist.is_initialized():
        return 1, local_sample_count

    world_size = dist.get_world_size()
    count = torch.tensor(local_sample_count, dtype=torch.long, device=device)
    has_batch = torch.tensor(local_has_batch, dtype=torch.long, device=device)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    dist.all_reduce(has_batch, op=dist.ReduceOp.MIN)
    if int(has_batch) == 0:
        raise ValueError("every DDP rank must have at least one microbatch per optimizer step")
    return world_size, int(count)


def _validate_observed_sample_count(
    *,
    observed: int,
    expected: int,
    world_size: int,
    device: torch.device,
) -> None:
    mismatch = int(observed != expected)
    if world_size > 1:
        mismatch_tensor = torch.tensor(mismatch, dtype=torch.long, device=device)
        dist.all_reduce(mismatch_tensor, op=dist.ReduceOp.MAX)
        mismatch = int(mismatch_tensor)
    if mismatch:
        raise ValueError(
            "local_sample_count does not match supplied microbatches: "
            f"expected {expected}, observed {observed} on this rank"
        )


def _first_trainable_parameter(model: nn.Module) -> torch.nn.Parameter:
    try:
        return next(parameter for parameter in model.parameters() if parameter.requires_grad)
    except StopIteration as exc:
        raise ValueError("model must contain at least one trainable parameter") from exc


def _positive_count(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        count = operator_index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if count <= 0:
        raise ValueError(f"{name} must be positive")
    return count
