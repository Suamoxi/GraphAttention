import pytest
import torch

from graph_attention.data import SplitManifest, SyntheticMeshDataset
from graph_attention.models import NodeLinearBaseline
from graph_attention.tasks import NodeRegressionTask
from graph_attention.training import (
    equal_sample_ddp_backward_scale,
    fit_train_standardizers,
    train_equal_sample_optimizer_step,
)


def _setup_samples(count: int = 4):
    dataset = SyntheticMeshDataset(num_samples=count, spatial_dim=2, seed=23)
    samples = [dataset[index] for index in range(count)]
    task = NodeRegressionTask(input_fields=("momentum",), target_fields=("rho",))
    split = SplitManifest(train_ids=tuple(sample.sample_id for sample in samples))
    scalers = fit_train_standardizers(task, samples, dataset.field_catalog, split)
    return dataset, samples, task, scalers


def test_optimizer_step_is_invariant_to_microbatch_partitioning() -> None:
    dataset, samples, task, scalers = _setup_samples()
    full_batch = task.pack_and_prepare(samples, dataset.field_catalog)
    split_batches = [
        task.pack_and_prepare(samples[:1], dataset.field_catalog),
        task.pack_and_prepare(samples[1:3], dataset.field_catalog),
        task.pack_and_prepare(samples[3:], dataset.field_catalog),
    ]

    torch.manual_seed(11)
    reference = NodeLinearBaseline(in_channels=2, out_channels=1)
    partitioned = NodeLinearBaseline(in_channels=2, out_channels=1)
    partitioned.load_state_dict(reference.state_dict())

    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.05)
    partitioned_optimizer = torch.optim.SGD(partitioned.parameters(), lr=0.05)

    reference_result = train_equal_sample_optimizer_step(
        reference,
        reference_optimizer,
        [full_batch],
        local_sample_count=len(samples),
        standardizers=scalers,
    )
    partitioned_result = train_equal_sample_optimizer_step(
        partitioned,
        partitioned_optimizer,
        split_batches,
        local_sample_count=len(samples),
        standardizers=scalers,
    )

    torch.testing.assert_close(reference.linear.weight, partitioned.linear.weight)
    torch.testing.assert_close(reference.linear.bias, partitioned.linear.bias)
    torch.testing.assert_close(reference_result.objective, partitioned_result.objective)
    assert partitioned_result.microbatch_count == 3


def test_optimizer_step_rejects_wrong_declared_sample_count_without_stepping() -> None:
    dataset, samples, task, scalers = _setup_samples(count=2)
    batch = task.pack_and_prepare(samples, dataset.field_catalog)
    model = NodeLinearBaseline(in_channels=2, out_channels=1)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="does not match supplied microbatches"):
        train_equal_sample_optimizer_step(
            model,
            optimizer,
            [batch],
            local_sample_count=3,
            standardizers=scalers,
        )

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, initial[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_optimizer_step_supports_cpu_bfloat16_autocast() -> None:
    dataset, samples, task, scalers = _setup_samples(count=2)
    batch = task.pack_and_prepare(samples, dataset.field_catalog)
    model = NodeLinearBaseline(in_channels=2, out_channels=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    result = train_equal_sample_optimizer_step(
        model,
        optimizer,
        [batch],
        local_sample_count=2,
        standardizers=scalers,
        autocast_dtype=torch.bfloat16,
    )

    assert torch.isfinite(result.objective)
    assert result.world_size == 1


def test_equal_sample_ddp_backward_scale_matches_global_mean_formula() -> None:
    rank0_loss_sum = 6.0
    rank1_loss_sum = 9.0
    global_sample_count = 5
    world_size = 2
    scale = equal_sample_ddp_backward_scale(global_sample_count, world_size)

    ddp_averaged_scaled_gradient_factor = (
        scale * rank0_loss_sum + scale * rank1_loss_sum
    ) / world_size

    assert ddp_averaged_scaled_gradient_factor == pytest.approx(
        (rank0_loss_sum + rank1_loss_sum) / global_sample_count
    )
