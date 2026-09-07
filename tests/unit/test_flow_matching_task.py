from dataclasses import replace

import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import FlowMatchingTask


class _TimeVelocity(torch.nn.Module):
    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, coords
        return conditioning[batch_index, -1:].expand_as(inputs)


def _batch():
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = FlowMatchingTask(state_fields=("rho", "momentum"), validation_seed=91)
    return task, dataset, task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)


def test_flow_matching_training_problem_matches_straight_path_and_graph_time() -> None:
    task, _, batch = _batch()
    generator = torch.Generator().manual_seed(123)
    problem = task.make_training_problem(batch, generator=generator)

    reference_generator = torch.Generator().manual_seed(123)
    times = torch.rand((batch.num_graphs,), generator=reference_generator)
    source = torch.randn(batch.inputs.shape, generator=reference_generator)
    node_times = times[batch.batch_index].unsqueeze(1)

    torch.testing.assert_close(
        problem.inputs,
        (1.0 - node_times) * source + node_times * batch.inputs,
    )
    torch.testing.assert_close(problem.targets, batch.inputs - source)
    torch.testing.assert_close(problem.conditioning[:, -1], times)
    assert problem.conditioning_names[-1] == "flow_time"
    assert problem.target_channels == tuple(f"d_dt:{name}" for name in batch.input_channels)


def test_flow_matching_validation_is_sample_id_deterministic_across_batch_order() -> None:
    task, dataset, _ = _batch()
    forward = task.make_validation_problem(
        task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    )
    reverse = task.make_validation_problem(
        task.pack_and_prepare([dataset[1], dataset[0]], dataset.field_catalog)
    )

    forward_first = slice(int(forward.ptr[0]), int(forward.ptr[1]))
    reverse_first = slice(int(reverse.ptr[1]), int(reverse.ptr[2]))
    torch.testing.assert_close(forward.inputs[forward_first], reverse.inputs[reverse_first])
    torch.testing.assert_close(forward.targets[forward_first], reverse.targets[reverse_first])
    torch.testing.assert_close(forward.conditioning[0], reverse.conditioning[1])


def test_flow_matching_heun_sampling_uses_raw_time_and_not_clean_state_values() -> None:
    task, _, batch = _batch()
    source = task.sample_standardized(
        _TimeVelocity(),
        batch,
        steps=1,
        solver="euler",
        sampling_seed=44,
    )

    # With v(x,t)=t, Heun integrates the linear-in-time velocity exactly: integral_0^1 t dt=1/2.
    generated = task.sample_standardized(
        _TimeVelocity(),
        batch,
        steps=4,
        solver="heun",
        sampling_seed=44,
    )
    # The one-step Euler call above has t=0, so it returns the deterministic Gaussian source.
    torch.testing.assert_close(generated, source + 0.5)

    altered = replace(batch, inputs=batch.inputs + 100.0, targets=batch.targets + 100.0)
    altered_generated = task.sample_standardized(
        _TimeVelocity(),
        altered,
        steps=4,
        solver="heun",
        sampling_seed=44,
    )
    torch.testing.assert_close(altered_generated, generated)
