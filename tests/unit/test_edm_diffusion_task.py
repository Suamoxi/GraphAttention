from dataclasses import replace

import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import EDMDenoisingTask, karras_sigma_schedule


class _ZeroEDMOutput(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        inputs: torch.Tensor,
        *,
        edge_index: torch.Tensor,
        coords: torch.Tensor,
        batch_index: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        del edge_index, coords, batch_index, conditioning
        self.calls += 1
        return torch.zeros_like(inputs)


def _task_and_batch() -> tuple[EDMDenoisingTask, SyntheticMeshDataset, object]:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = EDMDenoisingTask(
        state_fields=("rho", "momentum"),
        sigma_data=1.0,
        p_mean=-0.5,
        p_std=1.2,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, dataset, batch


def test_karras_sigma_schedule_has_expected_endpoints_and_is_monotone() -> None:
    sigmas = karras_sigma_schedule(
        18,
        sigma_min=0.004,
        sigma_max=160.0,
        rho=7.0,
    )

    assert sigmas.shape == (19,)
    assert float(sigmas[0]) == pytest.approx(160.0)
    assert float(sigmas[-2]) == pytest.approx(0.004)
    assert float(sigmas[-1]) == 0.0
    assert bool(torch.all(sigmas[:-2] > sigmas[1:-1]))


def test_edm_problem_uses_karras_preconditioning_and_log_sigma_conditioning() -> None:
    task, _, batch = _task_and_batch()
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(123),
    )

    node_sigma = problem.sigmas[batch.batch_index].unsqueeze(1)
    expected_noisy = problem.clean_state + node_sigma * (
        (problem.noisy_state - problem.clean_state) / node_sigma
    )
    c_in = torch.rsqrt(node_sigma.square() + task.sigma_data**2)

    torch.testing.assert_close(problem.noisy_state, expected_noisy)
    torch.testing.assert_close(problem.model_batch.inputs, c_in * problem.noisy_state)
    torch.testing.assert_close(
        problem.model_batch.conditioning[:, -1],
        torch.log(problem.sigmas) / 4.0,
    )
    assert problem.model_batch.conditioning_names[-1] == "edm_log_sigma_over_4"


def test_edm_loss_matches_weighted_denoising_definition() -> None:
    task, _, batch = _task_and_batch()
    problem = task.make_training_problem(
        batch,
        generator=torch.Generator().manual_seed(321),
    )
    raw = torch.zeros_like(problem.clean_state)
    result = task.edm_loss(raw, problem)

    node_sigma = problem.sigmas[batch.batch_index].unsqueeze(1)
    c_skip = task.sigma_data**2 / (node_sigma.square() + task.sigma_data**2)
    denoised = c_skip * problem.noisy_state
    weight = (node_sigma.square() + task.sigma_data**2) / (
        node_sigma * task.sigma_data
    ).square()
    node_loss = (weight * (denoised - problem.clean_state).square()).mean(dim=1)
    expected = torch.stack(
        [
            node_loss[int(batch.ptr[index]) : int(batch.ptr[index + 1])].mean()
            for index in range(batch.num_graphs)
        ]
    )

    torch.testing.assert_close(result.per_sample, expected)


def test_edm_validation_is_deterministic() -> None:
    task, _, batch = _task_and_batch()
    first = task.make_validation_problem(batch)
    second = task.make_validation_problem(batch)

    torch.testing.assert_close(first.sigmas, second.sigmas)
    torch.testing.assert_close(first.noisy_state, second.noisy_state)
    torch.testing.assert_close(first.model_batch.inputs, second.model_batch.inputs)


def test_edm_sampling_is_independent_of_clean_reference_values() -> None:
    task, _, batch = _task_and_batch()
    model = _ZeroEDMOutput()
    generated = task.sample_standardized(
        model,
        batch,
        steps=4,
        sigma_min=0.01,
        sigma_max=4.0,
        rho=3.0,
        solver="heun",
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )

    altered = replace(batch, inputs=batch.inputs + 100.0, targets=batch.targets + 100.0)
    altered_generated = task.sample_standardized(
        model,
        altered,
        steps=4,
        sigma_min=0.01,
        sigma_max=4.0,
        rho=3.0,
        solver="heun",
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )

    torch.testing.assert_close(altered_generated, generated)
    assert model.calls == 14


def test_edm_sampling_rejects_invalid_sigma_range() -> None:
    task, _, batch = _task_and_batch()
    model = _ZeroEDMOutput()

    with pytest.raises(ValueError, match="sigma_max must be greater than sigma_min"):
        task.sample_standardized(
            model,
            batch,
            steps=4,
            sigma_min=1.0,
            sigma_max=1.0,
        )
