from dataclasses import replace

import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import (
    ImprovedDiffusionDenoisingTask,
    LossSecondMomentTimestepSampler,
)


class _ZeroImprovedOutput(torch.nn.Module):
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
        return torch.zeros(
            (inputs.shape[0], 2 * inputs.shape[1]),
            device=inputs.device,
            dtype=inputs.dtype,
        )


def _task_and_batch(timesteps: int = 10):
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = ImprovedDiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=timesteps,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    return task, dataset, batch


def test_improved_diffusion_model_probe_doubles_output_channels() -> None:
    task, _, batch = _task_and_batch()
    problem = task.make_validation_problem(batch)
    probe = task.make_model_probe(problem)

    assert probe.inputs.shape[1] == batch.inputs.shape[1]
    assert probe.targets.shape[1] == 2 * batch.inputs.shape[1]
    assert len(probe.target_channels) == 2 * len(batch.input_channels)
    variance_channels = probe.target_channels[len(batch.input_channels) :]
    assert all(name.startswith("variance_range:") for name in variance_channels)


def test_hybrid_loss_detaches_epsilon_from_vlb_branch() -> None:
    task, _, batch = _task_and_batch(timesteps=10)
    timesteps = torch.tensor([5, 7], dtype=torch.long)
    problem = task.make_training_problem(
        batch,
        timesteps=timesteps,
        generator=torch.Generator().manual_seed(123),
    )

    epsilon_hat = problem.noise.detach().clone().requires_grad_(True)
    variance_values = torch.zeros_like(problem.noise, requires_grad=True)
    model_output = torch.cat((epsilon_hat, variance_values), dim=1)
    loss = task.hybrid_loss(model_output, problem)
    loss.mean.backward()

    torch.testing.assert_close(epsilon_hat.grad, torch.zeros_like(epsilon_hat.grad))
    assert variance_values.grad is not None
    assert torch.isfinite(variance_values.grad).all()
    assert float(variance_values.grad.abs().sum()) > 0.0
    torch.testing.assert_close(loss.simple_per_sample, torch.zeros_like(loss.simple_per_sample))
    assert torch.isfinite(loss.vlb_per_sample).all()


def test_loss_second_moment_sampler_is_uniform_before_warmup() -> None:
    sampler = LossSecondMomentTimestepSampler(
        4,
        history_per_timestep=2,
        uniform_probability=0.01,
    )
    probabilities = sampler.probabilities()
    torch.testing.assert_close(probabilities, torch.full((4,), 0.25, dtype=torch.float64))

    timesteps, importance = sampler.sample(
        20,
        generator=torch.Generator().manual_seed(5),
        device=torch.device("cpu"),
    )
    assert timesteps.min() >= 1
    assert timesteps.max() <= 4
    torch.testing.assert_close(importance, torch.ones_like(importance))


def test_loss_second_moment_sampler_weights_high_loss_timesteps_more() -> None:
    sampler = LossSecondMomentTimestepSampler(
        2,
        history_per_timestep=2,
        uniform_probability=0.0,
    )
    sampler.update(
        torch.tensor([1, 1, 2, 2], dtype=torch.long),
        torch.tensor([1.0, 1.0, 4.0, 4.0]),
    )

    probabilities = sampler.probabilities()
    assert sampler.warmed_up
    assert probabilities[1] > probabilities[0]
    assert probabilities.sum() == pytest.approx(1.0)


def test_improved_ancestral_sampling_does_not_use_clean_reference_values() -> None:
    task, _, batch = _task_and_batch(timesteps=4)
    model = _ZeroImprovedOutput()
    generated = task.sample_standardized(
        model,
        batch,
        steps=4,
        eta=1.0,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )

    altered = replace(batch, inputs=batch.inputs + 100.0, targets=batch.targets + 100.0)
    altered_generated = task.sample_standardized(
        model,
        altered,
        steps=4,
        eta=1.0,
        sampling_seed=44,
        sampling_keys=("gen_000000", "gen_000001"),
    )
    torch.testing.assert_close(altered_generated, generated)


def test_improved_sampler_rejects_non_full_or_non_ancestral_sampling() -> None:
    task, _, batch = _task_and_batch(timesteps=4)
    model = _ZeroImprovedOutput()

    with pytest.raises(ValueError, match="exact full learned-variance ancestral chain"):
        task.sample_standardized(model, batch, steps=2, eta=1.0)
    with pytest.raises(ValueError, match="exact full learned-variance ancestral chain"):
        task.sample_standardized(model, batch, steps=4, eta=0.0)
