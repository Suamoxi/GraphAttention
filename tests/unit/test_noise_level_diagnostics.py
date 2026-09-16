from __future__ import annotations

import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks import VPSDEDenoisingTask
from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask
from scripts.compare_noise_level_denoising import (
    _diagnostic_levels,
    _problem_at_fraction,
    _reconstruct_x0,
)


def test_diagnostic_levels_require_strictly_increasing_open_zero_closed_one() -> None:
    assert _diagnostic_levels([0.001, 0.1, 0.5, 1.0]) == [0.001, 0.1, 0.5, 1.0]

    for values in ([0.0, 0.5], [0.1, 0.1], [0.2, 0.1], [0.5, 1.1]):
        try:
            _diagnostic_levels(values)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected invalid diagnostic levels: {values}")


def test_discrete_and_continuous_forward_problems_reconstruct_clean_state_with_exact_noise() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    tasks = (
        LinearBetaDiffusionDenoisingTask(
            state_fields=("rho", "momentum"),
            timesteps=1000,
            beta_start=1.0e-4,
            beta_end=2.0e-2,
        ),
        VPSDEDenoisingTask(
            state_fields=("rho", "momentum"),
            beta_min=0.1,
            beta_max=20.0,
            training_eps=1.0e-5,
        ),
    )

    generator = torch.Generator().manual_seed(91)
    for task in tasks:
        batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
        noise = torch.randn(batch.inputs.shape, generator=generator, dtype=batch.inputs.dtype)
        problem, _, alpha, sigma, metadata = _problem_at_fraction(task, batch, noise, 0.5)
        reconstructed = _reconstruct_x0(
            problem.inputs,
            noise,
            alpha,
            sigma,
            batch.batch_index,
        )
        torch.testing.assert_close(reconstructed, batch.inputs, rtol=2.0e-5, atol=2.0e-5)
        assert 0.0 < metadata["alpha"] <= 1.0
        assert 0.0 <= metadata["sigma"] <= 1.0

    discrete = tasks[0].pack_and_prepare([dataset[0]], dataset.field_catalog)
    noise = torch.zeros_like(discrete.inputs)
    _, conditioning, _, _, metadata = _problem_at_fraction(tasks[0], discrete, noise, 0.5)
    assert metadata["process"] == "discrete_ddpm"
    assert metadata["discrete_timestep"] == 500
    torch.testing.assert_close(conditioning, torch.tensor([0.5], dtype=conditioning.dtype))

    continuous = tasks[1].pack_and_prepare([dataset[0]], dataset.field_catalog)
    noise = torch.zeros_like(continuous.inputs)
    _, conditioning, _, _, metadata = _problem_at_fraction(tasks[1], continuous, noise, 0.5)
    assert metadata["process"] == "continuous_vp_sde"
    assert metadata["discrete_timestep"] is None
    torch.testing.assert_close(conditioning, torch.tensor([0.5], dtype=conditioning.dtype))


def test_reconstruct_x0_supports_different_graph_noise_levels() -> None:
    clean = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    batch_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    alpha = torch.tensor([0.8, 0.4])
    sigma = torch.sqrt(1.0 - alpha.square())
    epsilon = torch.tensor([[0.5], [-0.5], [1.0], [-1.0]])
    noisy = alpha[batch_index, None] * clean + sigma[batch_index, None] * epsilon

    reconstructed = _reconstruct_x0(noisy, epsilon, alpha, sigma, batch_index)
    torch.testing.assert_close(reconstructed, clean)
