import pytest
import torch

from graph_attention.data import SyntheticMeshDataset
from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask


def _task(**overrides: object) -> LinearBetaDiffusionDenoisingTask:
    kwargs: dict[str, object] = {
        "state_fields": ("rho", "momentum"),
        "timesteps": 4,
        "beta_start": 0.1,
        "beta_end": 0.4,
        "validation_seed": 91,
    }
    kwargs.update(overrides)
    return LinearBetaDiffusionDenoisingTask(**kwargs)


def test_linear_beta_schedule_matches_configured_betas_without_clipping() -> None:
    task = _task()
    expected_betas = torch.linspace(0.1, 0.4, 4, dtype=torch.float64)
    expected_alpha_bar = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            torch.cumprod(1.0 - expected_betas, dim=0),
        )
    )

    torch.testing.assert_close(task._alpha_bar_cpu, expected_alpha_bar)
    reconstructed_betas = 1.0 - task._alpha_bar_cpu[1:] / task._alpha_bar_cpu[:-1]
    torch.testing.assert_close(reconstructed_betas, expected_betas)
    assert task.noise_schedule == "linear_beta"


def test_reference_linear_schedule_has_finite_terminal_transition() -> None:
    task = LinearBetaDiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        timesteps=1000,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
    )
    final_beta = 1.0 - task._alpha_bar_cpu[-1] / task._alpha_bar_cpu[-2]

    assert float(final_beta) == pytest.approx(2.0e-2, rel=1.0e-12)
    assert float(task._alpha_bar_cpu[-1]) == pytest.approx(4.035829765e-5, rel=1.0e-9)


def test_linear_beta_schedule_rejects_invalid_endpoints() -> None:
    with pytest.raises(ValueError, match="0 < beta_start <= beta_end < 1"):
        _task(beta_start=0.0)
    with pytest.raises(ValueError, match="0 < beta_start <= beta_end < 1"):
        _task(beta_start=0.5, beta_end=0.4)
    with pytest.raises(ValueError, match="0 < beta_start <= beta_end < 1"):
        _task(beta_end=1.0)


class _ZeroEpsilon(torch.nn.Module):
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
        return torch.zeros_like(inputs)


def test_continuous_embedding_recovers_discrete_ddpm_marginals_on_grid() -> None:
    task = _task()
    times = torch.arange(0, task.timesteps + 1, dtype=torch.float64) / task.timesteps

    alpha, sigma = task.continuous_marginal_coefficients(times)

    torch.testing.assert_close(alpha.square(), task._alpha_bar_cpu)
    torch.testing.assert_close(sigma.square(), 1.0 - task._alpha_bar_cpu)


def test_continuous_beta_rate_integrates_each_discrete_transition_exactly() -> None:
    task = _task()
    times = torch.arange(1, task.timesteps + 1, dtype=torch.float64) / task.timesteps
    beta_rate = task.continuous_beta_rate(times)
    reconstructed_beta = 1.0 - torch.exp(-beta_rate / task.timesteps)
    expected_beta = 1.0 - task._alpha_bar_cpu[1:] / task._alpha_bar_cpu[:-1]

    torch.testing.assert_close(reconstructed_beta, expected_beta)


def test_linear_ddpm_reverse_sde_sampling_is_deterministic_for_fixed_keys() -> None:
    dataset = SyntheticMeshDataset(num_samples=2, spatial_dim=2, seed=7)
    task = LinearBetaDiffusionDenoisingTask(
        state_fields=("rho", "momentum"),
        physical_nondimensionalization=False,
        timesteps=1000,
        beta_start=1.0e-4,
        beta_end=2.0e-2,
        validation_seed=91,
    )
    batch = task.pack_and_prepare([dataset[0], dataset[1]], dataset.field_catalog)
    model = _ZeroEpsilon()
    kwargs = {
        "steps": 5,
        "method": "reverse_sde",
        "solver": "euler_maruyama",
        "sampling_eps": 1.0e-3,
        "final_denoise": True,
        "sampling_seed": 5678,
        "sampling_keys": ("gen_000000", "gen_000001"),
    }

    first = task.sample_standardized(model, batch, **kwargs)
    second = task.sample_standardized(model, batch, **kwargs)

    torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
    assert task.sampler_name(
        steps=50,
        method="reverse_sde",
        solver="euler_maruyama",
        final_denoise=True,
    ) == "linear_ddpm_reverse_sde_euler_maruyama_steps50_denoise"


def test_linear_ddpm_legacy_sampler_contract_is_preserved() -> None:
    task = _task()

    assert task.sampler_name(steps=4, eta=1.0) == "ddpm_ancestral"
