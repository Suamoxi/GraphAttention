import pytest
import torch

from graph_attention.tasks.linear_diffusion import LinearBetaDiffusionDenoisingTask


def _task(**overrides: float | int) -> LinearBetaDiffusionDenoisingTask:
    kwargs = {
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
