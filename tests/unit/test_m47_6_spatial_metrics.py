"""Synthetic-field and boundary tests for the M47.6 spatial diagnostics."""
from types import SimpleNamespace

import numpy as np
import torch

from scripts.m47_6_spatial_diagnostics import heun_trajectory
import pytest

from scripts.m47_6_spatial_metrics import (
    helmholtz_spectrum, interior_divcurl, physical_velocity,
    standardized_momentum_diagnostics,
)


def _grid(n=32):
    return SimpleNamespace(
        shape=(n, n, n), source_shape=(n+1, n+1, n+1),
        spacing=(1/n, 1/n, 1/n), k_nyquist_min=np.pi*n,
        periodic_endpoint_mode="drop_max",
        linear_indices=np.arange((n+1)**3),
    )


def _state(velocity, grid):
    full = np.empty((*grid.source_shape, 3))
    full[:-1, :-1, :-1] = velocity
    full[-1, :, :] = full[0, :, :]
    full[:, -1, :] = full[:, 0, :]
    full[:, :, -1] = full[:, :, 0]
    state = np.zeros((int(np.prod(grid.source_shape)), 5))
    state[:, 0] = 1.0
    state[:, 1:4] = full.reshape(-1, 3)
    state[:, 4] = 100.0
    return state


def test_solenoidal_periodic_field_has_zero_divergence():
    grid = _grid()
    x, y, z = np.meshgrid(*[np.arange(32)/32]*3, indexing="ij")
    velocity = np.stack([np.sin(2*np.pi*y), np.sin(2*np.pi*z),
                         np.sin(2*np.pi*x)], axis=-1)
    physical = physical_velocity(_state(velocity, grid), grid, 1.0)
    assert np.max(np.abs(physical - velocity)) < 1e-12
    derivative = interior_divcurl(physical, grid.spacing)
    assert derivative["2"]["rms_divergence_s_minus_1"] < 1e-10
    spec = helmholtz_spectrum(physical, grid)
    assert spec["fraction_dilatational_all_modes"] < 1e-12
    assert spec["max_relative_shell_closure_error"] < 1e-12


def test_potential_field_is_dilatational_and_curl_free():
    grid = _grid()
    x, y, z = np.meshgrid(*[np.arange(32)/32]*3, indexing="ij")
    velocity = np.stack([np.sin(2*np.pi*x),np.zeros_like(x),np.zeros_like(x)],axis=-1)
    derivative = interior_divcurl(velocity, grid.spacing)
    assert derivative["2"]["rms_vorticity_s_minus_1"] < 1e-10
    spec = helmholtz_spectrum(velocity, grid)
    assert spec["fraction_dilatational_all_modes"] > 1 - 1e-12
    assert spec["max_relative_shell_closure_error"] < 1e-12
    assert helmholtz_spectrum(velocity, grid, window="hann")["fraction_dilatational_all_modes"] is not None


def test_seam_artifact_does_not_leak_into_trimmed_gradient():
    grid = _grid()
    velocity = np.zeros((*grid.shape,3))
    velocity[0, :, :, 0] = 5.0
    metrics = interior_divcurl(velocity, grid.spacing)
    assert metrics["0"]["rms_divergence_s_minus_1"] > 0
    assert metrics["2"]["rms_divergence_s_minus_1"] == 0


def test_standardized_intermediate_allows_nonphysical_density():
    grid = _grid(8)
    q = np.ones((int(np.prod(grid.source_shape)),5))
    q[:,0] = -3  # intermediate FM states need not be physically admissible
    mid = standardized_momentum_diagnostics(q,grid,k_bins=4)
    assert np.max(np.abs(mid["E_standardized_momentum_per_rad_per_m"])) < 1e-12
    with pytest.raises(ValueError,match="strictly positive density"):
        physical_velocity(q,grid,1.0)

class _CanonicalToyFlowTask:
    """Small deterministic two-evaluation Heun sampler for hook regression."""

    def sample_standardized(
        self, model, batch, *, steps, method, solver, sampling_eps,
        final_denoise, sampling_seed, sampling_keys,
    ):
        assert (method, solver, sampling_eps, final_denoise) == (
            "flow_ode", "heun", 0.0, False,
        )
        assert sampling_keys == ("matched-latent",)
        state = torch.randn(
            batch.inputs.shape,
            generator=torch.Generator().manual_seed(sampling_seed),
        )
        dt = 1.0 / steps
        for _ in range(steps):
            k1 = model(state, conditioning=None)
            k2 = model(state + dt * k1, conditioning=None)
            state = state + 0.5 * dt * (k1 + k2)
        return state


class _ToyVelocityModel(torch.nn.Module):
    def forward(self, state, *, conditioning):
        return -0.08 * state + 0.02


def test_trajectory_observes_canonical_heun_without_resampling():
    grid = _grid(8)
    batch = SimpleNamespace(num_graphs=1, inputs=torch.zeros((9**3, 5)))
    model = _ToyVelocityModel()
    task = _CanonicalToyFlowTask()
    state, report = heun_trajectory(
        task, model, batch, steps=50, seed=23,
        key="matched-latent", grid=grid, k_bins=4,
    )
    canonical = task.sample_standardized(
        model, batch, steps=50, method="flow_ode", solver="heun",
        sampling_eps=0.0, final_denoise=False,
        sampling_seed=23, sampling_keys=("matched-latent",),
    )
    assert torch.equal(state, canonical)
    assert report["observed_model_calls"] == 100
    assert report["sampler"].startswith("FlowMatchingTask")
    assert [entry["step"] for entry in report["observations"]] == [0, 10, 20, 30, 40, 50]
    assert [entry["flow_time"] for entry in report["observations"]] == pytest.approx(
        [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    )
    assert len(model._forward_pre_hooks) == 0


def test_trajectory_removes_hook_after_sampler_exception():
    class FailingTask:
        def sample_standardized(self, model, batch, **kwargs):
            model(batch.inputs, conditioning=None)
            raise RuntimeError("expected injected failure")

    grid = _grid(8)
    model = _ToyVelocityModel()
    batch = SimpleNamespace(num_graphs=1, inputs=torch.zeros((9**3, 5)))
    with pytest.raises(RuntimeError, match="injected failure"):
        heun_trajectory(
            FailingTask(), model, batch, steps=50, seed=23,
            key="matched-latent", grid=grid, k_bins=4,
        )
    assert len(model._forward_pre_hooks) == 0
