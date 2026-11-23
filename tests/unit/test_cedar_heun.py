"""Exact algebraic E1/E2 tests independent of external CEDAR data."""
import pytest
import torch

from graph_attention.tasks.cedar_heun import integrate_cedar_heun


def _project(x):
    # C=[1, 0], P=C.T, Pi=diag(0, 1).
    return torch.stack((torch.zeros_like(x[..., 0]), x[..., 1]), dim=-1)


@pytest.mark.parametrize("steps", [1, 2, 5, 20])
def test_terminal_projection_enforces_constraint_but_not_same_sample(steps):
    x0 = torch.zeros(3, 2, dtype=torch.float64)
    # Normal motion creates tangential motion via feedback: dx/dt=(1,x[0]).
    def velocity(x, _t):
        return torch.stack((torch.ones_like(x[..., 0]), x[..., 0]), dim=-1)
    e1 = integrate_cedar_heun(x0, velocity, _project, steps=steps, strategy="E1")
    e2 = integrate_cedar_heun(x0, velocity, _project, steps=steps, strategy="E2")
    assert torch.equal(e1, x0)
    torch.testing.assert_close(e2[:, 0], torch.ones(3, dtype=x0.dtype), atol=1e-14, rtol=0)
    torch.testing.assert_close(e2[:, 1], torch.full((3,), 0.5, dtype=x0.dtype), atol=1e-14, rtol=0)
    assert torch.all(_project(e2)[:, 0] == 0)
    assert not torch.allclose(_project(e1), _project(e2))


def test_tangent_velocity_gives_identical_e1_e2():
    x0 = torch.tensor([[0.0, -0.1]], dtype=torch.float64)
    def velocity(x, t):
        return torch.stack((torch.zeros_like(x[..., 0]), torch.full_like(x[..., 1], 1 + t)), -1)
    e1 = integrate_cedar_heun(x0, velocity, _project, steps=7, strategy="E1")
    e2 = integrate_cedar_heun(x0, velocity, _project, steps=7, strategy="E2")
    torch.testing.assert_close(e1, e2, atol=0, rtol=0)
    torch.testing.assert_close(e1[:, 1], torch.tensor([1.4], dtype=x0.dtype), atol=1e-14, rtol=0)


def test_heun_counts_and_observer():
    calls, projected, observed = [], [], []
    def velocity(x, t):
        calls.append(t)
        return torch.ones_like(x)
    def proj(x):
        projected.append(1)
        return _project(x)
    x0 = torch.zeros(1, 2)
    integrate_cedar_heun(x0, velocity, proj, steps=4, strategy="E1", observer=lambda i, x: observed.append(i))
    assert len(calls) == 8 and len(projected) == 8 and observed == [0, 1, 2, 3, 4]
    calls.clear(); projected.clear()
    integrate_cedar_heun(x0, velocity, proj, steps=4, strategy="E2")
    assert len(calls) == 8 and not projected


def test_invalid_inputs_fail_closed():
    x = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        integrate_cedar_heun(x, lambda x,t:x, _project, steps=0, strategy="E1")
    with pytest.raises(ValueError):
        integrate_cedar_heun(x, lambda x,t:x, _project, steps=4, strategy="X")
    with pytest.raises(ValueError):
        integrate_cedar_heun(x, lambda x,t:torch.zeros(1,3), _project, steps=1, strategy="E1")
    with pytest.raises(ValueError):
        integrate_cedar_heun(torch.tensor([float('nan')]), lambda x,t:x, lambda x:x, steps=1, strategy="E2")
