"""Regression test: precision branches must not mutate the shared AdamW snapshot."""
import torch

from scripts.benchmark_m49_precision import (
    cpu_copy,
    load_independent_optimizer_state,
    optimizer_steps,
)


def test_precision_branches_start_with_independent_optimizer_state():
    parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1e-4, weight_decay=0.0)
    for _ in range(3):
        parameter.grad = torch.tensor([0.2, -0.3])
        optimizer.step()
    weights = parameter.detach().clone()
    snapshot = cpu_copy(optimizer.state_dict())
    updates = []
    for _ in range(2):
        candidate = torch.nn.Parameter(weights.clone())
        opt = torch.optim.AdamW([candidate], lr=1e-4, weight_decay=0.0)
        load_independent_optimizer_state(opt, snapshot)
        assert set(optimizer_steps(opt.state_dict()).values()) == {3.0}
        candidate.grad = torch.tensor([0.4, -0.1])
        opt.step()
        updates.append(candidate.detach().clone() - weights)
        assert set(optimizer_steps(snapshot).values()) == {3.0}
    torch.testing.assert_close(updates[0], updates[1], rtol=0.0, atol=0.0)
