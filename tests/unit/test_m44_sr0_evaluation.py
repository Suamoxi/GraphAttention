"""CPU tests of scientific boundaries for point-sampled SR-0 diagnostics."""

from dataclasses import dataclass

import pytest
import torch

from graph_attention.tasks.sr0_evaluation import (
    ensemble_point_proxy_metrics,
    per_field_mse,
    point_sample_anchors,
    project_point_sample,
)


@dataclass
class _CSR:
    crow_indices: torch.Tensor
    col_indices: torch.Tensor
    weights: torch.Tensor
    num_coarse_nodes: int

    @property
    def num_fine_nodes(self):
        return self.crow_indices.numel() - 1


def _transfer():
    return _CSR(
        crow_indices=torch.tensor([0, 1, 3, 4], dtype=torch.long),
        col_indices=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        weights=torch.tensor([1., 0.5, 0.5, 1.]),
        num_coarse_nodes=2,
    )


def test_point_sample_anchor_mapping_is_bijective_and_projection_is_nondestructive():
    tr = _transfer()
    anchor, coarse = point_sample_anchors(tr)
    assert torch.equal(anchor, torch.tensor([0, 2]))
    assert torch.equal(coarse, torch.tensor([0, 1]))
    generated = torch.tensor([[9.] * 5, [7.] * 5, [8.] * 5])
    coarse_state = torch.tensor([[10.] * 5, [20.] * 5])
    restored = project_point_sample(generated, coarse_state, tr)
    torch.testing.assert_close(restored[anchor], coarse_state[coarse])
    torch.testing.assert_close(restored[1], generated[1])
    torch.testing.assert_close(generated[0], torch.tensor([9.] * 5))


def test_point_projection_rejects_noninjective_or_wrong_weights():
    tr = _transfer()
    with pytest.raises(ValueError, match="each coarse ID"):
        point_sample_anchors(_CSR(tr.crow_indices, torch.tensor([0, 0, 1, 0]), tr.weights, 2))
    with pytest.raises(ValueError, match="weight exactly one"):
        point_sample_anchors(_CSR(tr.crow_indices, tr.col_indices,
                                  torch.tensor([0.9, 0.5, 0.5, 1.0]), 2))
    with pytest.raises(ValueError, match="one node per coarse ID"):
        point_sample_anchors(_CSR(torch.tensor([0, 2, 4]),
                                  torch.tensor([0, 1, 0, 1]),
                                  torch.tensor([0.5] * 4), 2))


def test_per_field_mse_reproduces_velocity_zero_baseline():
    target = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    got = per_field_mse(torch.zeros_like(target), target)
    torch.testing.assert_close(got, target.square().mean(dim=0))


def test_ensemble_statistics_distinguish_raw_and_projected_field():
    tr = _transfer()
    baseline = torch.tensor([[10.] * 5, [15.] * 5, [20.] * 5])
    target = baseline + torch.tensor([[0.] * 5, [2.] * 5, [0.] * 5])
    coarse = torch.tensor([[10.] * 5, [20.] * 5])
    samples = torch.stack((baseline + 1., baseline - 1.), dim=0)
    diag = ensemble_point_proxy_metrics(samples, baseline, target, coarse, tr)
    assert diag["raw_point_max_abs_by_field"]["rho"] == 1.
    assert diag["projected_point_max_abs_by_field"]["rho"] == 0.
    assert diag["sample_to_sample_std_rms_by_field"]["rho"] == 1.
    assert diag["reference_residual_rms_by_field"]["rho"] > 0
    assert diag["projection_nonanchor_unchanged"] is True
    assert diag["nonpositive_density_fraction_raw"] == 0.0
