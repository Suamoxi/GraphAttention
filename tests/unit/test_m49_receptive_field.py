"""Analytic Jacobian and shell reductions, plus nontrivial sparse-model support."""
from itertools import product

import numpy as np
import pytest
import torch

from graph_attention.evaluation.receptive_field import (
    qualify_open_cartesian_edges, query_jacobian, shell_statistics,
)


def test_exact_jacobian_orientation_and_graph_reuse():
    x = torch.tensor([[1., 2.], [3., 4.], [5., 6.]], requires_grad=True)
    y = torch.stack((2 * x + torch.roll(x, 1, 0),
                     x + 3 * torch.roll(x, 1, 0)), dim=1).sum(dim=2)
    actual = query_jacobian(y, x, 1, retain_graph=True)
    expected = np.zeros((3, 2, 2))
    expected[1] = [[2, 2], [1, 1]]
    expected[0] = [[1, 1], [3, 3]]
    np.testing.assert_array_equal(actual, expected)
    second = query_jacobian(y, x, 2)
    np.testing.assert_array_equal(second, np.roll(expected, 1, axis=0))


def test_shell_statistics_include_self_and_cross_channels():
    j = np.zeros((4, 2, 2))
    j[0, 0, 0] = 2
    j[1, 0, 1] = 1
    j[2, 1, 0] = 2
    j[3, 1, 1] = 1
    stats = shell_statistics(j, np.array([0, 1, 1, 3]))
    np.testing.assert_array_equal(stats["counts"], [1, 2, 0, 1])
    np.testing.assert_array_equal(stats["shell"], [4, 5, 0, 1])
    np.testing.assert_allclose(stats["mean"], [4, 2.5, 0, 1])
    np.testing.assert_allclose(stats["cumulative"], [.4, .9, .9, 1])
    assert stats["radii"] == {"R50": 1, "R90": 1, "R95": 3}
    assert stats["pairs"][1, 0, 1] == 1
    assert stats["pairs"][1, 1, 0] == 4
    zero = shell_statistics(j * 0, np.array([0, 1, 1, 3]))
    assert zero["total"] == 0 and zero["cumulative"] is None
    assert zero["radii"] == {"R50": None, "R90": None, "R95": None}
    with pytest.raises(ValueError, match="invalid Jacobian"):
        shell_statistics(j * np.nan, np.array([0, 1, 1, 3]))


def test_open_cartesian_qualification_and_renumbering():
    shape = (3, 4, 5)
    ijk = np.asarray(list(product(*(range(n) for n in shape))))
    pairs = np.argwhere(np.abs(ijk[:, None] - ijk[None, :]).sum(axis=2) == 1).T
    qualify_open_cartesian_edges(ijk, shape, pairs)
    permutation = np.random.default_rng(42).permutation(len(ijk))
    qualify_open_cartesian_edges(ijk[permutation], shape, np.argsort(permutation)[pairs])
    bad = pairs.copy()
    bad[:, 0] = bad[:, 1]
    with pytest.raises(ValueError, match="repeated"):
        qualify_open_cartesian_edges(ijk, shape, bad)


def test_sparse_model_input_jacobian_has_finite_hop_support():
    from graph_attention.geometry import build_attention_edge_indices
    from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT

    n = 9
    edges = torch.tensor([(i, j) for i in range(n)
                          for j in (i - 1, i + 1) if 0 <= j < n]).T.contiguous()
    topology = build_attention_edge_indices(edges, n, {"dilated": "exact_two_hop"})
    model = AlternatingDilatedGeometricDiT(
        in_channels=2, out_channels=2, hidden_dim=8, num_heads=2, num_layers=2,
        spatial_dim=3, conditioning_channels=1, condition_embed_dim=8,
        sparse_attention_backend="scatter", use_absolute_coordinate_embedding=False,
    ).eval()
    with torch.no_grad():
        model.final_layer.projection.weight.normal_()
        for block in model.blocks:
            block.adaLN_modulation[-1].weight.normal_(std=.05)
    model.requires_grad_(False)
    x = torch.randn(n, 2, requires_grad=True)
    v = model(x, edge_index=edges, attention_edge_indices=topology,
              coords=torch.arange(n, dtype=torch.float32)[:, None].expand(n, 3).contiguous(),
              batch_index=torch.zeros(n, dtype=torch.long), conditioning=torch.tensor([[.5]]))
    j = query_jacobian(v, x, 0)
    assert np.any(j[:4] != 0)
    assert np.count_nonzero(j[4:]) == 0
    assert all(p.grad is None for p in model.parameters())
