"""M44 SR-0 structured-patch geometry, independent of CFD operator assumptions."""

from dataclasses import replace

import pytest
import torch

from graph_attention.geometry.sr0_cartesian import build_sr0_cartesian_patch

CORNERS = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
           (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1))


def _native_grid(side: int = 7, *, permute: bool = False):
    axes = [torch.arange(side, dtype=torch.float32) * h + offset
            for h, offset in ((0.2, 0.1), (0.5, -3.0), (1.25, 10.0))]
    xyz = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    starts = torch.stack(torch.meshgrid(
        *(torch.arange(side - 1) for _ in range(3)), indexing="ij"
    ), -1).reshape(-1, 3)
    cells = starts[:, None, :] + torch.tensor(CORNERS)[None, :, :]
    conn = (cells[..., 0] * side + cells[..., 1]) * side + cells[..., 2]
    if permute:
        perm = torch.randperm(side**3, generator=torch.Generator().manual_seed(49))
        inv = torch.empty_like(perm)
        inv[perm] = torch.arange(perm.numel())
        xyz, conn = xyz[perm], inv[conn]
    return xyz, conn


def _transfer(patch, coarse):
    T = torch.sparse_csr_tensor(
        patch.transfer_rowptr, patch.transfer_columns, patch.transfer_weights,
        size=(patch.fine_side**3, patch.coarse_side**3),
    )
    return torch.sparse.mm(T, coarse)


def test_nonzero_patch_origin_refinement_and_permutation() -> None:
    original = build_sr0_cartesian_patch(*_native_grid(), side=7, patch_cells=4,
                                         refinement=2, origin=(1, 0, 1))
    permuted = build_sr0_cartesian_patch(*_native_grid(permute=True), side=7,
                                         patch_cells=4, refinement=2, origin=(1, 0, 1))
    assert original.fine_side == 5
    assert original.coarse_side == 3
    assert original.fine_hexes.shape == (64, 8)
    assert original.coarse_local_ids.shape == (27,)
    assert original.fine_node_ids.shape == (125,)
    torch.testing.assert_close(original.fine_coords, permuted.fine_coords)
    torch.testing.assert_close(original.coarse_coords, permuted.coarse_coords)
    torch.testing.assert_close(original.transfer_weights, permuted.transfer_weights)
    assert torch.equal(original.transfer_columns, permuted.transfer_columns)


def test_trilinear_transfer_reproduces_affine_fields_and_coarse_nodes() -> None:
    p = build_sr0_cartesian_patch(*_native_grid(), side=7, patch_cells=4, refinement=2)
    coarse_field = 1.0 + 2 * p.coarse_coords[:, :1] - p.coarse_coords[:, 1:2] + 3 * p.coarse_coords[:, 2:3]
    interpolated = _transfer(p, coarse_field)
    true = 1.0 + 2 * p.fine_coords[:, :1] - p.fine_coords[:, 1:2] + 3 * p.fine_coords[:, 2:3]
    torch.testing.assert_close(interpolated, true, atol=3e-5, rtol=1e-5)
    torch.testing.assert_close(_transfer(p, torch.ones(27, 5)), torch.ones(125, 5))
    torch.testing.assert_close(interpolated[p.coarse_local_ids], coarse_field)
    counts = p.transfer_rowptr[1:] - p.transfer_rowptr[:-1]
    assert int(counts.min()) == 1 and int(counts.max()) == 8


def test_refinement_four_preserves_anisotropic_physical_spacing() -> None:
    p = build_sr0_cartesian_patch(*_native_grid(), side=7, patch_cells=4, refinement=4)
    assert p.coarse_side == 2
    torch.testing.assert_close(_transfer(p, p.coarse_coords), p.fine_coords)


def test_rejects_invalid_refinement_and_nonlattice_geometry() -> None:
    xyz, cells = _native_grid()
    with pytest.raises(ValueError, match="divide patch_cells"):
        build_sr0_cartesian_patch(xyz, cells, side=7, patch_cells=5, refinement=2)
    with pytest.raises(ValueError, match="origin"):
        build_sr0_cartesian_patch(xyz, cells, side=7, patch_cells=4, origin=(3, 0, 0))
    xyz = xyz.clone()
    xyz[20, 0] += 0.04
    with pytest.raises(ValueError, match="Cartesian lattice"):
        build_sr0_cartesian_patch(xyz, cells, side=7, patch_cells=4)


def test_rejects_native_mesh_connectivity_mismatch() -> None:
    xyz, cells = _native_grid()
    cells = cells.clone()
    cells[0, 1] = cells[0, 0]
    with pytest.raises(ValueError, match="native"):
        build_sr0_cartesian_patch(xyz, cells, side=7, patch_cells=4)


def test_rejects_missing_grid_point_even_when_node_count_matches() -> None:
    xyz, cells = _native_grid()
    xyz = xyz.clone()
    xyz[0] = xyz[1]
    with pytest.raises(ValueError, match="bijection"):
        build_sr0_cartesian_patch(xyz, cells, side=7, patch_cells=4)
