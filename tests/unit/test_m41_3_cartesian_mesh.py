"""Cartesian M41.3 qualification and partition edge parity on shuffled hex meshes."""
from __future__ import annotations

from itertools import product
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.qualify_m41_3_cartesian_mesh import qualify_cartesian_hex_mesh
from graph_attention.geometry.cartesian_halo import (
    cartesian_block_bounds, partition_cartesian_edges,
)

HEX_EDGES = np.array([
    (0,1), (1,2), (2,3), (3,0), (4,5), (5,6),
    (6,7), (7,4), (0,4), (1,5), (2,6), (3,7),
])


def mesh_fixture(tmp_path, side=5, permute=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1039)
    node_count = side**3
    global_to_linear = rng.permutation(node_count) if permute else np.arange(node_count)
    linear_to_global = np.empty(node_count, dtype=np.int64)
    linear_to_global[global_to_linear] = np.arange(node_count)
    coords = np.stack((
        global_to_linear // side**2,
        (global_to_linear // side) % side,
        global_to_linear % side,
    ), axis=-1).astype(float)
    coords = coords * np.array([0.125, 0.18, 0.075]) + np.array([1.3, -3.0, 0.2])
    cells = []
    for a, b, c in product(range(side - 1), repeat=3):
        coords_cell = (
            (a,b,c), (a+1,b,c), (a+1,b+1,c), (a,b+1,c),
            (a,b,c+1), (a+1,b,c+1), (a+1,b+1,c+1), (a,b+1,c+1),
        )
        lattice_ids = np.array([(i*side+j)*side+k for i,j,k in coords_cell])
        cells.append(linear_to_global[lattice_ids] + 1)
    cells = np.asarray(cells, dtype=np.uint64)
    cells = cells[rng.permutation(len(cells))]
    path = tmp_path / "mesh.mesh.h5"
    with h5py.File(path, "w") as f:
        co = f.create_group("Coordinates")
        for axis, name in enumerate("xyz"):
            co.create_dataset(name, data=coords[:, axis])
        f.create_group("Connectivity").create_dataset("hex->node", data=cells.reshape(-1))
    return path, cells, global_to_linear, linear_to_global


def test_all_cells_and_partition_edges_match_reference(tmp_path):
    side = 5
    mesh, cells, forward_expected, inverse_expected = mesh_fixture(tmp_path, side)
    result = qualify_cartesian_hex_mesh(
        mesh, tmp_path / "maps", side=side, chunk_nodes=17, chunk_cells=7
    )
    assert result["num_cells_verified"] == (side - 1)**3
    assert result["status"] == "qualified_complete_nonperiodic_cartesian_hex_graph"
    fwd = np.load(result["global_to_cartesian"], mmap_mode="r")
    inv = np.load(result["cartesian_to_global"], mmap_mode="r")
    np.testing.assert_array_equal(fwd, forward_expected)
    np.testing.assert_array_equal(inv, inverse_expected)
    N = side**3
    adj = np.zeros((N, N), dtype=bool)
    for cell in cells.astype(np.int64) - 1:
        for a, b in HEX_EDGES:
            adj[cell[a], cell[b]] = True
            adj[cell[b], cell[a]] = True
    assert adj.sum() == 6 * side * side * (side - 1)
    twohop = (adj.astype(np.int32) @ adj.astype(np.int32)) > 0
    twohop &= ~adj
    np.fill_diagonal(twohop, False)
    one_hop_part = np.zeros((N, N), dtype=bool)
    two_hop_part = np.zeros((N, N), dtype=bool)
    seen = np.zeros(N, dtype=bool)
    for _, lo, hi in cartesian_block_bounds(side, (2, 2, 2)):
        plan = partition_cartesian_edges(inv, side=side, lower=lo, upper=hi)
        assert not np.any(seen[plan.owned_global])
        seen[plan.owned_global] = True
        assert np.all(np.isin(plan.owned_global, plan.read_global))
        assert np.all(np.isin(plan.local_edge_index[0], plan.read_global))
        assert np.all(np.isin(plan.dilated_edge_index[0], plan.read_global))
        one_hop_part[plan.local_edge_index[0], plan.local_edge_index[1]] = True
        two_hop_part[plan.dilated_edge_index[0], plan.dilated_edge_index[1]] = True
    assert np.all(seen)
    np.testing.assert_array_equal(one_hop_part, adj)
    np.testing.assert_array_equal(two_hop_part, twohop)
    assert twohop.sum() == (
        6*side*side*(side-2) + 12*side*(side-1)**2
    )


def test_non_cartesian_hex_local_order_is_rejected(tmp_path):
    path, cells, _, _ = mesh_fixture(tmp_path)
    with h5py.File(path, "r+") as f:
        d = f["Connectivity/hex->node"]
        block = d[:8]
        block[[1, 2]] = block[[2, 1]]
        d[:8] = block
    with pytest.raises(ValueError, match="canonical local hex edge"):
        qualify_cartesian_hex_mesh(path, tmp_path / "broken", side=5)
    assert not (tmp_path / "broken" / "cartesian_qualification.json").exists()


def test_duplicate_cells_are_rejected(tmp_path):
    path, cells, _, _ = mesh_fixture(tmp_path)
    with h5py.File(path, "r+") as f:
        d = f["Connectivity/hex->node"]
        d[-8:] = d[:8]
    with pytest.raises(ValueError, match="duplicate physical hexahedral"):
        qualify_cartesian_hex_mesh(path, tmp_path / "broken", side=5, chunk_cells=10)


def test_nonuniform_axis_coordinate_rejected(tmp_path):
    path, _, _, _ = mesh_fixture(tmp_path)
    with h5py.File(path, "r+") as f:
        d = f["Coordinates/x"]
        values = d[:]
        values[0] += 0.03
        d[:] = values
    with pytest.raises(ValueError, match="coordinate lattice mismatch"):
        qualify_cartesian_hex_mesh(path, tmp_path / "broken", side=5)


def test_one_based_connectivity_required(tmp_path):
    path, _, _, _ = mesh_fixture(tmp_path)
    with h5py.File(path, "r+") as f:
        d = f["Connectivity/hex->node"]
        d[0] = 0
    with pytest.raises(ValueError, match="one-based"):
        qualify_cartesian_hex_mesh(path, tmp_path / "broken", side=5)


def test_non_cartesian_cell_rejected(tmp_path):
    path, _, _, _ = mesh_fixture(tmp_path)
    with h5py.File(path, "r+") as f:
        d = f["Connectivity/hex->node"]
        d[0] = np.uint64(5) if d[0] != 5 else np.uint64(6)
    with pytest.raises(ValueError, match="non-unit hexahedral|eight lattice corners"):
        qualify_cartesian_hex_mesh(path, tmp_path / "broken", side=5)
