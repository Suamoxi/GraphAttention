"""M41.4: irregular unstructured mesh CSR, two-hop and exact M37 parity."""
from __future__ import annotations

from itertools import product

import h5py
import numpy as np
import pytest
import torch

from graph_attention.geometry.connectivity import (
    hex_connectivity_to_edge_index, exact_two_hop_edge_index,
)
from graph_attention.geometry.external_csr import (
    DiskCSR, build_disk_csr, build_hdf5_disk_csr,
)
from graph_attention.geometry.partition_csr import (
    build_partition_files, spatial_owners,
)
from graph_attention.models.dinat_dit import AlternatingDilatedGeometricDiT
from graph_attention.models.disk_streamed_dinat import build_disk_streamed_dinat_plan
from graph_attention.models.streamed_dinat import forward_streamed_dinat


def _irregular_hexahedra(tmp_path, side=4):
    """Nonuniform perturbed 3D hex mesh with globally permuted node IDs."""
    rng = np.random.default_rng(409)
    n = side**3
    original_coords = np.array(list(product(range(side), repeat=3)), dtype=np.float32)
    coords = original_coords + rng.normal(scale=0.065, size=(n,3))
    permutation = rng.permutation(n)
    inverse = np.argsort(permutation)
    cells = []
    for x, y, z in product(range(side-1), repeat=3):
        corners = ((x,y,z),(x+1,y,z),(x+1,y+1,z),(x,y+1,z),
                   (x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1))
        old = np.asarray([(a*side+b)*side+c for a,b,c in corners])
        cells.append(inverse[old] + 1)
    cells = np.asarray(cells, dtype=np.uint64)
    mesh_path = tmp_path / "mesh.h5"
    with h5py.File(mesh_path, "w") as f:
        g = f.create_group("Coordinates")
        for d, label in enumerate("xyz"):
            g.create_dataset(label, data=coords[permutation, d])
        f.create_group("Connectivity").create_dataset("hex->node", data=cells.reshape(-1))
    return mesh_path, coords[permutation], cells.astype(np.int64) - 1


def _pairs(edge_index: np.ndarray) -> set[tuple[int,int]]:
    return set(map(tuple, np.asarray(edge_index).T.tolist()))


def _collect_partition_edges(directory, n, name):
    actual = set()
    for path in sorted(directory.glob("partition_*.npz")):
        with np.load(path, allow_pickle=False) as data:
            read = data["read"]
            edges = np.stack((
                read[data[name + "_source"]],
                read[data[name + "_target"]],
            ), axis=0)
            for a, b in edges.T:
                assert (int(a), int(b)) not in actual
                actual.add((int(a), int(b)))
    return actual


@pytest.mark.parametrize("include_self,include_local", [
    (False, False), (True, False), (False, True),
])
def test_irregular_hex_csr_halo_and_m37_forward(tmp_path, include_self, include_local):
    mesh, coords_np, hex_cells = _irregular_hexahedra(tmp_path)
    n = len(coords_np)
    csr_dir = tmp_path / "csr"
    stats = build_hdf5_disk_csr(
        mesh, num_nodes=n, output_dir=csr_dir,
        chunk_cells=4, bucket_nodes=11,
    )
    assert stats["status"] == "complete"
    csr = DiskCSR.open(csr_dir)
    oracle = hex_connectivity_to_edge_index(
        torch.tensor(hex_cells), n
    )
    original = _pairs(oracle.numpy())
    got = set()
    for j in range(n):
        got.update((int(s),j) for s in csr.indices[csr.indptr[j]:csr.indptr[j+1]])
    assert got == original
    dilated = exact_two_hop_edge_index(oracle, n)
    exact = _pairs(dilated.numpy())
    owners = spatial_owners(coords_np, blocks=(2,2,2), chunk_nodes=7)
    part_dir = tmp_path / "partitions"
    manifest = build_partition_files(
        csr, owners, output_dir=part_dir, chunk_targets=3,
        max_owned_nodes=100,
        include_self_attention=include_self,
        include_local_self_attention=include_local,
    )
    assert manifest["num_partitions"] >= 2
    expected_local = original | ({(i,i) for i in range(n)} if include_self or include_local else set())
    expected_dilated = exact | ({(i,i) for i in range(n)} if include_self else set())
    assert _collect_partition_edges(part_dir, n, "local") == expected_local
    assert _collect_partition_edges(part_dir, n, "dilated") == expected_dilated

    torch.manual_seed(414)
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5, hidden_dim=32, num_heads=4,
        num_layers=3, spatial_dim=3, conditioning_channels=1,
        sparse_attention_backend="torch_sparse",
        include_self_attention=include_self,
        include_local_self_attention=include_local,
    ).eval()
    # Break adaLN-zero initial identity to test attention and MLP numerically.
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.025)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.045)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.08)
    coords = torch.tensor(coords_np, dtype=torch.float32)
    data = torch.randn(n, 5)
    condition = torch.tensor([[500.0]], dtype=torch.float32)
    with torch.no_grad():
        expected = model(
            data, coords=coords, edge_index=oracle,
            attention_edge_indices={"dilated": dilated},
            conditioning=condition,
        )
        # M37 forward caches the reference CSR on model; clear them before
        # moving to the independent disk-backed plan.
        model._torch_sparse_topology_cache.clear()
        model._self_attention_edge_cache.clear()
        for preload in (False, True):
            plan = build_disk_streamed_dinat_plan(
                model, coords=coords, partition_dir=part_dir, preload=preload,
            )
            # Production uses torch.inference_mode(). Lazy CSR partitions
            # are constructed inside it; they must be ordinary versioned tensors.
            with torch.inference_mode():
                first = plan.partitions[0]
                for topology in (first.local, first.dilated):
                    assert not topology.pattern.is_inference()
                    assert not topology.pattern.crow_indices().is_inference()
                    assert not topology.pattern.col_indices().is_inference()
                result = forward_streamed_dinat(
                    model, data, conditioning=condition, plan=plan
                )
            torch.testing.assert_close(
                result, expected, atol=2e-4, rtol=2e-4
            )


def test_tetrahedral_elements_work_without_cartesian_indexing(tmp_path):
    mesh = tmp_path / "tet.h5"
    cells = np.array([[3,1,4,2], [2,3,5,4], [6,4,2,5]], dtype=np.uint64)
    with h5py.File(mesh, "w") as f:
        f.create_group("Connectivity").create_dataset("tet->node", data=cells)
    stats = build_hdf5_disk_csr(
        mesh, num_nodes=6, output_dir=tmp_path/"csr",
        element_type="tet", connectivity_path="Connectivity/tet->node",
        bucket_nodes=2, chunk_cells=2,
    )
    csr = DiskCSR.open(tmp_path/"csr")
    expected = set()
    for cell in cells.astype(np.int64)-1:
        for i in range(4):
            for j in range(i+1,4):
                expected.add((int(cell[i]),int(cell[j])))
                expected.add((int(cell[j]),int(cell[i])))
    result = set()
    for target in range(6):
        result |= {(int(s),target) for s in csr.indices[csr.indptr[target]:csr.indptr[target+1]]}
    assert result == expected
    assert stats["directed_edges"] == len(expected)
    owners = np.array([0,0,0,1,1,1], dtype=np.int32)
    build_partition_files(csr, owners, output_dir=tmp_path/"p", max_owned_nodes=3)
    direct = _collect_partition_edges(tmp_path/"p", 6, "local")
    assert direct == expected


def test_generic_directed_csr_and_true_distance_two(tmp_path):
    blocks = [
        (np.array([0, 1, 2, 3]), np.array([1, 2, 0, 4])),
        (np.array([0, 0, 4, 3]), np.array([1, 1, 3, 4])),
    ]
    build_disk_csr(blocks, num_nodes=5, output_dir=tmp_path/"csr", bucket_nodes=2)
    csr = DiskCSR.open(tmp_path/"csr")
    assert csr.directed_edges == 5
    owners = np.array([0,0,1,1,1], dtype=np.int32)
    build_partition_files(csr, owners, output_dir=tmp_path/"p", max_owned_nodes=3)
    assert (0,2) in _collect_partition_edges(tmp_path/"p",5,"dilated")
    # One-hop pairs remain excluded even if an alternative length-two
    # path happens to connect the same nodes in a directed graph.
    assert not (_collect_partition_edges(tmp_path/"p",5,"local") &
                _collect_partition_edges(tmp_path/"p",5,"dilated"))


def test_reject_wrong_indexing_and_unsized_partitions(tmp_path):
    mesh, coords, _ = _irregular_hexahedra(tmp_path)
    with pytest.raises(ValueError, match="one-based"):
        build_hdf5_disk_csr(
            mesh, num_nodes=len(coords)-1, output_dir=tmp_path/"bad",
            bucket_nodes=10, chunk_cells=2,
        )
    # Complete build is independently safe and does not rely on coordinates.
    build_hdf5_disk_csr(mesh, num_nodes=len(coords), output_dir=tmp_path/"ok")
    csr = DiskCSR.open(tmp_path/"ok")
    with pytest.raises(ValueError, match="owns"):
        build_partition_files(
            csr, np.zeros(len(coords), np.int32),
            output_dir=tmp_path/"huge", max_owned_nodes=4,
        )


def test_true_unstructured_tetrahedral_m37_full_forward_parity(tmp_path):
    """M37 numerical parity on non-Cartesian tetrahedral connectivity."""
    vertices = np.array([
        [-0.7, 0.2, 0.1], [1.5, -0.2, 0.3], [0.2, 1.4, 0.7],
        [-0.4, 0.5, 2.2], [2.1, 1.0, 1.8], [1.1, -0.9, 2.5],
        [0.8, 1.6, -1.3],
    ], dtype=np.float32)
    cells = np.array([
        [5, 1, 3, 2], [6, 4, 3, 5], [2, 3, 4, 7], [4, 5, 6, 7],
    ], dtype=np.uint64)
    mesh = tmp_path/"irregular_tet.h5"
    with h5py.File(mesh, "w") as h5:
        h5.create_group("Connectivity").create_dataset("tet->node", data=cells)
    build_hdf5_disk_csr(
        mesh, num_nodes=len(vertices), output_dir=tmp_path/"csr",
        element_type="tet", connectivity_path="Connectivity/tet->node",
        chunk_cells=2, bucket_nodes=2,
    )
    graph = DiskCSR.open(tmp_path/"csr")
    source, target = [], []
    for j in range(len(vertices)):
        neighbors = graph.indices[graph.indptr[j]:graph.indptr[j+1]]
        source.extend(int(i) for i in neighbors)
        target.extend([j]*len(neighbors))
    edge_index = torch.tensor([source, target], dtype=torch.long)
    dilated = exact_two_hop_edge_index(edge_index, len(vertices))
    owners = spatial_owners(vertices, blocks=(2,1,2), chunk_nodes=3)
    build_partition_files(
        graph, owners, output_dir=tmp_path/"parts",
        chunk_targets=2, max_owned_nodes=5,
        include_self_attention=True,
    )
    torch.manual_seed(429)
    model = AlternatingDilatedGeometricDiT(
        in_channels=5, out_channels=5, hidden_dim=32, num_heads=4,
        num_layers=4, spatial_dim=3, conditioning_channels=1,
        sparse_attention_backend="torch_sparse",
        include_self_attention=True,
    ).eval()
    with torch.no_grad():
        for block in model.blocks:
            torch.nn.init.normal_(block.adaLN_modulation[-1].weight, std=0.025)
            torch.nn.init.normal_(block.adaLN_modulation[-1].bias, std=0.045)
        torch.nn.init.normal_(model.final_layer.projection.weight, std=0.08)
        x = torch.randn(len(vertices), 5)
        cond = torch.tensor([[350.0]], dtype=torch.float32)
        xyz = torch.tensor(vertices)
        expected = model(
            x, coords=xyz, edge_index=edge_index,
            attention_edge_indices={"dilated": dilated},
            conditioning=cond,
        )
        model._torch_sparse_topology_cache.clear()
        model._self_attention_edge_cache.clear()
        plan = build_disk_streamed_dinat_plan(
            model, coords=xyz, partition_dir=tmp_path/"parts", preload=False,
        )
        with torch.inference_mode():
            observed = forward_streamed_dinat(
                model, x, plan=plan, conditioning=cond,
            )
        torch.testing.assert_close(
            observed, expected, atol=2e-4, rtol=2e-4,
        )
