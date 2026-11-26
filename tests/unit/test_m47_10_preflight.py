"""Pure CPU regression gates for M47.10 preflight arithmetic and manifests."""
from dataclasses import replace
from itertools import product

import pytest
import torch

from graph_attention.data.contracts import Mesh
from graph_attention.data.m47_variants import periodic_edges_in_native_order
from graph_attention.geometry.connectivity import exact_two_hop_edge_index
from scripts.m47_10_preflight import atomic_write, inspect_one_hop, require_report


def periodic_mesh(side: int = 8):
    canonical = torch.tensor(list(product(range(side), repeat=3)), dtype=torch.float64)
    order = torch.randperm(len(canonical), generator=torch.Generator().manual_seed(42))
    coords = canonical[order] * 0.25
    n = side ** 3
    return Mesh(
        coords=coords,
        edge_index=periodic_edges_in_native_order(coords, side=side),
        mesh_id="synthetic_periodic",
        metadata={
            "periodic_box_lengths": (side * 0.25,) * 3,
            "periodic_length_nondimensional": side * 0.25 / 0.5,
            "periodic_wrap_edges": True,
        },
    )


def test_periodic_neighbor_and_two_hop_shell_on_permuted_nodes():
    mesh = periodic_mesh(8)
    result = inspect_one_hop(mesh, unique_side=8, length_ref=0.5, chunk=123)
    assert result["nodes"] == 512
    assert result["one_hop_directed_edges"] == 3072
    assert result["wrap_directed_edges"] == 384
    assert result["normalized_box_length"] == 4.0
    two = exact_two_hop_edge_index(mesh.edge_index, mesh.num_nodes)
    assert two.shape == (2, 18 * mesh.num_nodes)
    assert torch.equal(torch.bincount(two[1], minlength=mesh.num_nodes),
                       torch.full((mesh.num_nodes,), 18))


def test_neighbor_validator_catches_duplicate_or_wrong_wrap():
    mesh = periodic_mesh(8)
    bad = mesh.edge_index.clone()
    bad[:, 0] = bad[:, 1]
    with pytest.raises(AssertionError, match="repeated signed periodic neighbor"):
        inspect_one_hop(replace(mesh, edge_index=bad), unique_side=8, length_ref=0.5)
    metadata = dict(mesh.metadata, periodic_length_nondimensional=3.0)
    with pytest.raises(AssertionError, match="integer grid displacement|axial nearest neighbor|periodic wrap edges"):
        inspect_one_hop(replace(mesh, metadata=metadata), unique_side=8, length_ref=0.5)


def test_report_gate_checks_revision_and_integrity(tmp_path):
    baseline = {"git_sha": "known-revision", "qualified_manifest_sha256": "test"}
    path = tmp_path / "preflight" / "mesh.json"
    atomic_write(path, {"stage": "mesh", "status": "passed", "identity": baseline})
    assert require_report(path, baseline, "mesh")["status"] == "passed"
    with pytest.raises(RuntimeError, match="identity differs"):
        require_report(path, {**baseline, "git_sha": "changed"}, "mesh")
    with pytest.raises(RuntimeError, match="did not pass"):
        require_report(path, baseline, "gpu")
