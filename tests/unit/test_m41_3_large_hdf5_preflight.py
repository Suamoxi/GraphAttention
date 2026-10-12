"""CPU-only tests for bounded-RAM HDF5 preflight (never materialize graphs)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from scripts.preflight_m41_3_large_hdf5 import inspect


def _fixture(
    tmp_path,
    *,
    index_base=0,
    shape_mode="cells_first",
    bad_field=False,
    ambiguous=False,
):
    n = 3
    tmp_path.mkdir(parents=True, exist_ok=True)
    mesh_path = tmp_path / "mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    points = np.array(
        [(i, j, k) for i in range(n) for j in range(n) for k in range(n)],
        dtype=np.float64,
    )
    # Eight hexahedral cells on a regular 3^3 node lattice.
    cells = []
    for i in range(n - 1):
        for j in range(n - 1):
            for k in range(n - 1):
                cells.append([
                    (ii * n + jj) * n + kk
                    for ii, jj, kk in (
                        (i,j,k), (i+1,j,k), (i+1,j+1,k), (i,j+1,k),
                        (i,j,k+1), (i+1,j,k+1), (i+1,j+1,k+1), (i,j+1,k+1),
                    )
                ])
    conn = np.asarray(cells, dtype=np.int64) + index_base
    if ambiguous:
        conn[:] = np.arange(1, len(conn) * 8 + 1).reshape(-1, 8) % 26 + 1
    if shape_mode == "cells_second":
        conn = conn.T
    if shape_mode == "flat":
        conn = conn.ravel()

    with h5py.File(mesh_path, "w") as mesh:
        group = mesh.create_group("Coordinates")
        for index, name in enumerate(("x", "y", "z")):
            group.create_dataset(name, data=points[:, index])
        mesh.create_group("Connectivity").create_dataset(
            "hex->node", data=conn, compression="gzip"
        )

    with h5py.File(snapshot_path, "w") as sol:
        group = sol.create_group("GaseousPhase")
        for name in ("rho", "rhou", "rhov", "rhow", "rhoE"):
            length = len(points) + int(bad_field and name == "rhoE")
            group.create_dataset(name, data=np.linspace(1, 2, length))
    return mesh_path, snapshot_path


@pytest.mark.parametrize("mode", ["cells_first", "cells_second", "flat"])
@pytest.mark.parametrize("index_base", [0, 1])
def test_preflight_supports_hexahedra_and_explicit_global_indexing(
    tmp_path, mode, index_base
):
    mesh, sol = _fixture(tmp_path, index_base=index_base, shape_mode=mode)
    report = inspect(
        mesh, sol, expected_nodes=27,
        scan_connectivity_extrema=True, scan_elements=16,
    )
    assert report["num_nodes"] == 27
    assert report["num_hex_cells"] == 8
    assert report["node_count_is_perfect_cube"]
    assert report["expected_regular_cube_hex_cells"] == 8
    assert report["theoretical_6neighbor_directed_edges"] == 108
    assert report["theoretical_exact2hop_directed_edges"] == 198
    assert report["connectivity_indexing"] == (
        "zero_based" if index_base == 0 else "one_based"
    )
    assert report["host_hidden_two_buffer_gib"] > 0
    assert len(report["field_datasets"]) == 5
    assert report["status"] == "metadata_compatible_only_not_graph_or_inference_validated"


def test_metadata_only_never_guesses_connectivity_indexing(tmp_path):
    mesh, sol = _fixture(tmp_path, index_base=1)
    report = inspect(mesh, sol, expected_nodes=27)
    assert report["connectivity_indexing"].startswith("not scanned")
    assert report["connectivity_global_extrema"] is None


def test_preflight_rejects_wrong_mesh_size_and_field_count(tmp_path):
    mesh, sol = _fixture(tmp_path)
    with pytest.raises(ValueError, match="expected 28"):
        inspect(mesh, sol, expected_nodes=28)

    mesh_bad, sol_bad = _fixture(
        tmp_path / "bad", bad_field=True
    )
    with pytest.raises(ValueError, match="rhoE has 28"):
        inspect(mesh_bad, sol_bad, expected_nodes=27)


def test_connectivity_ambiguous_indexing_is_not_claimed_known(tmp_path):
    mesh, sol = _fixture(tmp_path, ambiguous=True)
    report = inspect(
        mesh, sol, expected_nodes=27, scan_connectivity_extrema=True
    )
    assert report["connectivity_indexing"].startswith("ambiguous:")


def test_missing_conservative_variable_is_rejected(tmp_path):
    mesh, sol = _fixture(tmp_path)
    with h5py.File(sol, "a") as h5:
        del h5["GaseousPhase/rhoE"]
    with pytest.raises(KeyError, match="rhoE"):
        inspect(mesh, sol, expected_nodes=27)
