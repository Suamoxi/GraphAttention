"""Independent AVBP order, periodic connectivity and value regression tests.

No CEDAR install or real CFD archives are needed for these synthetic tests.
"""
from __future__ import annotations

from itertools import product

import h5py
import numpy as np
import pytest

from scripts.m48_4d_ordering_audit import (
    _hex_bases_from_periodic_corners,
    compare_filtered_levels,
    compare_native_component,
    independent_cartesian_map,
    verify_lattice_mapping,
    verify_periodic_hex_cells,
)


def _rearranged_nodal_mesh(n: int, seed: int = 71):
    shape = (n + 1,) * 3
    grid = np.indices(shape).reshape(3, -1).T
    rng = np.random.default_rng(seed)
    # The raw AVBP node enumeration has no relationship to C-order grid IDs.
    permutation = rng.permutation(len(grid))
    original_to_shuffled = np.empty(len(grid), dtype=np.int64)
    original_to_shuffled[permutation] = np.arange(len(grid))
    coords = grid[permutation].astype(np.float64)
    original_ids = np.arange(len(grid)).reshape(shape)
    corners = list(product((0, 1), repeat=3))
    cells = np.array([
        [original_ids[i + di, j + dj, k + dk] for di, dj, dk in corners]
        for i, j, k in product(range(n), repeat=3)
    ], dtype=np.int64)
    cells = original_to_shuffled[cells] + 1
    cells = cells[rng.permutation(len(cells))]
    indices = np.mod(coords.astype(np.int64), n)
    linear = (indices[:, 0] * n + indices[:, 1]) * n + indices[:, 2]
    unique, first = np.unique(linear, return_index=True)
    assert np.array_equal(unique, np.arange(n**3))
    reps = first
    return coords, cells, reps, linear


@pytest.mark.parametrize("native_side", [4, 8, 10])
def test_avbp_arbitrary_node_permutation_and_complete_periodic_seams(
    tmp_path, native_side,
):
    n = native_side
    coords, cells, reps, source_to_unique = _rearranged_nodal_mesh(n)
    periodic_ijk, independent_ids, report = independent_cartesian_map(
        coords, native_side=n, native_spacing=np.ones(3),
    )
    assert report["periodic_endpoint_alias_nodes"] == (n + 1)**3 - n**3
    assert np.array_equal(independent_ids, source_to_unique)
    mapping = verify_lattice_mapping(periodic_ijk, independent_ids, reps,
                                     source_to_unique, n)
    assert mapping["source_to_unique_mismatches"] == 0
    file = tmp_path / "mesh.h5"
    with h5py.File(file, "w") as h:
        h["Connectivity/hex->node"] = cells.reshape(-1)
    with h5py.File(file) as h:
        topo = verify_periodic_hex_cells(
            h["Connectivity/hex->node"], periodic_ijk, native_side=n,
            chunk_cells=7,
        )
    assert topo["all_native_hex_cells_verified"] == n**3
    assert topo["periodic_face_seam_cells_xyz"] == [n*n] * 3
    assert topo["periodic_pair_seam_cells_xy_xz_yz"] == [n] * 3
    assert topo["periodic_corner_seam_cells"] == 1


@pytest.mark.parametrize("problem", ["wrong_representatives", "wrong_source_map", "swapped_xyz"])
def test_independent_mapping_detects_wrong_node_order(problem):
    n = 8
    coords, cells, reps, source_to_unique = _rearranged_nodal_mesh(n)
    if problem == "swapped_xyz":
        independent_ijk, independent_ids, _ = independent_cartesian_map(
            coords[:, [2, 1, 0]], native_side=n,
            native_spacing=np.ones(3),
        )
    else:
        independent_ijk, independent_ids, _ = independent_cartesian_map(
            coords, native_side=n, native_spacing=np.ones(3),
        )
    if problem == "wrong_representatives":
        reps = reps.copy()
        reps[0], reps[1] = reps[1], reps[0]
    if problem == "wrong_source_map":
        source_to_unique = source_to_unique.copy()
        source_to_unique[1] = (source_to_unique[1] + 1) % n**3
    with pytest.raises(RuntimeError, match="source_to_unique|representatives"):
        verify_lattice_mapping(independent_ijk, independent_ids,
                               reps, source_to_unique, n)


def test_rejects_non_cartesian_physical_coordinates():
    n = 8
    coords, *_ = _rearranged_nodal_mesh(n)
    coords[0, 2] += 0.08
    with pytest.raises(RuntimeError, match="Cartesian grid"):
        independent_cartesian_map(
            coords, native_side=n, native_spacing=np.ones(3),
        )


def test_rejects_corrupt_hex_topology_and_duplicate_periodic_cells(tmp_path):
    n = 8
    coords, cells, *_ = _rearranged_nodal_mesh(n)
    ijk, _, _ = independent_cartesian_map(
        coords, native_side=n, native_spacing=np.ones(3),
    )
    bad = cells.copy()
    bad[0, 0] = bad[0, 1]
    with h5py.File(tmp_path / "bad.h5", "w") as h:
        h["cells"] = bad.reshape(-1)
    with h5py.File(tmp_path / "bad.h5", "r") as h:
        with pytest.raises(RuntimeError, match="eight Cartesian corners"):
            verify_periodic_hex_cells(h["cells"], ijk, native_side=n, chunk_cells=11)
    bad = cells.copy()
    bad[0] = bad[1]
    with h5py.File(tmp_path / "duplicate.h5", "w") as h:
        h["cells"] = bad.reshape(-1)
    with h5py.File(tmp_path / "duplicate.h5", "r") as h:
        with pytest.raises(RuntimeError, match="duplicate periodic Cartesian cell"):
            verify_periodic_hex_cells(h["cells"], ijk, native_side=n, chunk_cells=11)


def test_native_qf_full_field_and_periodic_aliases_detect_transpose():
    n = 8
    coords, _, reps, source_to_unique = _rearranged_nodal_mesh(n)
    raw = (source_to_unique.astype(np.float64) + 0.125)
    expected = raw[reps].reshape((n, n, n))
    out = compare_native_component(raw, expected, reps, source_to_unique,
                                   native_side=n)
    assert out["exact_native_value_mismatches"] == 0
    assert out["periodic_alias_values_checked"] == (n + 1)**3 - n**3
    assert out["periodic_alias_relative_l2"] == 0
    with pytest.raises(RuntimeError, match="published MS_SR qf values differ"):
        compare_native_component(raw, expected.swapaxes(0, 2), reps,
                                 source_to_unique, native_side=n)
    bad = raw.copy()
    nonrep = np.flatnonzero(~np.isin(np.arange(len(raw)), reps))
    bad[nonrep[0]] += 2
    with pytest.raises(RuntimeError, match="periodic alias field mismatch"):
        compare_native_component(bad, expected, reps, source_to_unique,
                                 native_side=n)


class _SyntheticBlockAverage:
    def repeat(self, count):
        assert count == 3
        return self

    def restrict_levels(self, fine, shape):
        assert fine.shape == shape
        levels = [fine]
        for _ in range(3):
            previous = levels[-1]
            n = previous.shape[0] // 2
            levels.append(previous.reshape(n, 2, n, 2, n, 2).mean(axis=(1, 3, 5)))
        return levels


def test_full_multilevel_comparison_rejects_q4_axis_swap(tmp_path):
    n = 8
    qf = np.arange(n**3, dtype=np.float64).reshape(n, n, n)
    operator = _SyntheticBlockAverage()
    _, q2, q4, q8 = operator.restrict_levels(qf, qf.shape)
    file = tmp_path / "levels.h5"
    with h5py.File(file, "w") as h:
        for k, values in ((2, q2), (4, q4), (8, q8)):
            h[f"filters/w1p25_l1/q{k}"] = values[..., None]
    with h5py.File(file) as h:
        checks = compare_filtered_levels(qf, h, component=0, operator=operator)
        assert all(value == 0 for value in checks.values())
    with h5py.File(file, "r+") as h:
        h["filters/w1p25_l1/q4"][...] = q4.swapaxes(0, 2)[..., None]
    with h5py.File(file) as h:
        with pytest.raises(RuntimeError, match="q4 disagrees"):
            compare_filtered_levels(qf, h, component=0, operator=operator)
