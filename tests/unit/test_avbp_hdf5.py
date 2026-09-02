from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from graph_attention.data import AVBP_FIELD_CATALOG, AVBPHDF5Dataset, AVBPSampleSpec


def _cube_coords() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def _write_mesh(path: Path, *, one_based: bool = True, extra_node: bool = False) -> None:
    coords = _cube_coords()
    if extra_node:
        coords = np.concatenate((coords, np.asarray([[2.0, 2.0, 2.0]])), axis=0)
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("Coordinates")
        group.create_dataset("x", data=coords[:, 0])
        group.create_dataset("y", data=coords[:, 1])
        group.create_dataset("z", data=coords[:, 2])
        start = 1 if one_based else 0
        stop = 9 if one_based else 8
        connectivity = np.arange(start, stop, dtype=np.int32)
        h5f.create_group("Connectivity").create_dataset(
            "hex->node", data=connectivity.reshape(1, 8)
        )


def _write_snapshot(path: Path, *, num_nodes: int = 8) -> None:
    values = np.arange(1, num_nodes + 1, dtype=np.float64)
    with h5py.File(path, "w") as h5f:
        group = h5f.create_group("GaseousPhase")
        group.create_dataset("rho", data=values)
        group.create_dataset("rhou", data=2.0 * values)
        group.create_dataset("rhov", data=3.0 * values)
        group.create_dataset("rhow", data=4.0 * values)
        group.create_dataset("rhoE", data=5.0 * values)


def _spec(sample_id: str, snapshot: Path, mesh_id: str, mesh: Path) -> AVBPSampleSpec:
    return AVBPSampleSpec(sample_id, snapshot, mesh_id, mesh)


def test_avbp_reader_loads_requested_fields_from_separate_mesh(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    _write_mesh(mesh_path)
    _write_snapshot(snapshot_path)
    dataset = AVBPHDF5Dataset(
        samples=[_spec("sample-0", snapshot_path, "mesh-a", mesh_path)],
        field_names=["rho", "rhoE"],
    )

    sample = dataset[0]

    assert sample.sample_id == "sample-0"
    assert tuple(sample.fields) == ("rho", "rhoE")
    assert sample.fields["rho"].dtype == torch.float64
    assert sample.mesh.coords.dtype == torch.float64
    assert sample.mesh.mesh_id == "mesh-a"
    assert sample.mesh.edge_index.shape == (2, 0)
    assert sample.mesh.cell_connectivity is not None
    assert sample.mesh.cell_connectivity.tolist() == [list(range(8))]
    assert sample.metadata["mesh_file"] == str(mesh_path.resolve())
    sample.validate_against(AVBP_FIELD_CATALOG)


def test_avbp_reader_reuses_one_cached_mesh_for_multiple_snapshots(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    first_path = tmp_path / "snapshot_0.h5"
    second_path = tmp_path / "snapshot_1.h5"
    _write_mesh(mesh_path)
    _write_snapshot(first_path)
    _write_snapshot(second_path)
    dataset = AVBPHDF5Dataset(
        samples=[
            _spec("sample-0", first_path, "mesh-a", mesh_path),
            _spec("sample-1", second_path, "mesh-a", mesh_path),
        ]
    )

    first = dataset[0]
    second = dataset[1]

    assert first.mesh is second.mesh
    assert first.fields["rho"] is not second.fields["rho"]


def test_avbp_reader_supports_multiple_explicit_meshes(tmp_path: Path) -> None:
    mesh_a = tmp_path / "mesh_a.h5"
    mesh_b = tmp_path / "mesh_b.h5"
    snapshot_a = tmp_path / "snapshot_a.h5"
    snapshot_b = tmp_path / "snapshot_b.h5"
    _write_mesh(mesh_a)
    _write_mesh(mesh_b)
    _write_snapshot(snapshot_a)
    _write_snapshot(snapshot_b)
    dataset = AVBPHDF5Dataset(
        samples=[
            _spec("sample-a", snapshot_a, "mesh-a", mesh_a),
            _spec("sample-b", snapshot_b, "mesh-b", mesh_b),
        ]
    )

    first = dataset[0]
    second = dataset[1]

    assert first.mesh is not second.mesh
    assert first.mesh.mesh_id == "mesh-a"
    assert second.mesh.mesh_id == "mesh-b"


def test_avbp_reader_accepts_mapping_specs_for_hydra(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    _write_mesh(mesh_path)
    _write_snapshot(snapshot_path)
    dataset = AVBPHDF5Dataset(
        samples=[
            {
                "sample_id": "sample-0",
                "snapshot_file": str(snapshot_path),
                "mesh_id": "mesh-a",
                "mesh_file": str(mesh_path),
            }
        ]
    )

    assert dataset[0].mesh.mesh_id == "mesh-a"


def test_avbp_reader_reports_missing_requested_path(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    _write_mesh(mesh_path)
    _write_snapshot(snapshot_path)
    dataset = AVBPHDF5Dataset(
        samples=[_spec("sample-0", snapshot_path, "mesh-a", mesh_path)],
        field_names=["pressure"],
    )

    with pytest.raises(KeyError, match="Additionals/pressure"):
        dataset[0]


def test_avbp_reader_rejects_unknown_catalog_field_at_construction(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    _write_mesh(mesh_path)
    _write_snapshot(snapshot_path)

    with pytest.raises(KeyError, match="Unknown field"):
        AVBPHDF5Dataset(
            samples=[_spec("sample-0", snapshot_path, "mesh-a", mesh_path)],
            field_names=["not_a_field"],
        )


def test_avbp_reader_rejects_duplicate_sample_ids(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    first_path = tmp_path / "snapshot_0.h5"
    second_path = tmp_path / "snapshot_1.h5"
    _write_mesh(mesh_path)
    _write_snapshot(first_path)
    _write_snapshot(second_path)

    with pytest.raises(ValueError, match="sample_id values must be unique"):
        AVBPHDF5Dataset(
            samples=[
                _spec("duplicate", first_path, "mesh-a", mesh_path),
                _spec("duplicate", second_path, "mesh-a", mesh_path),
            ]
        )


def test_avbp_reader_rejects_inconsistent_mesh_identity(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    first_path = tmp_path / "snapshot_0.h5"
    second_path = tmp_path / "snapshot_1.h5"
    _write_mesh(mesh_path)
    _write_snapshot(first_path)
    _write_snapshot(second_path)

    with pytest.raises(ValueError, match="multiple mesh_id values"):
        AVBPHDF5Dataset(
            samples=[
                _spec("sample-0", first_path, "mesh-a", mesh_path),
                _spec("sample-1", second_path, "mesh-b", mesh_path),
            ]
        )


def test_auto_indexing_rejects_ambiguous_connectivity(tmp_path: Path) -> None:
    mesh_path = tmp_path / "ambiguous_mesh.h5"
    snapshot_path = tmp_path / "snapshot.h5"
    _write_mesh(mesh_path, one_based=True, extra_node=True)
    _write_snapshot(snapshot_path, num_nodes=9)
    dataset = AVBPHDF5Dataset(samples=[_spec("sample-0", snapshot_path, "mesh-a", mesh_path)])

    with pytest.raises(ValueError, match="indexing is ambiguous"):
        dataset[0]
