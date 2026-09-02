from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from graph_attention.data import AVBP_FIELD_CATALOG, AVBPHDF5Dataset


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


def _write_mesh(h5f: h5py.File, *, one_based: bool = True, extra_node: bool = False) -> None:
    coords = _cube_coords()
    if extra_node:
        coords = np.concatenate((coords, np.asarray([[2.0, 2.0, 2.0]])), axis=0)
    group = h5f.create_group("Coordinates")
    group.create_dataset("x", data=coords[:, 0])
    group.create_dataset("y", data=coords[:, 1])
    group.create_dataset("z", data=coords[:, 2])
    connectivity = np.arange(1 if one_based else 0, 9 if one_based else 8, dtype=np.int32)
    h5f.create_group("Connectivity").create_dataset("hex->node", data=connectivity.reshape(1, 8))


def _write_fields(h5f: h5py.File) -> None:
    values = np.arange(1, 9, dtype=np.float64)
    group = h5f.create_group("GaseousPhase")
    group.create_dataset("rho", data=values)
    group.create_dataset("rhou", data=2.0 * values)
    group.create_dataset("rhov", data=3.0 * values)
    group.create_dataset("rhow", data=4.0 * values)
    group.create_dataset("rhoE", data=5.0 * values)


def _write_sample(path: Path, *, include_mesh: bool = True) -> None:
    with h5py.File(path, "w") as h5f:
        if include_mesh:
            _write_mesh(h5f)
        _write_fields(h5f)


def test_avbp_reader_loads_only_requested_named_fields(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.h5"
    _write_sample(path)
    dataset = AVBPHDF5Dataset(files=[path], field_names=["rho", "rhoE"])

    sample = dataset[0]

    assert tuple(sample.fields) == ("rho", "rhoE")
    assert sample.fields["rho"].dtype == torch.float64
    assert sample.mesh.coords.dtype == torch.float64
    assert sample.mesh.edge_index.shape == (2, 0)
    assert sample.mesh.cell_connectivity is not None
    assert sample.mesh.cell_connectivity.shape == (1, 8)
    assert sample.mesh.cell_connectivity.tolist() == [list(range(8))]
    sample.validate_against(AVBP_FIELD_CATALOG)


def test_avbp_reader_supports_separate_shared_mesh_file(tmp_path: Path) -> None:
    mesh_path = tmp_path / "mesh.h5"
    with h5py.File(mesh_path, "w") as h5f:
        _write_mesh(h5f)
    snapshot_path = tmp_path / "snapshot.h5"
    _write_sample(snapshot_path, include_mesh=False)

    dataset = AVBPHDF5Dataset(files=[snapshot_path], mesh_file=mesh_path)
    sample = dataset[0]

    assert sample.mesh.mesh_id == "mesh"
    assert sample.mesh.num_nodes == 8
    assert set(sample.fields) == {"rho", "rhou", "rhov", "rhow", "rhoE"}


def test_avbp_reader_reports_missing_requested_path(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.h5"
    _write_sample(path)
    dataset = AVBPHDF5Dataset(files=[path], field_names=["pressure"])

    with pytest.raises(KeyError, match="Additionals/pressure"):
        dataset[0]


def test_avbp_reader_rejects_unknown_catalog_field_at_construction(tmp_path: Path) -> None:
    path = tmp_path / "snapshot.h5"
    _write_sample(path)

    with pytest.raises(KeyError, match="Unknown field"):
        AVBPHDF5Dataset(files=[path], field_names=["not_a_field"])


def test_avbp_data_dir_discovery_is_sorted(tmp_path: Path) -> None:
    _write_sample(tmp_path / "b.h5")
    _write_sample(tmp_path / "a.h5")

    dataset = AVBPHDF5Dataset(data_dir=tmp_path)

    assert [path.name for path in dataset.files] == ["a.h5", "b.h5"]


def test_auto_indexing_rejects_ambiguous_connectivity(tmp_path: Path) -> None:
    path = tmp_path / "ambiguous.h5"
    with h5py.File(path, "w") as h5f:
        _write_mesh(h5f, one_based=True, extra_node=True)
        _write_fields(h5f)
    dataset = AVBPHDF5Dataset(files=[path])

    with pytest.raises(ValueError, match="indexing is ambiguous"):
        dataset[0]
