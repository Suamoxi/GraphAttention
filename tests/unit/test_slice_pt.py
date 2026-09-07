from pathlib import Path

import pytest
import torch

from graph_attention.data import PrecomputedSlicePTDataset

CHANNELS = ["rho", "rhou", "rhov", "rhow", "rhoE"]
COORDINATE_POLICY = "shared_physical_reference_2d_mesh"


def _write_case(path: Path) -> None:
    path.write_text(
        """case_id: HIT_LES_FORCED
reference_scheme: hit_test
references:
  rho_ref:
    value: 1.0
    units: kg/m^3
    definition: prescribed_reference_density
    provenance: test
    inference_available: true
  U_ref:
    value: 2.0
    units: m/s
    definition: prescribed_reference_velocity
    provenance: test
    inference_available: true
  L_ref:
    value: 4.0
    units: m
    definition: prescribed_reference_length
    provenance: test
    inference_available: true
"""
    )


def _write_dataset(root: Path) -> tuple[Path, Path, Path]:
    samples = root / "samples"
    meshes = root / "meshes"
    samples.mkdir()
    meshes.mkdir()
    coords = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    mesh_file = meshes / "slice_mesh.pt"
    torch.save(
        {
            "coords": coords,
            "metadata": {
                "mesh_type": "fixed_2d_reference_slice",
                "coord_dim": 2,
                "grid_shape_2d": [2, 2],
                "coordinate_policy": COORDINATE_POLICY,
                "all_samples_share_coords": True,
            },
        },
        mesh_file,
    )
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    metadata = {
        "source_file": "/source/solut_hit_00000770.h5",
        "source_stem": "solut_hit_00000770",
        "axis": "x",
        "axis_id": 0,
        "slice_index": 8,
        "slice_coordinate": 0.25,
        "base_slice_index": 8,
        "channel_names": CHANNELS,
        "coord_dim": 2,
        "grid_shape_2d": [2, 2],
        "coordinate_policy": COORDINATE_POLICY,
        "all_samples_share_coords": True,
    }
    torch.save(
        {"x": x, "coords": coords, "metadata": metadata},
        samples / "slice_00000770_x008_b008.pt",
    )
    case_file = root / "case.yaml"
    _write_case(case_file)
    return samples, mesh_file, case_file


def test_slice_dataset_uses_shared_2d_mesh_for_model_geometry(tmp_path: Path) -> None:
    samples, mesh_file, case_file = _write_dataset(tmp_path)
    dataset = PrecomputedSlicePTDataset(
        data_dir=samples,
        mesh_file=mesh_file,
        case_file=case_file,
    )

    sample = dataset[0]

    assert sample.sample_id == "slice_00000770_x008_b008"
    assert dataset.group_id(0) == "solut_hit_00000770"
    assert dataset.grid_shape_2d == (2, 2)
    assert dataset.coordinate_policy == COORDINATE_POLICY
    torch.testing.assert_close(sample.fields["rho"], torch.tensor([0.0, 5.0, 10.0, 15.0]))
    torch.testing.assert_close(sample.fields["rhoE"], torch.tensor([4.0, 9.0, 14.0, 19.0]))
    expected_coords = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        dtype=torch.float32,
    )
    torch.testing.assert_close(sample.mesh.coords, expected_coords)
    assert sample.mesh.coords.shape == (4, 2)
    assert sample.metadata["axis"] == "x"
    assert sample.metadata["slice_coordinate"] == 0.25
    assert sample.metadata["slice_orientation_is_model_input"] is False
    assert sample.mesh.edge_index.shape == (2, 0)
    assert sample.reference_scales["L_ref"].value == 4.0


@pytest.mark.parametrize(
    ("axis", "axis_id", "slice_coordinate"),
    [("x", 0, 0.25), ("y", 1, 1.25), ("z", 2, 2.25)],
)
def test_slice_extraction_orientation_does_not_change_model_coordinates(
    tmp_path: Path,
    axis: str,
    axis_id: int,
    slice_coordinate: float,
) -> None:
    samples, mesh_file, case_file = _write_dataset(tmp_path)
    path = next(samples.glob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["axis"] = axis
    payload["metadata"]["axis_id"] = axis_id
    payload["metadata"]["slice_coordinate"] = slice_coordinate
    torch.save(payload, path)

    dataset = PrecomputedSlicePTDataset(
        data_dir=samples,
        mesh_file=mesh_file,
        case_file=case_file,
    )

    torch.testing.assert_close(dataset[0].mesh.coords, payload["coords"])
    assert dataset[0].mesh.coords.shape[1] == 2


def test_slice_dataset_rejects_sample_coords_different_from_shared_mesh(tmp_path: Path) -> None:
    samples, mesh_file, case_file = _write_dataset(tmp_path)
    path = next(samples.glob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["coords"] = payload["coords"].clone()
    payload["coords"][0, 0] = 99.0
    torch.save(payload, path)
    dataset = PrecomputedSlicePTDataset(
        data_dir=samples,
        mesh_file=mesh_file,
        case_file=case_file,
    )

    with pytest.raises(ValueError, match="do not exactly match"):
        dataset[0]


def test_slice_dataset_rejects_coordinate_policy_mismatch(tmp_path: Path) -> None:
    samples, mesh_file, case_file = _write_dataset(tmp_path)
    path = next(samples.glob("*.pt"))
    payload = torch.load(path, weights_only=True)
    payload["metadata"]["coordinate_policy"] = "different_policy"
    torch.save(payload, path)
    dataset = PrecomputedSlicePTDataset(
        data_dir=samples,
        mesh_file=mesh_file,
        case_file=case_file,
    )

    with pytest.raises(ValueError, match="coordinate_policy"):
        dataset[0]
