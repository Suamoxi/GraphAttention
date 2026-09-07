"""Reader for precomputed fixed-grid 2-D slice artifacts from diffusion4avbp."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .avbp import AVBP_FIELD_CATALOG
from .case_definition import load_case_definition
from .contracts import FieldCatalog, FieldSupport, Mesh, Sample

_DEFAULT_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
_AXIS_IDS = {"x": 0, "y": 1, "z": 2}
_COORDINATE_POLICY = "shared_physical_reference_2d_mesh"


class PrecomputedSlicePTDataset(Dataset[Sample]):
    """Load diffusion4avbp fixed 2-D slice ``.pt`` samples as GraphAttention samples.

    The source artifacts store one shared canonical 2-D coordinate mesh plus
    per-slice conservative fields and extraction metadata. The slicing axis and
    slice coordinate are provenance only; they are not embedded into model
    coordinates. Graph connectivity is intentionally left empty here; the
    geometry layer constructs the canonical Cartesian 4-neighbour topology used
    by the ablation experiment.
    """

    def __init__(
        self,
        data_dir: str | Path,
        mesh_file: str | Path,
        case_file: str | Path,
        case_id: str = "HIT_LES_FORCED",
        file_pattern: str = "*.pt",
        field_names: Sequence[str] = _DEFAULT_FIELDS,
        catalog: FieldCatalog | None = None,
    ) -> None:
        self.data_dir = _directory(data_dir, "data_dir")
        self.mesh_file = _file(mesh_file, "mesh_file")
        self.case_file = _file(case_file, "case_file")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(file_pattern, str) or not file_pattern.strip():
            raise ValueError("file_pattern must be a non-empty string")

        self.files = tuple(
            sorted(path for path in self.data_dir.glob(file_pattern) if path.is_file())
        )
        if not self.files:
            raise FileNotFoundError(
                f"no slice files matched pattern '{file_pattern}' in '{self.data_dir}'"
            )
        sample_ids = tuple(path.stem for path in self.files)
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("slice sample file stems must be unique")
        self.sample_ids = sample_ids

        self.field_catalog = catalog or AVBP_FIELD_CATALOG
        self.field_names = tuple(field_names)
        if not self.field_names:
            raise ValueError("field_names must contain at least one field")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be unique")
        for spec in self.field_catalog.require(list(self.field_names)):
            if spec.support is not FieldSupport.NODE:
                raise ValueError(
                    f"slice reader supports node fields only; '{spec.name}' is {spec.support.value}"
                )
            if len(spec.components) != 1:
                raise ValueError(
                    f"slice reader currently requires scalar stored columns; '{spec.name}' "
                    f"declares {len(spec.components)} components"
                )

        self.case_definition = load_case_definition(self.case_file)
        if self.case_definition.case_id != case_id:
            raise ValueError(
                f"configured case_id '{case_id}' does not match '{self.case_definition.case_id}' "
                f"in '{self.case_file}'"
            )
        self.case_id = case_id

        self._coords_2d, self._mesh_metadata = _load_shared_mesh(self.mesh_file)
        self.grid_shape_2d = _grid_shape(self._mesh_metadata, self._coords_2d.shape[0])
        self.coordinate_policy = _coordinate_policy(self._mesh_metadata, self.mesh_file)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Sample:
        path = self.files[index]
        payload = _load_dict(path)
        x = _required_tensor(payload, "x", path)
        sample_coords = _required_tensor(payload, "coords", path)
        metadata = _required_metadata(payload, path)

        _validate_sample_tensors(
            path=path,
            x=x,
            sample_coords=sample_coords,
            shared_coords=self._coords_2d,
            grid_shape=self.grid_shape_2d,
            metadata=metadata,
            coordinate_policy=self.coordinate_policy,
        )
        _validate_extraction_metadata(metadata, path)

        channel_names = _channel_names(metadata, path, x.shape[1])
        channel_to_index = {name: channel_index for channel_index, name in enumerate(channel_names)}
        missing = [name for name in self.field_names if name not in channel_to_index]
        if missing:
            raise ValueError(f"slice '{path}' does not contain requested fields {missing}")

        fields = {name: x[:, channel_to_index[name]] for name in self.field_names}
        if self._coords_2d.dtype != x.dtype:
            raise TypeError(
                f"slice fields and shared coordinates must share one dtype, got {x.dtype} "
                f"and {self._coords_2d.dtype}"
            )

        source_stem = _required_text(metadata, "source_stem", path)
        sample_metadata = dict(metadata)
        sample_metadata.update(
            {
                "format": "precomputed_slice_pt",
                "sample_file": str(path),
                "shared_mesh_file": str(self.mesh_file),
                "case_definition_file": str(self.case_file),
                "split_group": source_stem,
                "model_coordinate_policy": self.coordinate_policy,
                "model_coordinate_dim": 2,
                "slice_orientation_is_model_input": False,
                "periodic_cross_boundary_edges": "not_augmented",
            }
        )
        mesh = Mesh(
            coords=self._coords_2d,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            mesh_id=f"{self.case_id}:fixed_slice_grid",
            metadata={
                "format": "precomputed_slice_pt",
                "grid_shape_2d": self.grid_shape_2d,
                "shared_mesh_file": str(self.mesh_file),
                "coordinate_policy": self.coordinate_policy,
                "coord_dim": 2,
                "topology": "pending_geometry_cartesian_4_neighbor",
                "periodic_cross_boundary_edges": "not_augmented",
            },
        )
        sample = Sample(
            sample_id=path.stem,
            mesh=mesh,
            fields=fields,
            reference_scales=self.case_definition.reference_scales,
            metadata=sample_metadata,
            case_id=self.case_id,
            regime_parameters=self.case_definition.regime_parameters,
        )
        sample.validate_against(self.field_catalog)
        return sample

    def group_id(self, index: int, metadata_key: str = "source_stem") -> str:
        """Return a metadata group identifier without inferring it from the filename."""

        if not isinstance(metadata_key, str) or not metadata_key.strip():
            raise ValueError("metadata_key must be a non-empty string")
        path = self.files[index]
        metadata = _required_metadata(_load_dict(path), path)
        return _required_text(metadata, metadata_key, path)


def _load_shared_mesh(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    payload = _load_dict(path)
    coords = _required_tensor(payload, "coords", path)
    metadata = _required_metadata(payload, path)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"shared slice mesh coords in '{path}' must have shape [N, 2]")
    if not coords.is_floating_point():
        raise TypeError(f"shared slice mesh coords in '{path}' must use a floating-point dtype")
    if not torch.isfinite(coords).all():
        raise ValueError(f"shared slice mesh coords in '{path}' contain NaN or Inf")
    if metadata.get("coord_dim") != 2:
        raise ValueError("shared slice mesh metadata must declare coord_dim=2")
    if metadata.get("all_samples_share_coords") is not True:
        raise ValueError("shared slice mesh metadata must declare all_samples_share_coords=true")
    _coordinate_policy(metadata, path)
    return coords, metadata


def _coordinate_policy(metadata: dict[str, object], path: Path) -> str:
    value = metadata.get("coordinate_policy")
    if value != _COORDINATE_POLICY:
        raise ValueError(
            f"slice coordinate policy in '{path}' must be '{_COORDINATE_POLICY}', got {value!r}"
        )
    return _COORDINATE_POLICY


def _grid_shape(metadata: dict[str, object], num_nodes: int) -> tuple[int, int]:
    raw = metadata.get("grid_shape_2d")
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("shared slice mesh metadata must contain grid_shape_2d with two entries")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in raw):
        raise ValueError("grid_shape_2d entries must be positive integers")
    shape = (int(raw[0]), int(raw[1]))
    if shape[0] * shape[1] != num_nodes:
        raise ValueError(
            f"grid_shape_2d {shape} implies {shape[0] * shape[1]} nodes, got {num_nodes}"
        )
    return shape


def _validate_sample_tensors(
    *,
    path: Path,
    x: torch.Tensor,
    sample_coords: torch.Tensor,
    shared_coords: torch.Tensor,
    grid_shape: tuple[int, int],
    metadata: dict[str, object],
    coordinate_policy: str,
) -> None:
    if x.ndim != 2 or x.shape[0] != shared_coords.shape[0]:
        raise ValueError(
            f"slice x in '{path}' must have shape [N, C] with N={shared_coords.shape[0]}"
        )
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError(f"slice x in '{path}' must be finite floating-point data")
    if sample_coords.shape != shared_coords.shape:
        raise ValueError(f"slice coords in '{path}' do not match the shared mesh shape")
    if sample_coords.dtype != shared_coords.dtype:
        raise TypeError(f"slice coords in '{path}' do not match the shared mesh dtype")
    if not torch.equal(sample_coords, shared_coords):
        raise ValueError(f"slice coords in '{path}' do not exactly match the shared mesh coords")
    raw_grid = metadata.get("grid_shape_2d")
    if raw_grid is not None and tuple(raw_grid) != grid_shape:
        raise ValueError(f"slice grid_shape_2d in '{path}' does not match the shared mesh")
    if metadata.get("coord_dim") != 2:
        raise ValueError(f"slice metadata in '{path}' must declare coord_dim=2")
    if metadata.get("all_samples_share_coords") is not True:
        raise ValueError(f"slice metadata in '{path}' must declare all_samples_share_coords=true")
    if metadata.get("coordinate_policy") != coordinate_policy:
        raise ValueError(f"slice coordinate_policy in '{path}' does not match the shared mesh")


def _validate_extraction_metadata(metadata: dict[str, object], path: Path) -> None:
    axis = _required_text(metadata, "axis", path)
    if axis not in _AXIS_IDS:
        raise ValueError(f"slice axis in '{path}' must be one of x, y, z")
    axis_id = metadata.get("axis_id")
    if axis_id != _AXIS_IDS[axis]:
        raise ValueError(f"slice axis and axis_id disagree in '{path}'")
    coordinate = metadata.get("slice_coordinate")
    if not isinstance(coordinate, (int, float)) or isinstance(coordinate, bool):
        raise TypeError(f"slice_coordinate in '{path}' must be a real scalar")
    if not isfinite(float(coordinate)):
        raise ValueError(f"slice_coordinate in '{path}' must be finite")


def _channel_names(metadata: dict[str, object], path: Path, num_channels: int) -> tuple[str, ...]:
    raw = metadata.get("channel_names")
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"slice metadata in '{path}' must contain channel_names")
    names = tuple(raw)
    if len(names) != num_channels:
        raise ValueError(
            f"slice channel_names in '{path}' has {len(names)} entries for {num_channels} columns"
        )
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"slice channel_names in '{path}' must contain non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"slice channel_names in '{path}' contains duplicates")
    return names


def _required_text(metadata: dict[str, object], key: str, path: Path) -> str:
    value = metadata.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"slice metadata in '{path}' requires non-empty '{key}'")
    return value


def _load_dict(path: Path) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"expected mapping payload in '{path}', got {type(payload).__name__}")
    return payload


def _required_tensor(payload: dict[str, object], key: str, path: Path) -> torch.Tensor:
    value = payload.get(key)
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"payload '{path}' requires tensor key '{key}'")
    return value


def _required_metadata(payload: dict[str, object], path: Path) -> dict[str, object]:
    value = payload.get("metadata")
    if not isinstance(value, dict):
        raise TypeError(f"payload '{path}' requires mapping metadata")
    return dict(value)


def _directory(path_value: str | Path, name: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_dir():
        raise NotADirectoryError(f"{name} is not an accessible directory: {path}")
    return path


def _file(path_value: str | Path, name: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not an accessible file: {path}")
    return path
