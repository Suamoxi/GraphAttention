"""AVBP HDF5 reader backed by the explicit M2 field and mesh contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from operator import index as operator_index
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .contracts import FieldCatalog, FieldRole, FieldSpec, FieldSupport, Mesh, Sample

_DEFAULT_FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
_DEFAULT_COORD_PATHS = ("Coordinates/x", "Coordinates/y", "Coordinates/z")
_DEFAULT_CONNECTIVITY_PATH = "Connectivity/hex->node"
_VALID_INDEXING = {"auto", "zero", "one"}
_SAMPLE_SPEC_KEYS = {"sample_id", "snapshot_file", "mesh_id", "mesh_file"}


AVBP_FIELD_CATALOG = FieldCatalog(
    (
        FieldSpec(
            name="rho",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            source_path="GaseousPhase/rho",
            provenance="AVBP stored conservative density",
        ),
        FieldSpec(
            name="rhou",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            components=("x",),
            source_path="GaseousPhase/rhou",
            provenance="AVBP stored x-component of momentum density",
        ),
        FieldSpec(
            name="rhov",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            components=("y",),
            source_path="GaseousPhase/rhov",
            provenance="AVBP stored y-component of momentum density",
        ),
        FieldSpec(
            name="rhow",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            components=("z",),
            source_path="GaseousPhase/rhow",
            provenance="AVBP stored z-component of momentum density",
        ),
        FieldSpec(
            name="rhoE",
            support=FieldSupport.NODE,
            role=FieldRole.PRIMARY_STATE,
            source_path="GaseousPhase/rhoE",
            provenance="AVBP stored total-energy density",
        ),
        FieldSpec(
            name="pressure",
            support=FieldSupport.NODE,
            role=FieldRole.AUXILIARY_PHYSICAL,
            source_path="Additionals/pressure",
            provenance="AVBP stored auxiliary pressure field",
        ),
        FieldSpec(
            name="temperature",
            support=FieldSupport.NODE,
            role=FieldRole.AUXILIARY_PHYSICAL,
            source_path="Additionals/temperature",
            provenance="AVBP stored auxiliary temperature field",
        ),
        FieldSpec(
            name="vis_lam",
            support=FieldSupport.NODE,
            role=FieldRole.AUXILIARY_PHYSICAL,
            source_path="Additionals/vis_lam",
            provenance="AVBP stored laminar-viscosity field",
        ),
        FieldSpec(
            name="vis_turb",
            support=FieldSupport.NODE,
            role=FieldRole.AUXILIARY_PHYSICAL,
            source_path="Additionals/vis_turb",
            provenance="AVBP stored turbulent-viscosity field",
        ),
        FieldSpec(
            name="mpi_rank",
            support=FieldSupport.NODE,
            role=FieldRole.SOLVER_METADATA,
            source_path="Additionals/mpi_rank",
            provenance="AVBP computational partition metadata",
        ),
    )
)


@dataclass(frozen=True, slots=True)
class AVBPSampleSpec:
    """Explicit association between one AVBP snapshot and its separate mesh file."""

    sample_id: str
    snapshot_file: str | Path
    mesh_id: str
    mesh_file: str | Path

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("AVBPSampleSpec.sample_id must be a non-empty string")
        if not isinstance(self.mesh_id, str) or not self.mesh_id.strip():
            raise ValueError("AVBPSampleSpec.mesh_id must be a non-empty string")


class AVBPHDF5Dataset(Dataset[Sample]):
    """Read explicitly paired AVBP snapshots and separate native mesh files.

    AVBP solution snapshots and meshes are separate files in this project. Each
    sample therefore declares its snapshot-to-mesh association explicitly.
    Meshes are decoded lazily and cached once per unique mesh file in each
    dataset process. Cell-to-node connectivity is stored on
    ``Mesh.cell_connectivity``; graph-edge construction remains a geometry-layer
    operation.
    """

    def __init__(
        self,
        samples: Sequence[AVBPSampleSpec | Mapping[str, object]] | None = None,
        field_names: Sequence[str] = _DEFAULT_FIELDS,
        coord_paths: Sequence[str] = _DEFAULT_COORD_PATHS,
        connectivity_path: str = _DEFAULT_CONNECTIVITY_PATH,
        connectivity_indexing: str = "auto",
        catalog: FieldCatalog | None = None,
    ) -> None:
        self.sample_specs = _normalize_sample_specs(samples)
        self.field_catalog = catalog or AVBP_FIELD_CATALOG
        self.field_names = tuple(field_names)
        if not self.field_names:
            raise ValueError("field_names must contain at least one field")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be unique")
        self.field_specs = self.field_catalog.require(list(self.field_names))
        for spec in self.field_specs:
            if not spec.stored or spec.source_path is None:
                raise ValueError(
                    f"AVBP reader requires a stored source_path for field '{spec.name}'"
                )
            if spec.support is not FieldSupport.NODE:
                raise ValueError(
                    f"M3.2 AVBP reader currently supports node fields only; "
                    f"'{spec.name}' is {spec.support.value}-supported"
                )

        self.coord_paths = tuple(coord_paths)
        if not self.coord_paths or any(not path for path in self.coord_paths):
            raise ValueError("coord_paths must contain non-empty HDF5 paths")
        if not connectivity_path:
            raise ValueError("connectivity_path must be non-empty")
        if connectivity_indexing not in _VALID_INDEXING:
            raise ValueError("connectivity_indexing must be one of: auto, zero, one")
        self.connectivity_path = connectivity_path
        self.connectivity_indexing = connectivity_indexing
        self._mesh_cache: dict[Path, Mesh] = {}

    def __len__(self) -> int:
        return len(self.sample_specs)

    def __getitem__(self, index: int) -> Sample:
        sample_index = operator_index(index)
        spec = self.sample_specs[sample_index]
        mesh = self._mesh_for(spec)
        fields = self._read_fields(spec.snapshot_file)
        sample = Sample(
            sample_id=spec.sample_id,
            mesh=mesh,
            fields=fields,
            metadata={
                "format": "avbp_hdf5",
                "snapshot_file": str(spec.snapshot_file),
                "mesh_id": spec.mesh_id,
                "mesh_file": str(spec.mesh_file),
            },
        )
        sample.validate_against(self.field_catalog)
        return sample

    def _mesh_for(self, spec: AVBPSampleSpec) -> Mesh:
        mesh_path = _path_value(spec.mesh_file)
        mesh = self._mesh_cache.get(mesh_path)
        if mesh is None:
            mesh = self._read_mesh(mesh_path, mesh_id=spec.mesh_id)
            self._mesh_cache[mesh_path] = mesh
        return mesh

    def _read_fields(self, path_value: str | Path) -> dict[str, torch.Tensor]:
        path = _path_value(path_value)
        fields: dict[str, torch.Tensor] = {}
        with h5py.File(path, "r") as h5f:
            for spec in self.field_specs:
                tensor = _read_tensor(h5f, spec.source_path, path)
                fields[spec.name] = _canonicalize_node_field(tensor, spec, path)
        return fields

    def _read_mesh(self, path: Path, mesh_id: str) -> Mesh:
        with h5py.File(path, "r") as h5f:
            coords = _read_coordinates(h5f, self.coord_paths, path)
            raw_connectivity = _read_tensor(h5f, self.connectivity_path, path)
        cell_connectivity = _normalize_hex_connectivity(
            raw_connectivity,
            num_nodes=coords.shape[0],
            indexing=self.connectivity_indexing,
        )
        return Mesh(
            coords=coords,
            edge_index=torch.empty((2, 0), dtype=torch.long),
            mesh_id=mesh_id,
            cell_connectivity=cell_connectivity,
            metadata={
                "format": "avbp_hdf5",
                "mesh_file": str(path),
                "cell_type": "hex",
                "coord_paths": self.coord_paths,
                "connectivity_path": self.connectivity_path,
            },
        )


def _normalize_sample_specs(
    samples: Sequence[AVBPSampleSpec | Mapping[str, object]] | None,
) -> tuple[AVBPSampleSpec, ...]:
    if samples is None or len(samples) == 0:
        raise ValueError("samples must contain at least one explicit snapshot/mesh association")
    if isinstance(samples, (str, Path)):
        raise TypeError("samples must be a sequence of AVBP sample specifications")

    normalized: list[AVBPSampleSpec] = []
    for position, raw in enumerate(samples):
        if isinstance(raw, AVBPSampleSpec):
            sample_id = raw.sample_id
            snapshot_value = raw.snapshot_file
            mesh_id = raw.mesh_id
            mesh_value = raw.mesh_file
        elif isinstance(raw, Mapping):
            keys = set(raw)
            missing = _SAMPLE_SPEC_KEYS - keys
            extra = keys - _SAMPLE_SPEC_KEYS
            if missing:
                raise ValueError(
                    f"AVBP sample specification {position} is missing keys: {sorted(missing)}"
                )
            if extra:
                raise ValueError(
                    f"AVBP sample specification {position} has unsupported keys: {sorted(extra)}"
                )
            sample_id = _text_value(raw["sample_id"], "sample_id", position)
            snapshot_value = raw["snapshot_file"]
            mesh_id = _text_value(raw["mesh_id"], "mesh_id", position)
            mesh_value = raw["mesh_file"]
        else:
            raise TypeError("each AVBP sample specification must be an AVBPSampleSpec or mapping")

        snapshot_path = _existing_file(snapshot_value, "snapshot_file", position)
        mesh_path = _existing_file(mesh_value, "mesh_file", position)
        normalized.append(
            AVBPSampleSpec(
                sample_id=sample_id,
                snapshot_file=snapshot_path,
                mesh_id=mesh_id,
                mesh_file=mesh_path,
            )
        )

    sample_ids = [spec.sample_id for spec in normalized]
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("AVBP sample_id values must be unique")

    mesh_id_to_path: dict[str, Path] = {}
    mesh_path_to_id: dict[Path, str] = {}
    for spec in normalized:
        mesh_path = _path_value(spec.mesh_file)
        known_path = mesh_id_to_path.setdefault(spec.mesh_id, mesh_path)
        if known_path != mesh_path:
            raise ValueError(f"mesh_id '{spec.mesh_id}' is associated with multiple mesh files")
        known_id = mesh_path_to_id.setdefault(mesh_path, spec.mesh_id)
        if known_id != spec.mesh_id:
            raise ValueError(f"mesh file '{mesh_path}' is associated with multiple mesh_id values")
    return tuple(normalized)


def _text_value(value: object, name: str, position: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"AVBP sample specification {position} has invalid {name}")
    return value


def _existing_file(value: object, name: str, position: int) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"AVBP sample specification {position} has non-path {name}")
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"AVBP sample specification {position} {name} does not exist: {path}"
        )
    return path


def _path_value(value: str | Path) -> Path:
    return value if isinstance(value, Path) else Path(value)


def _read_tensor(h5f: h5py.File, dataset_path: str, file_path: Path) -> torch.Tensor:
    if dataset_path not in h5f:
        raise KeyError(f"HDF5 path '{dataset_path}' not found in '{file_path}'")
    dataset = h5f[dataset_path]
    if not isinstance(dataset, h5py.Dataset):
        raise TypeError(f"HDF5 path '{dataset_path}' in '{file_path}' is not a dataset")
    array = np.asarray(dataset)
    try:
        return torch.as_tensor(array)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"HDF5 dataset '{dataset_path}' in '{file_path}' cannot be represented as a tensor"
        ) from exc


def _read_coordinates(
    h5f: h5py.File,
    coord_paths: tuple[str, ...],
    file_path: Path,
) -> torch.Tensor:
    axes: list[torch.Tensor] = []
    for path in coord_paths:
        axis = _read_tensor(h5f, path, file_path)
        if not axis.is_floating_point():
            raise TypeError(f"coordinate dataset '{path}' in '{file_path}' must be floating-point")
        if axis.ndim == 2 and axis.shape[1] == 1:
            axis = axis[:, 0]
        if axis.ndim != 1:
            raise ValueError(
                f"coordinate dataset '{path}' in '{file_path}' must have shape [N] or [N, 1]"
            )
        if not torch.isfinite(axis).all():
            raise ValueError(f"coordinate dataset '{path}' in '{file_path}' contains NaN or Inf")
        axes.append(axis)

    node_counts = {axis.shape[0] for axis in axes}
    if len(node_counts) != 1:
        raise ValueError(f"coordinate datasets in '{file_path}' have inconsistent node counts")
    return torch.stack(axes, dim=1)


def _canonicalize_node_field(
    tensor: torch.Tensor,
    spec: FieldSpec,
    file_path: Path,
) -> torch.Tensor:
    if tensor.is_complex():
        raise TypeError(f"field '{spec.name}' in '{file_path}' must be real-valued")
    if tensor.is_floating_point() and not torch.isfinite(tensor).all():
        raise ValueError(f"field '{spec.name}' in '{file_path}' contains NaN or Inf")

    component_count = len(spec.components)
    if component_count == 1:
        if tensor.ndim == 2 and tensor.shape[1] == 1:
            tensor = tensor[:, 0]
        if tensor.ndim != 1:
            raise ValueError(
                f"scalar node field '{spec.name}' in '{file_path}' must have shape [N] or [N, 1]"
            )
    elif tensor.ndim != 2 or tensor.shape[1] != component_count:
        raise ValueError(
            f"field '{spec.name}' in '{file_path}' declares {component_count} components "
            f"but has shape {tuple(tensor.shape)}"
        )
    return tensor


def _normalize_hex_connectivity(
    connectivity: torch.Tensor,
    num_nodes: int,
    indexing: str,
) -> torch.Tensor:
    if (
        connectivity.dtype == torch.bool
        or connectivity.is_floating_point()
        or connectivity.is_complex()
    ):
        raise TypeError("AVBP hex connectivity must use an integer dtype")
    conn = connectivity.to(dtype=torch.long)

    if conn.ndim == 1:
        if conn.numel() % 8 != 0:
            raise ValueError("flattened AVBP hex connectivity must contain a multiple of 8 entries")
        conn = conn.reshape(-1, 8)
    elif conn.ndim == 2:
        if conn.shape[1] == 8:
            pass
        elif conn.shape[0] == 8:
            conn = conn.transpose(0, 1).contiguous()
        else:
            raise ValueError("AVBP hex connectivity must have shape [C, 8] or [8, C]")
    else:
        raise ValueError("AVBP hex connectivity must be a 1D or 2D tensor")

    if conn.numel() == 0:
        return conn.reshape(0, 8)

    minimum = int(conn.min())
    maximum = int(conn.max())
    zero_valid = minimum >= 0 and maximum < num_nodes
    one_valid = minimum >= 1 and maximum <= num_nodes

    if indexing == "zero":
        if not zero_valid:
            raise ValueError(
                f"zero-based connectivity must satisfy 0 <= index < {num_nodes}; "
                f"observed [{minimum}, {maximum}]"
            )
        return conn
    if indexing == "one":
        if not one_valid:
            raise ValueError(
                f"one-based connectivity must satisfy 1 <= index <= {num_nodes}; "
                f"observed [{minimum}, {maximum}]"
            )
        return conn - 1
    if indexing != "auto":
        raise ValueError("connectivity indexing must be one of: auto, zero, one")

    if zero_valid and one_valid:
        raise ValueError(
            "AVBP connectivity indexing is ambiguous; configure "
            "connectivity_indexing as zero or one"
        )
    if zero_valid:
        return conn
    if one_valid:
        return conn - 1
    raise ValueError(
        f"connectivity indices [{minimum}, {maximum}] are invalid for num_nodes={num_nodes}"
    )
