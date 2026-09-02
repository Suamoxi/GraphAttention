"""AVBP HDF5 reader backed by the explicit M2 field and mesh contracts."""

from __future__ import annotations

from collections.abc import Sequence
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


class AVBPHDF5Dataset(Dataset[Sample]):
    """Read named AVBP HDF5 fields and native mesh information.

    M3.2 intentionally stops at the data boundary. Cell-to-node connectivity is
    decoded and stored on ``Mesh.cell_connectivity``. ``Mesh.edge_index`` stays
    empty until a geometry transform explicitly constructs graph edges.
    """

    def __init__(
        self,
        files: Sequence[str | Path] | None = None,
        data_dir: str | Path | None = None,
        file_pattern: str = "*.h5",
        recursive: bool = False,
        field_names: Sequence[str] = _DEFAULT_FIELDS,
        coord_paths: Sequence[str] = _DEFAULT_COORD_PATHS,
        connectivity_path: str = _DEFAULT_CONNECTIVITY_PATH,
        connectivity_indexing: str = "auto",
        mesh_file: str | Path | None = None,
        catalog: FieldCatalog | None = None,
    ) -> None:
        self.files = _resolve_files(files, data_dir, file_pattern, recursive)
        self.field_catalog = catalog or AVBP_FIELD_CATALOG
        self.field_names = tuple(field_names)
        if not self.field_names:
            raise ValueError("field_names must contain at least one field")
        if len(set(self.field_names)) != len(self.field_names):
            raise ValueError("field_names must be unique")
        self.field_specs = self.field_catalog.require(list(self.field_names))
        for spec in self.field_specs:
            if not spec.stored or spec.source_path is None:
                raise ValueError(f"AVBP reader requires a stored source_path for field '{spec.name}'")
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

        self.mesh_file = Path(mesh_file).expanduser() if mesh_file is not None else None
        if self.mesh_file is not None and not self.mesh_file.is_file():
            raise FileNotFoundError(f"mesh_file does not exist: {self.mesh_file}")
        self._shared_mesh = self._read_mesh(self.mesh_file) if self.mesh_file is not None else None

        sample_ids = [path.stem for path in self.files]
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("AVBP sample file stems must be unique")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Sample:
        sample_index = operator_index(index)
        path = self.files[sample_index]
        mesh = self._shared_mesh if self._shared_mesh is not None else self._read_mesh(path)
        fields = self._read_fields(path)
        sample = Sample(
            sample_id=path.stem,
            mesh=mesh,
            fields=fields,
            metadata={"format": "avbp_hdf5", "source_file": str(path)},
        )
        sample.validate_against(self.field_catalog)
        return sample

    def _read_fields(self, path: Path) -> dict[str, torch.Tensor]:
        fields: dict[str, torch.Tensor] = {}
        with h5py.File(path, "r") as h5f:
            for spec in self.field_specs:
                tensor = _read_tensor(h5f, spec.source_path, path)
                fields[spec.name] = _canonicalize_node_field(tensor, spec, path)
        return fields

    def _read_mesh(self, path: Path) -> Mesh:
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
            mesh_id=path.stem,
            cell_connectivity=cell_connectivity,
            metadata={
                "format": "avbp_hdf5",
                "source_file": str(path),
                "cell_type": "hex",
                "coord_paths": self.coord_paths,
                "connectivity_path": self.connectivity_path,
            },
        )


def _resolve_files(
    files: Sequence[str | Path] | None,
    data_dir: str | Path | None,
    file_pattern: str,
    recursive: bool,
) -> tuple[Path, ...]:
    if isinstance(files, (str, Path)):
        raise TypeError("files must be a sequence of paths, not one path string")
    explicit = tuple(Path(path).expanduser() for path in (files or ()))
    if explicit and data_dir is not None:
        raise ValueError("configure either files or data_dir, not both")

    if explicit:
        resolved = explicit
    elif data_dir is not None:
        root = Path(data_dir).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"data_dir does not exist or is not a directory: {root}")
        if not file_pattern:
            raise ValueError("file_pattern must be non-empty")
        iterator = root.rglob(file_pattern) if recursive else root.glob(file_pattern)
        resolved = tuple(sorted(path for path in iterator if path.is_file()))
    else:
        raise ValueError("configure at least one AVBP file through files or data_dir")

    if not resolved:
        raise ValueError("no AVBP HDF5 files matched the configured input")
    for path in resolved:
        if not path.is_file():
            raise FileNotFoundError(f"AVBP sample file does not exist: {path}")
    if len(set(resolved)) != len(resolved):
        raise ValueError("AVBP sample files must be unique")
    return resolved


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
            "AVBP connectivity indexing is ambiguous; configure connectivity_indexing as zero or one"
        )
    if zero_valid:
        return conn
    if one_valid:
        return conn - 1
    raise ValueError(
        f"connectivity indices [{minimum}, {maximum}] are invalid for num_nodes={num_nodes}"
    )
