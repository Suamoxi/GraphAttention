"""Explicit six-case 3D compressible-HIT AVBP inventory for Experiment B.

Discover fixed-mesh snapshots without loading full CFD arrays. Associate
each source case with its correct physical M3.3 reference file; fail closed
when preparing scientific training. Physical reference values are NEVER
inferred from resolution, case directory names, or snapshots.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import h5py

from .avbp import AVBPHDF5Dataset, AVBPSampleSpec

_CASE = re.compile(r"^Re[0-9]+/Ma[0-9]+(?:\.[0-9]+)?$")
_STATE = ("rho", "rhou", "rhov", "rhow", "rhoE")


@dataclass(frozen=True, slots=True)
class HITCaseSpec:
    case_id: str
    side: int

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not _CASE.fullmatch(self.case_id):
            raise ValueError("case_id must be of the form 'Re54/Ma0.3'")
        if isinstance(self.side, bool) or not isinstance(self.side, int) or self.side < 3:
            raise ValueError("side must be an integer >= 3")


DEFAULT_M43_CASES: tuple[HITCaseSpec, ...] = (
    HITCaseSpec("Re54/Ma0.05", 65),
    HITCaseSpec("Re94/Ma0.05", 129),
    HITCaseSpec("Re54/Ma0.2", 65),
    HITCaseSpec("Re94/Ma0.2", 129),
    HITCaseSpec("Re54/Ma0.3", 65),
    HITCaseSpec("Re94/Ma0.3", 129),
)


def reference_filename(case_id: str) -> str:
    if not _CASE.fullmatch(case_id):
        raise ValueError(f"invalid case ID {case_id!r}")
    return case_id.replace("/", "_") + ".yaml"


def _find_snapshots(directory: Path) -> tuple[Path, ...]:
    locations = (directory / "SOLUT", directory / "RUN" / "SOLUT")
    valid = []
    for candidate in locations:
        if candidate.is_dir():
            paths = tuple(sorted(p.resolve(strict=True) for p in candidate.glob("*.h5")
                                 if p.is_file()))
            if paths:
                valid.append(paths)
    if len(valid) != 1:
        raise ValueError(
            f"expected exactly one nonempty HDF5 SOLUT directory in {directory}; "
            f"found {len(valid)} (refuse ambiguous/empty discovery)"
        )
    return valid[0]


def _inspect_mesh(mesh_path: Path, expected_side: int) -> dict[str, int]:
    # This opens HDF5 metadata only, NOT the coordinate arrays or connectivity.
    with h5py.File(mesh_path, "r") as data:
        dims = {axis: data[f"Coordinates/{axis}"].shape for axis in "xyz"}
        if any(dim != (expected_side**3,) for dim in dims.values()):
            raise ValueError(
                f"{mesh_path}: expected {expected_side**3} nodes/coordinate; got {dims}"
            )
        conn = data["Connectivity/hex->node"]
        cell_count = int(conn.size // 8)
        if conn.size % 8 or cell_count != (expected_side-1)**3:
            raise ValueError(
                f"{mesh_path}: expected {(expected_side-1)**3} hex cells, "
                f"found connectivity shape {conn.shape}"
            )
    return {"nodes": expected_side**3, "hex_cells": cell_count}


def _inspect_snapshot(path: Path, nodes: int) -> None:
    with h5py.File(path, "r") as snapshot:
        for name in _STATE:
            tensor = snapshot[f"GaseousPhase/{name}"]
            if tensor.shape != (nodes,):
                raise ValueError(
                    f"{path}: field {name} expected {(nodes,)}, "
                    f"got {tensor.shape}"
                )
            if tensor.dtype.kind not in "f":
                raise TypeError(f"{path}: field {name} must be floating point")


class AVBPMultiCaseHITDataset(AVBPHDF5Dataset):
    """Read mixed 65^3/129^3 cases with explicit per-case mesh/metadata.

    The dataset exposes sample_ids, case_by_sample_id,
    resolution_by_sample_id and indices_by_case without loading any
    physical snapshot. Mesh tensors are decoded lazily by AVBPHDF5Dataset.
    """

    def __init__(
        self,
        *,
        root: str | Path,
        cases: Sequence[HITCaseSpec] = DEFAULT_M43_CASES,
        reference_dir: str | Path | None = None,
        require_references: bool = True,
        expected_snapshots_per_case: int | None = 100,
    ) -> None:
        self.root = Path(root).expanduser().resolve(strict=True)
        if not self.root.is_dir():
            raise NotADirectoryError(self.root)
        self.cases = tuple(cases)
        if not self.cases or len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("cases must contain unique nonempty physical case IDs")
        if require_references and reference_dir is None:
            raise ValueError(
                "scientific training requires --reference-dir containing explicit "
                "per-case M3.3 YAML definitions; no reference values are inferred"
            )
        self.reference_dir = (
            None if reference_dir is None
            else Path(reference_dir).expanduser().resolve()
        )
        if (expected_snapshots_per_case is not None
            and (isinstance(expected_snapshots_per_case, bool)
                 or expected_snapshots_per_case < 3)):
            raise ValueError("expected_snapshots_per_case must be >=3 or None")

        samples: list[AVBPSampleSpec] = []
        case_files: dict[str, Path] = {}
        inventory: list[dict] = []
        self.case_by_sample_id: dict[str, str] = {}
        self.resolution_by_sample_id: dict[str, int] = {}
        self.indices_by_case: dict[str, tuple[int, ...]] = {}

        for spec in self.cases:
            directory = self.root / spec.case_id
            if not directory.is_dir():
                raise NotADirectoryError(directory)
            mesh = (directory / "MESH" / "mesh.mesh.h5").resolve(strict=True)
            dimensions = _inspect_mesh(mesh, spec.side)
            snapshots = _find_snapshots(directory)
            if (expected_snapshots_per_case is not None
                and len(snapshots) != expected_snapshots_per_case):
                raise ValueError(
                    f"{spec.case_id}: expected {expected_snapshots_per_case} snapshots, "
                    f"found {len(snapshots)}"
                )
            # Check first, middle and last HDF5 schemas. Loading only metadata
            # makes archive inventory inexpensive and catches common format drift.
            for index in sorted({0, len(snapshots)//2, len(snapshots)-1}):
                _inspect_snapshot(snapshots[index], dimensions["nodes"])

            definition = (
                None if self.reference_dir is None
                else self.reference_dir / reference_filename(spec.case_id)
            )
            if require_references:
                assert definition is not None
                if not definition.is_file():
                    raise FileNotFoundError(
                        f"missing authoritative case reference YAML: {definition}. "
                        "Review case_info.json and RUN/run.params first."
                    )
                case_files[spec.case_id] = definition.resolve(strict=True)

            start = len(samples)
            for path in snapshots:
                sample_id = f"{spec.case_id}:{path.stem}"
                if sample_id in self.case_by_sample_id:
                    raise ValueError(f"duplicate physical sample ID {sample_id}")
                self.case_by_sample_id[sample_id] = spec.case_id
                self.resolution_by_sample_id[sample_id] = spec.side
                samples.append(AVBPSampleSpec(
                    sample_id=sample_id,
                    snapshot_file=path,
                    mesh_id=f"HIT_DNS_COMP:{spec.case_id}",
                    mesh_file=mesh,
                    case_id=spec.case_id if require_references else None,
                ))
            self.indices_by_case[spec.case_id] = tuple(range(start, len(samples)))
            inventory.append({
                "case_id": spec.case_id,
                "side": spec.side,
                **dimensions,
                "count": len(snapshots),
                "solution_folder": str(snapshots[0].parent),
                "mesh": str(mesh),
                "first_snapshot": str(snapshots[0]),
                "last_snapshot": str(snapshots[-1]),
                "case_info_file": (
                    str(directory / "case_info.json")
                    if (directory / "case_info.json").is_file() else None
                ),
                "run_params_file": (
                    str(directory / "RUN" / "run.params")
                    if (directory / "RUN" / "run.params").is_file() else None
                ),
                "required_reference_file": (
                    str(definition) if definition is not None else None
                ),
                "reference_file_exists": (
                    bool(definition and definition.is_file())
                ),
            })

        self.inventory = tuple(inventory)
        self.sample_ids = tuple(spec.sample_id for spec in samples)
        super().__init__(samples=samples, case_files=case_files, field_names=_STATE)
        if require_references:
            for spec in self.cases:
                definitions = self.case_definitions[spec.case_id]
                names = set(definitions.reference_scales.names)
                needed = {"rho_ref", "U_ref", "L_ref"}
                if not needed.issubset(names):
                    raise ValueError(
                        f"{spec.case_id}: case YAML missing convective scales "
                        f"{sorted(needed-names)}"
                    )
        self.require_references = require_references

    def group_id(self, index: int, key: str) -> str:
        sample_id = self.sample_ids[index]
        if key == "case_id":
            return self.case_by_sample_id[sample_id]
        if key == "resolution":
            return str(self.resolution_by_sample_id[sample_id])
        raise KeyError(f"unsupported mixed-HIT group key {key!r}")
