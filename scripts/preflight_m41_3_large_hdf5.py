"""M41.3: bounded-RAM HDF5 preflight for a real large AVBP mesh/snapshot.

Reads only HDF5 metadata and small bounded samples by default: do NOT run the
existing M41.3 global-edge planner on a 257^3 mesh before this gate. Optional
connectivity-extrema scanning reads the file in bounded chunks and establishes
zero- versus one-based connectivity only when extrema disambiguate it.
No graph edges, exact-two-hop topology, training data or CUDA arrays are built.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

DEFAULT_COORDS = ("Coordinates/x", "Coordinates/y", "Coordinates/z")
DEFAULT_FIELDS = (
    "GaseousPhase/rho", "GaseousPhase/rhou", "GaseousPhase/rhov",
    "GaseousPhase/rhow", "GaseousPhase/rhoE",
)
DEFAULT_CONNECTIVITY = "Connectivity/hex->node"


def _gib(num_bytes: int | float) -> float:
    return round(float(num_bytes) / 1024**3, 4)


def _metadata(handle: h5py.File, name: str) -> dict:
    if name not in handle:
        raise KeyError(f"required HDF5 dataset is missing: {name}")
    obj = handle[name]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError(f"HDF5 item is not a dataset: {name}")
    return {
        "path": name,
        "shape": list(obj.shape),
        "dtype": str(obj.dtype),
        "uncompressed_gib": _gib(obj.size * obj.dtype.itemsize),
        "compression": obj.compression,
        "chunks": None if obj.chunks is None else list(obj.chunks),
    }


def _first_axis_count(dataset: h5py.Dataset) -> int:
    shape = dataset.shape
    if len(shape) == 1:
        return int(shape[0])
    if len(shape) == 2 and shape[1] == 1:
        return int(shape[0])
    raise ValueError(
        f"AVBP node field or axis must have [N] or [N,1] shape; found {shape}"
    )


def _sample(dataset: h5py.Dataset, elements: int = 128) -> dict:
    """Read at most `elements` first/last scalar entries of a dataset."""
    if dataset.size == 0:
        return {"checked_scalars": 0, "finite": True, "min": None, "max": None}
    # Read two bounded hyperslabs, not the complete dataset.
    if dataset.ndim == 1:
        a = np.asarray(dataset[: min(elements, dataset.shape[0])])
        b = np.asarray(dataset[max(0, dataset.shape[0] - elements) :])
    elif dataset.ndim == 2:
        if dataset.shape[0] == 8 and dataset.shape[1] > 8:
            # [8,C] AVBP connectivity: chunk along the cell axis.
            width = max(1, elements // dataset.shape[0])
            end = min(width, dataset.shape[1])
            a = np.asarray(dataset[:, :end])
            b = np.asarray(dataset[:, max(0, dataset.shape[1] - width):])
        else:
            cols = max(1, min(dataset.shape[1], elements))
            rows = max(1, elements // cols)
            first_rows = min(rows, dataset.shape[0])
            a = np.asarray(dataset[:first_rows, :cols])
            b = np.asarray(dataset[max(0, dataset.shape[0] - first_rows):, :cols])
    else:
        raise ValueError(f"unsupported HDF5 dataset rank {dataset.ndim}")
    values = np.concatenate((a.ravel(), b.ravel()))
    finite = bool(np.isfinite(values).all())
    return {
        "checked_scalars": int(values.size),
        "finite": finite,
        "min": float(values.min()) if finite else None,
        "max": float(values.max()) if finite else None,
    }


def _connectivity_count(dataset: h5py.Dataset) -> int:
    shape = dataset.shape
    if len(shape) == 1 and shape[0] % 8 == 0:
        return shape[0] // 8
    if len(shape) == 2 and shape[1] == 8:
        return shape[0]
    if len(shape) == 2 and shape[0] == 8:
        return shape[1]
    raise ValueError(
        f"AVBP hexahedral connectivity expects [C,8], [8,C], or [8*C]; got {shape}"
    )


def _scan_connectivity_extrema(dataset: h5py.Dataset, *, elements: int) -> dict:
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError("AVBP connectivity must have integer dtype")
    # Iterate along the first HDF5 axis with a constant <= elements
    # scalar read budget, regardless of [C,8], [8,C], or flat layout.
    minimum = None
    maximum = None
    if dataset.ndim == 2 and dataset.shape[0] == 8 and dataset.shape[1] > 8:
        step = max(1, elements // 8)
        slabs = (
            np.asarray(dataset[:, start : start + step])
            for start in range(0, dataset.shape[1], step)
        )
    else:
        width = 1 if dataset.ndim == 1 else dataset.shape[1]
        rows = max(1, elements // width)
        slabs = (
            np.asarray(dataset[start : start + rows])
            for start in range(0, dataset.shape[0], rows)
        )
    for slab in slabs:
        if slab.size == 0:
            continue
        lo, hi = int(slab.min()), int(slab.max())
        minimum = lo if minimum is None else min(minimum, lo)
        maximum = hi if maximum is None else max(maximum, hi)
    return {"minimum": minimum, "maximum": maximum, "all_connectivity_read": True}


def _indexing_verdict(extrema: dict, n: int) -> str:
    lower, upper = extrema["minimum"], extrema["maximum"]
    if lower is None or upper is None:
        return "unknown_empty"
    zero_valid = lower >= 0 and upper < n
    one_valid = lower >= 1 and upper <= n
    if zero_valid and one_valid:
        return "ambiguous: set connectivity_indexing=zero or one explicitly"
    if zero_valid:
        return "zero_based"
    if one_valid:
        return "one_based"
    return "INVALID_OUT_OF_RANGE"


def inspect(
    mesh_file: Path,
    snapshot_file: Path,
    *,
    expected_nodes: int = 257**3,
    hidden_dim: int = 128,
    connectivity_path: str = DEFAULT_CONNECTIVITY,
    coord_paths: tuple[str, ...] = DEFAULT_COORDS,
    field_paths: tuple[str, ...] = DEFAULT_FIELDS,
    scan_connectivity_extrema: bool = False,
    scan_elements: int = 262_144,
) -> dict:
    mesh_file = mesh_file.expanduser().resolve(strict=True)
    snapshot_file = snapshot_file.expanduser().resolve(strict=True)
    if expected_nodes < 1 or hidden_dim < 1 or scan_elements < 8:
        raise ValueError("expected_nodes, hidden_dim, and scan_elements must be positive")

    with h5py.File(mesh_file, "r") as mesh, h5py.File(snapshot_file, "r") as sol:
        coord_info = [_metadata(mesh, name) for name in coord_paths]
        coord_count = [_first_axis_count(mesh[name]) for name in coord_paths]
        if len(set(coord_count)) != 1:
            raise ValueError(f"mesh coordinate axes disagree on node count: {coord_count}")
        nodes = coord_count[0]
        if nodes != expected_nodes:
            raise ValueError(f"expected {expected_nodes} mesh nodes, found {nodes}")

        connectivity_info = _metadata(mesh, connectivity_path)
        dataset = mesh[connectivity_path]
        if not np.issubdtype(dataset.dtype, np.integer):
            raise TypeError("AVBP connectivity must have integer dtype")
        cells = _connectivity_count(dataset)

        field_info = [_metadata(sol, name) for name in field_paths]
        for name in field_paths:
            field = sol[name]
            if _first_axis_count(field) != nodes:
                raise ValueError(
                    f"{name} has {_first_axis_count(field)} values, but mesh has {nodes} nodes"
                )
            if not np.issubdtype(field.dtype, np.floating):
                raise TypeError(f"{name} must have floating dtype")
        for name in coord_paths:
            if not np.issubdtype(mesh[name].dtype, np.floating):
                raise TypeError(f"{name} must have floating dtype")

        coord_samples = {name: _sample(mesh[name]) for name in coord_paths}
        field_samples = {name: _sample(sol[name]) for name in field_paths}
        if not all(
            sample["finite"]
            for sample in (*coord_samples.values(), *field_samples.values())
        ):
            raise ValueError("nonfinite mesh coordinate or field detected in bounded sample")

        # Sampled connectivity extrema are merely indicative. They are NOT
        # used to decide zero- or one-based indexing.
        sampled_connectivity = _sample(dataset)
        extrema = None
        verdict = "not scanned: cannot establish global indexing from samples"
        if scan_connectivity_extrema:
            extrema = _scan_connectivity_extrema(dataset, elements=scan_elements)
            verdict = _indexing_verdict(extrema, nodes)
            if verdict == "INVALID_OUT_OF_RANGE":
                raise ValueError(
                    f"invalid connectivity extrema {extrema} for {nodes} nodes"
                )

    structured = round(nodes ** (1.0 / 3.0))
    is_cubic_count = structured**3 == nodes
    # These edge counts are **theoretical** for a nonperiodic
    # axis-aligned structured 6-neighbor cubic grid only.
    edges = 6 * structured * structured * (structured - 1) if is_cubic_count else None
    twohop = (
        6 * structured * structured * (structured - 2)
        + 12 * structured * (structured - 1) ** 2
        if is_cubic_count and structured >= 3 else None
    )
    return {
        "status": "metadata_compatible_only_not_graph_or_inference_validated",
        "mesh_file": str(mesh_file),
        "snapshot_file": str(snapshot_file),
        "mesh_file_gib": _gib(mesh_file.stat().st_size),
        "snapshot_file_gib": _gib(snapshot_file.stat().st_size),
        "num_nodes": nodes,
        "num_hex_cells": cells,
        "node_count_is_perfect_cube": is_cubic_count,
        "cube_side_length_if_cubic": structured if is_cubic_count else None,
        "expected_regular_cube_hex_cells": (
            (structured - 1) ** 3 if is_cubic_count else None
        ),
        "coord_datasets": coord_info,
        "connectivity_dataset": connectivity_info,
        "field_datasets": field_info,
        "coord_sample_extrema": coord_samples,
        "field_sample_extrema": field_samples,
        "connectivity_sample_extrema": sampled_connectivity,
        "connectivity_global_extrema": extrema,
        "connectivity_indexing": verdict,
        "theoretical_6neighbor_directed_edges": edges,
        "theoretical_exact2hop_directed_edges": twohop,
        "host_hidden_two_buffer_gib": _gib(2 * nodes * hidden_dim * 4),
        "one_global_gpu_hidden_gib": _gib(nodes * hidden_dim * 4),
        "one_global_gpu_qkv_gib": _gib(3 * nodes * hidden_dim * 4),
        "one_global_gpu_mlp_ratio4_gib": _gib(4 * nodes * hidden_dim * 4),
        "raw_hex_connectivity_int64_gib": _gib(cells * 8 * 8),
        "existing_hex_pair_candidates_int64_gib": _gib(cells * 12 * 2 * 8),
        "caveats": [
            "Shape and bounded samples do not prove Cartesian ordering or periodic connectivity.",
            "Current GraphTaskCollator materializes complete hex edge candidates, unique sorted edges, and global two-hop paths.",
            "Current M41.3 host planner still scans global edges per partition; this preflight does not attempt it.",
            "Neither full graph construction nor GPU memory scaling is validated here.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh-file", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-nodes", type=int, default=257**3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--connectivity-path", default=DEFAULT_CONNECTIVITY)
    parser.add_argument("--scan-connectivity-extrema", action="store_true")
    parser.add_argument("--scan-elements", type=int, default=262_144)
    args = parser.parse_args()
    result = inspect(
        args.mesh_file, args.snapshot_file,
        expected_nodes=args.expected_nodes,
        hidden_dim=args.hidden_dim,
        connectivity_path=args.connectivity_path,
        scan_connectivity_extrema=args.scan_connectivity_extrema,
        scan_elements=args.scan_elements,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
