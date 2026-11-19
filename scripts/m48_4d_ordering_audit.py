"""Independent read-only AVBP -> MS_SR -> GraphAttention periodic ordering audit.

A physical Cartesian index is independently derived from original AVBP XYZ
coordinates; it is never inferred from the CEDAR or GraphAttention ordering.
The complete mesh, source-to-unique map, all five source fields, and all
published hierarchy levels are checked. No source HDF5 is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

import h5py
import numpy as np

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
FILTER = "w1p25_l1"


def sha256_file(path: Path, block_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for part in iter(lambda: source.read(block_bytes), b""):
            digest.update(part)
    return digest.hexdigest()


def _check(condition: bool, message: str) -> None:
    if not bool(condition):
        raise RuntimeError(message)


def _safe_path(root: Path, relative: str) -> Path:
    """Reject traversal while supporting the published payload symlinks."""
    p = Path(relative)
    if p.is_absolute() or not p.parts or any(s in ("", ".", "..") for s in p.parts):
        raise ValueError(f"invalid manifest relative path: {relative}")
    return root / p


def independent_cartesian_map(
    coords: np.ndarray, *, native_side: int, native_spacing: np.ndarray,
    tolerance_fraction: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Physical XYZ -> periodic (i,j,k) and C-order ID, independently of CEDAR.

    The verified HIT pilot is an axis-aligned Cartesian box with both periodic
    endpoint planes present. This must fail for rotated/irregular meshes.
    """
    n = native_side
    spacing = np.asarray(native_spacing, dtype=np.float64)
    if (coords.ndim != 2 or coords.shape[1] != 3 or n < 3
            or spacing.shape != (3,) or not np.isfinite(spacing).all()
            or np.any(spacing <= 0) or not np.isfinite(coords).all()):
        raise ValueError("invalid raw Cartesian geometry")
    origin = coords.min(axis=0)
    high = coords.max(axis=0)
    actual_length = high - origin
    expected_length = spacing * n
    _check(np.allclose(actual_length, expected_length, rtol=1e-7, atol=1e-11),
           "AVBP periodic box lengths disagree with MS_SR native spacing")
    scaled = (coords - origin) / spacing
    rounded = np.rint(scaled).astype(np.int64)
    error = np.max(np.abs(scaled - rounded), axis=0)
    _check(np.all(error <= tolerance_fraction),
           f"raw coordinates not on the expected Cartesian grid: {error.tolist()}")
    _check(np.all((rounded >= 0) & (rounded <= n)),
           "raw coordinates outside the N+1 periodic endpoint grid")
    periodic_ijk = rounded % n
    linear = (periodic_ijk[:, 0] * n + periodic_ijk[:, 1]) * n + periodic_ijk[:, 2]
    return periodic_ijk, linear, {
        "native_side": n, "native_spacing": spacing.tolist(),
        "coordinate_origin": origin.tolist(),
        "periodic_length_by_axis": actual_length.tolist(),
        "max_coordinate_error_in_cell_widths": error.tolist(),
        "periodic_endpoint_alias_nodes": int(np.any(rounded == n, axis=1).sum()),
    }


def verify_lattice_mapping(
    periodic_ijk: np.ndarray, independent_linear: np.ndarray,
    representatives: np.ndarray, source_to_unique: np.ndarray,
    native_side: int,
) -> dict:
    """Compare independently ordered AVBP nodes to the actual CEDAR map."""
    n = native_side
    reps = np.asarray(representatives, dtype=np.int64)
    mapping = np.asarray(source_to_unique, dtype=np.int64)
    count = n**3
    _check(reps.shape == (count,), "CEDAR representative count mismatch")
    _check(mapping.shape == independent_linear.shape, "CEDAR raw source count mismatch")
    _check(np.all((reps >= 0) & (reps < len(mapping))), "invalid representative IDs")
    _check(np.all((mapping >= 0) & (mapping < count)), "invalid CEDAR raw-node mapping")
    _check(np.array_equal(mapping, independent_linear),
           "CEDAR source_to_unique does not match physical XYZ -> GraphAttention ID")
    _check(np.array_equal(independent_linear[reps], np.arange(count, dtype=np.int64)),
           "CEDAR representatives are not ordered as GraphAttention (x*N+y)*N+z")
    _check(np.unique(reps).size == count, "duplicate CEDAR representative raw IDs")
    return {"original_nodes_verified": int(len(mapping)),
            "unique_cartesian_nodes_verified": count,
            "source_to_unique_mismatches": 0,
            "representative_cartesian_mismatches": 0}


def _hex_bases_from_periodic_corners(
    corners: np.ndarray, native_side: int,
) -> np.ndarray:
    """Independently recognize 8 corners of one periodic Cartesian hexahedron.

    `corners` is [C,8,3] with coordinates reduced modulo native_side.
    No assumption is made about the AVBP local vertex enumeration.
    """
    n = native_side
    if corners.ndim != 3 or corners.shape[1:] != (8, 3):
        raise ValueError("hex corners must have shape [C,8,3]")
    bases = np.empty((len(corners), 3), dtype=np.int64)
    differences = np.empty_like(corners)
    for axis in range(3):
        x = corners[:, :, axis]
        first = x[:, 0]
        following = (first + 1) % n
        previous = (first - 1) % n
        forward = np.all((x == first[:, None]) | (x == following[:, None]), axis=1)
        backward = np.all((x == previous[:, None]) | (x == first[:, None]), axis=1)
        valid = forward | backward
        _check(np.all(valid), f"AVBP hex has non-neighbor vertices on axis {axis}")
        base = np.where(forward, first, previous)
        bases[:, axis] = base
        differences[:, :, axis] = (x - base[:, None]) % n
    _check(np.all(differences <= 1), "AVBP hex has a nonlocal periodic edge")
    bits = 4 * differences[:, :, 0] + 2 * differences[:, :, 1] + differences[:, :, 2]
    _check(np.all(np.sort(bits, axis=1) == np.arange(8, dtype=np.int64)),
           "AVBP hex does not contain exactly the eight Cartesian corners")
    return bases


def verify_periodic_hex_cells(
    connectivity: h5py.Dataset, independent_ijk: np.ndarray, *,
    native_side: int, chunk_cells: int = 65536,
) -> dict:
    """Full streaming AVBP connectivity audit, including periodic seams.

    Require exactly one hex for each of N^3 periodic cells. This checks the
    actual original mesh, not a second graph created by GraphAttention.
    """
    n = native_side
    expected_cells = n**3
    _check(connectivity.ndim == 1 and connectivity.shape == (expected_cells * 8,),
           "AVBP connectivity is not the expected native Cartesian hex mesh")
    _check(chunk_cells > 0, "chunk_cells must be positive")
    seen = np.zeros(expected_cells, dtype=np.bool_)
    seam_counts = np.zeros(3, dtype=np.int64)
    pair_seams = np.zeros(3, dtype=np.int64)
    corner_seams = 0
    for start in range(0, expected_cells, chunk_cells):
        stop = min(start + chunk_cells, expected_cells)
        source_ids = np.asarray(connectivity[8*start:8*stop], dtype=np.int64).reshape(-1, 8)
        _check(np.all((source_ids >= 1) & (source_ids <= len(independent_ijk))),
               f"invalid one-based AVBP cell IDs in cells [{start},{stop})")
        corners = independent_ijk[source_ids - 1]
        bases = _hex_bases_from_periodic_corners(corners, n)
        lin = (bases[:, 0] * n + bases[:, 1]) * n + bases[:, 2]
        _check(np.unique(lin).size == len(lin) and not seen[lin].any(),
               f"duplicate periodic Cartesian cell in cells [{start},{stop})")
        seen[lin] = True
        seam = bases == n - 1
        seam_counts += seam.sum(axis=0)
        pair_seams += np.array([
            np.count_nonzero(seam[:, 0] & seam[:, 1]),
            np.count_nonzero(seam[:, 0] & seam[:, 2]),
            np.count_nonzero(seam[:, 1] & seam[:, 2]),
        ])
        corner_seams += int(np.all(seam, axis=1).sum())
    _check(bool(seen.all()), "missing periodic Cartesian cells in original AVBP mesh")
    _check(np.all(seam_counts == n*n), "not all three periodic face seams are connected")
    _check(np.all(pair_seams == n), "missing periodic edge seam connectivity")
    _check(corner_seams == 1, "missing periodic corner seam connectivity")
    return {
        "all_native_hex_cells_verified": expected_cells,
        "duplicate_or_missing_cells": 0,
        "periodic_face_seam_cells_xyz": seam_counts.tolist(),
        "periodic_pair_seam_cells_xy_xz_yz": pair_seams.tolist(),
        "periodic_corner_seam_cells": corner_seams,
    }


def compare_native_component(
    raw: np.ndarray, native_qf: np.ndarray, reps: np.ndarray,
    independent_ids: np.ndarray, *, native_side: int,
) -> dict:
    """Check all qf entries, not just samples, plus all periodic aliases."""
    n = native_side
    expected = raw[reps].reshape((n, n, n))
    _check(np.array_equal(expected, native_qf),
           "published MS_SR qf values differ from AVBP values in Cartesian order")
    alias_mask = np.ones(len(raw), dtype=np.bool_)
    alias_mask[reps] = False
    aliases = np.flatnonzero(alias_mask)
    representative_for_raw = reps[independent_ids[aliases]]
    reference = raw[representative_for_raw]
    error = raw[aliases] - reference
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(reference), 1e-300))
    _check(np.isfinite(relative) and relative <= 1e-10,
           f"AVBP periodic alias field mismatch: {relative}")
    return {"native_values_verified": n**3, "exact_native_value_mismatches": 0,
            "periodic_alias_values_checked": len(aliases),
            "periodic_alias_relative_l2": relative}


def compare_filtered_levels(
    native_qf: np.ndarray, h5: h5py.File, *, component: int,
    operator, filter_id: str = FILTER,
) -> dict:
    """Rebuild published q2/q4/q8 from qf with the pinned original CEDAR."""
    n = native_qf.shape[0]
    _check(native_qf.shape == (n, n, n), "invalid qf component shape")
    levels = operator.repeat(3).restrict_levels(native_qf, native_qf.shape)
    report = {}
    for index, factor in enumerate((2, 4, 8), start=1):
        key = f"filters/{filter_id}/q{factor}"
        stored = np.asarray(h5[key][..., component], dtype=np.float64)
        predicted = np.asarray(levels[index], dtype=np.float64)
        _check(stored.shape == predicted.shape == (n // factor,) * 3,
               f"unexpected published {key} shape")
        diff = float(np.max(np.abs(stored - predicted)))
        scale = max(1.0, float(np.max(np.abs(stored))))
        _check(np.isfinite(diff) and diff <= 1e-10 * scale,
               f"published {key} disagrees with full-domain CEDAR hierarchy: {diff}")
        report[f"q{factor}_max_abs_error"] = diff
    return report


def _verified_manifest_and_records(ms_sr_root: Path, case_names: list[str] | None):
    from graph_attention.data.ms_sr_training import select_role_records
    records, seal = select_role_records(
        ms_sr_root, role="teacher", filter_id=FILTER,
        case_prefix="", max_sources=0,
    )
    _check(len(records) == 44 and len({r["case"] for r in records}) == 22,
           "unexpected M48 44-teacher source selection")
    if case_names:
        _check(len(case_names) == 2 and len(set(case_names)) == 2,
               "choose exactly one original teacher case in each Reynolds group")
        selection = []
        for case in case_names:
            hit = next((r for r in records if r["case"] == case), None)
            _check(hit is not None, f"requested case is not an eligible teacher source: {case}")
            selection.append(hit)
    else:
        selection = []
        for group in ("ReT060/", "ReT100/"):
            hit = next((r for r in records if r["case"].startswith(group)), None)
            _check(hit is not None, f"missing teacher group: {group}")
            selection.append(hit)
    _check({r["case"].split("/")[0] for r in selection} == {"ReT060", "ReT100"},
           "source ordering audit must cover both Reynolds groups")
    return selection, seal


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def audit_record(args, record: dict, selected_sha: str) -> dict:
    from cedar.mesh.periodic_cartesian import PeriodicHexLattice
    from cedar.plugins.nodal_operator_design.operators import NodalOperator
    from preprocess import check_cedar
    from graph_attention.data.ms_sr_adapter import load_ms_sr_pair, flat_channels

    check_cedar()
    pilot_root = args.pilot_root
    root = args.ms_sr_root
    case = record["case"]
    snapshot = str(record["snapshot_id"])
    source_rel = None
    publication = json.loads((root / "dataset" / "manifest.json").read_text())
    sources = [s["source"] for s in publication["sources"]
               if s["source"]["case"] == case
               and str(s["source"]["snapshot_id"]) == snapshot]
    _check(len(sources) == 1, "source does not have exactly one published raw pilot identity")
    source = sources[0]
    _check(source["source_role"] == "teacher" and
           source["expected_source_sha256"] == record["source_sha256"],
           "raw source identity disagrees with selected training pair")
    source_rel = source["archive_relative_path"]
    avbp = _safe_path(pilot_root / "data", source_rel)
    mesh = _safe_path(pilot_root / "data", f"{case}/MESH/mesh.mesh.h5")
    payload = _safe_path(root / "dataset", record["path"])
    for path in (avbp, mesh, payload):
        _check(path.is_file(), f"missing read-only input file {path}")
    if not args.skip_file_hashes:
        for path, expected in (
            (avbp, record["source_sha256"]),
            (mesh, record["mesh_sha256"]),
            (payload, record["sha256"]),
        ):
            _check(sha256_file(path) == expected, f"SHA-256 mismatch for {path}")

    started = time.perf_counter()
    with h5py.File(mesh, "r") as mesh_h5, h5py.File(avbp, "r") as raw_h5, h5py.File(payload, "r") as pub_h5:
        n = int(pub_h5["qf"].shape[0])
        _check(pub_h5["qf"].shape == (n, n, n, 5) and n in (192, 320),
               "unexpected native DNS field shape")
        _check(tuple(record["target_shape"]) == (n//4,)*3 and
               tuple(record["input_shape"]) == (n//8,)*3,
               "M48 q8->q4 record dimensions do not match native qf")
        spacing = np.asarray(record["fine_spacing"], dtype=np.float64) / 4
        coords = np.column_stack([
            np.asarray(mesh_h5[f"Coordinates/{axis}"], dtype=np.float64)
            for axis in "xyz"
        ])
        independent_ijk, independent_ids, coord_report = independent_cartesian_map(
            coords, native_side=n, native_spacing=spacing,
            tolerance_fraction=args.coordinate_tolerance,
        )
        lattice = PeriodicHexLattice.from_coords(coords)
        _check(tuple(lattice.shape) == (n, n, n), "CEDAR native lattice dimensions differ")
        mapping_report = verify_lattice_mapping(
            independent_ijk, independent_ids,
            lattice.representative_ids, lattice.source_to_unique, n,
        )
        connectivity_report = verify_periodic_hex_cells(
            mesh_h5["Connectivity/hex->node"], independent_ijk,
            native_side=n, chunk_cells=args.chunk_cells,
        )
        operator = NodalOperator.builtin(2, family="optimized", omega=1.25, lambda_=1)
        _check(operator.repeat(1).hash == record["operator_sha256"],
               "CEDAR q4->q8 operator SHA differs from manifest")
        fields = {}
        reps = np.asarray(lattice.representative_ids, dtype=np.int64)
        for component, field in enumerate(FIELDS):
            raw_values = np.asarray(raw_h5[f"GaseousPhase/{field}"], dtype=np.float64)
            _check(raw_values.shape == (len(coords),), "AVBP conservative component shape mismatch")
            qf = np.asarray(pub_h5["qf"][..., component], dtype=np.float64)
            native_check = compare_native_component(
                raw_values, qf, reps, independent_ids, native_side=n,
            )
            levels_check = compare_filtered_levels(qf, pub_h5,
                component=component, operator=operator)
            fields[field] = {**native_check, **levels_check}
        # The exact current M48 loader must read both published arrays in the
        # same order as the independent HDF5 arrays and flattening convention.
        pair, _, cedar_checks = load_ms_sr_pair(
            root, role="teacher", filter_id=FILTER, fine_factor=4,
            coarse_factor=8, case_prefix=case, snapshot_case=case,
            snapshot_id=snapshot,
        )
        expected_fine = np.asarray(pub_h5[f"filters/{FILTER}/q4"][...], dtype=np.float64)
        expected_coarse = np.asarray(pub_h5[f"filters/{FILTER}/q8"][...], dtype=np.float64)
        _check(np.array_equal(flat_channels(pair.fine).numpy(), expected_fine.reshape(-1, 5)),
               "GraphAttention fine flattening changes q4 original HDF5 node order")
        _check(np.array_equal(flat_channels(pair.coarse).numpy(), expected_coarse.reshape(-1, 5)),
               "GraphAttention coarse flattening changes q8 original HDF5 node order")
        _check(all(x["max_component_scaled_linf"] < 1e-9 for x in cedar_checks.values()),
               "full pair CEDAR qualification failed")
    return {
        "status": "PASS", "case": case, "snapshot_id": snapshot,
        "role": "teacher", "filter_id": FILTER, "fine_factor": 4, "coarse_factor": 8,
        "source_selection_sha256": selected_sha,
        "original_source_sha256": record["source_sha256"],
        "original_mesh_sha256": record["mesh_sha256"],
        "published_payload_sha256": record["sha256"],
        "file_hashes_reverified": not args.skip_file_hashes,
        "source_path": str(avbp), "mesh_path": str(mesh),
        "published_payload_path": str(payload),
        "native_shape": [n, n, n], "coarse_shape": record["input_shape"],
        "fine_shape": record["target_shape"],
        "independent_coordinate_mapping": coord_report,
        "cedar_vs_independent_mapping": mapping_report,
        "original_periodic_cell_connectivity": connectivity_report,
        "field_reconstruction": fields,
        "graphattention_q4_q8_flattening": "bitwise_equal_all_nodes_all_components",
        "cedar_q4_to_q8_checks": cedar_checks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def _write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".ordering-", suffix=".json", delete=False) as stream:
        name = Path(stream.name)
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(name, path)


def run(args) -> dict:
    from graph_attention.data.ms_sr_adapter import _add_external_sources, MANIFEST_SHA256
    if (args.chunk_cells < 1 or not (0 < args.coordinate_tolerance <= 0.01)
            or args.output.exists()):
        raise ValueError("invalid audit options or output receipt already exists")
    _add_external_sources(args.ms_sr_root)
    records, selection_sha = _verified_manifest_and_records(args.ms_sr_root, args.cases)
    details = {
        "schema": "graph_attention.m48_4d_ordering_audit.v1",
        "status": "RUNNING", "read_only_inputs": True,
        "source_commit": _git_sha(), "manifest_sha256": MANIFEST_SHA256,
        "teacher_selection_sha256": selection_sha,
        "cases": [],
        "requirements": {
            "independent_original_xyz_cartesian_mapping": True,
            "original_mesh_all_periodic_cells_and_seams": True,
            "all_five_native_qf_fields_bitwise_recovered": True,
            "full_native_cedar_q2_q4_q8_regeneration": True,
            "graphattention_fine_coarse_loader_order_bitwise_equal": True,
        },
    }
    try:
        for record in records:
            entry = audit_record(args, record, selection_sha)
            details["cases"].append(entry)
            _write_json_atomic(args.output.with_name(
                args.output.stem + "_" + entry["case"].split("/")[0] + ".json"
            ), entry)
        _check({r["case"].split("/")[0] for r in details["cases"]} ==
               {"ReT060", "ReT100"}, "did not audit both Reynolds groups")
        details["status"] = "PASS"
    except Exception as exc:
        details["status"] = "FAIL"
        details["error"] = f"{type(exc).__name__}: {exc}"
        _write_json_atomic(args.output, details)
        raise
    _write_json_atomic(args.output, details)
    return details


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, default=Path("/scratch/ai/HIT_DNS_pilot"))
    parser.add_argument("--ms-sr-root", type=Path, default=Path("/scratch/ai/MS_SR"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs=2, help="One teacher case each from ReT060 and ReT100")
    parser.add_argument("--chunk-cells", type=int, default=65536)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-3)
    parser.add_argument("--skip-file-hashes", action="store_true", help="Faster but weaker; report explicitly records skipped hashes")
    args = parser.parse_args(argv)
    result = run(args)
    print(json.dumps({"status": result["status"], "output": str(args.output),
                      "cases": [(x["case"], x["snapshot_id"]) for x in result["cases"]]},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
