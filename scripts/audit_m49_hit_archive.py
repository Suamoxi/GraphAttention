"""Read-only first-pass inventory of HIT LES archive and current scratch run.

Discover unknown directory layouts; inspect representative HDF5 headers and
bounded field payloads. Directory names and counters are evidence, not physical
case definitions or proof of independent trajectories.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

import h5py
import numpy as np

from graph_attention.data.avbp import AVBP_FIELD_CATALOG
from graph_attention.utils.provenance import collect_runtime_provenance

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
FIELD_PATHS = {spec.name: spec.source_path for spec in AVBP_FIELD_CATALOG.require(FIELDS)}
MESH_PATHS = ("Coordinates/x", "Coordinates/y", "Coordinates/z", "Connectivity/hex->node")
COUNTER = re.compile(r"(\d+)$")
TIME_KEY = re.compile(r"(time|timestep|iteration|iter|step|cycle|dt)$", re.I)
HDF_SUFFIXES = {".h5", ".hdf5"}
META_SUFFIXES = {".params", ".yaml", ".yml", ".json", ".cfg", ".ini", ".in", ".nml"}
MAX_PAYLOAD = 64 * 1024**2
MAX_TEXT = 64 * 1024


def numeric_counter(path):
    match = COUNTER.search(path.stem)
    return int(match.group(1)) if match else None


def safe_value(value):
    arr = np.asarray(value)
    if arr.size > 16:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "omitted": True}
    return str(value)[:2048]


def inspect_hdf(path, paths):
    result = {"path": str(path), "status": "ok", "datasets": {}, "time_metadata": {}}
    try:
        with h5py.File(path, "r") as h:
            missing = []
            for name in paths:
                obj = h.get(name)
                if not isinstance(obj, h5py.Dataset):
                    missing.append(name)
                    continue
                result["datasets"][name] = {
                    "shape": list(obj.shape), "dtype": str(obj.dtype),
                    "bytes_uncompressed": int(obj.size * obj.dtype.itemsize),
                }
            result["missing_expected_datasets"] = missing
            def inspect(name, obj):
                if len(result["time_metadata"]) >= 100:
                    return
                for key, value in obj.attrs.items():
                    if TIME_KEY.search(key):
                        result["time_metadata"][f"{name}@{key}"] = safe_value(value)
                if isinstance(obj, h5py.Dataset) and obj.size <= 16 and TIME_KEY.search(name):
                    result["time_metadata"][name] = safe_value(obj[()])
            inspect("/", h)
            h.visititems(inspect)
            total = sum(x["bytes_uncompressed"] for x in result["datasets"].values())
            if not missing and total <= MAX_PAYLOAD:
                digest = hashlib.sha256()
                extents = {}
                for name in paths:
                    arr = np.ascontiguousarray(h[name][()])
                    digest.update(name.encode())
                    digest.update(arr.dtype.str.encode())
                    digest.update(str(arr.shape).encode())
                    digest.update(arr.tobytes())
                    if name.startswith("Coordinates/") and arr.size and np.isfinite(arr).all():
                        extents[name] = [float(arr.min()), float(arr.max())]
                result["payload_sha256"] = digest.hexdigest()
                if extents:
                    result["coordinate_extents_raw_units"] = extents
            else:
                result["payload_sha256"] = None
                result["hash_skipped_reason"] = "missing datasets" if missing else "payload exceeds 64 MiB"
    except Exception as exc:
        result.update(status="error", error=repr(exc))
    return result


def discover(root):
    if not root.is_dir():
        raise NotADirectoryError(root)
    result = {"root": str(root), "solutions": [], "meshes": [],
              "other_hdf": [], "setup_files": [], "skipped_directory_symlinks": [],
              "errors": []}
    def onerror(exc):
        result["errors"].append(str(exc))
    for directory, dirs, names in os.walk(root, followlinks=False, onerror=onerror):
        parent = Path(directory)
        dirs.sort()
        for name in dirs:
            if (parent / name).is_symlink():
                result["skipped_directory_symlinks"].append(str(parent / name))
        for name in sorted(names):
            path = parent / name
            suffix = path.suffix.lower()
            if suffix not in HDF_SUFFIXES | META_SUFFIXES and name.lower() not in {"run.params", "metadata", "readme", "readme.md"}:
                continue
            try:
                st = path.stat()
                record = {"path": str(path), "resolved_path": str(path.resolve()),
                          "bytes": st.st_size, "device": st.st_dev, "inode": st.st_ino,
                          "mtime_ns": st.st_mtime_ns}
                if suffix in HDF_SUFFIXES:
                    parts = {p.upper() for p in parent.relative_to(root).parts}
                    if "MESH" in parts or name.lower().endswith(".mesh.h5"):
                        result["meshes"].append(record)
                    elif "SOLUT" in parts or name.lower().startswith(("solut_", "last_solution")):
                        record.update(counter=numeric_counter(path), solution_directory=str(parent))
                        result["solutions"].append(record)
                    else:
                        result["other_hdf"].append(record)
                else:
                    with path.open("rb") as stream:
                        record["excerpt"] = stream.read(MAX_TEXT).decode("utf-8", errors="replace")
                    record["truncated"] = st.st_size > MAX_TEXT
                    result["setup_files"].append(record)
            except OSError as exc:
                result["errors"].append(f"{path}: {exc}")
    return result


def enrich(inventory):
    groups = defaultdict(list)
    for row in inventory["solutions"]:
        groups[row["solution_directory"]].append(row)
    meshes = {row["path"]: row for row in inventory["meshes"]}
    for row in meshes.values():
        row["inspection"] = inspect_hdf(Path(row["path"]), MESH_PATHS)
    streams = []
    for directory, rows in sorted(groups.items()):
        rows.sort(key=lambda r: (r["counter"] is None, r["counter"] or 0, r["path"]))
        numbered = [r for r in rows if r["counter"] is not None]
        chosen = []
        if numbered:
            chosen.extend(numbered[i] for i in sorted({0, len(numbered)//2, len(numbered)-1}))
        chosen.extend(r for r in rows if r["counter"] is None)
        for row in chosen:
            row["inspection"] = inspect_hdf(Path(row["path"]), tuple(FIELD_PATHS.values()))
        parent = Path(directory).parent
        case = parent.parent if parent.name.upper() == "RUN" else parent
        candidates = [name for name in meshes if Path(name).parent in (case/"MESH", case/"RUN"/"MESH", parent/"MESH")]
        counters = [r["counter"] for r in numbered]
        streams.append({"solution_directory": directory, "case_directory_candidate": str(case),
                        "count": len(rows), "numbered_count": len(numbered),
                        "first_counter": counters[0] if counters else None,
                        "last_counter": counters[-1] if counters else None,
                        "counter_gaps": sorted(set(b-a for a,b in zip(counters,counters[1:]))),
                        "mesh_candidates": candidates,
                        "representative_paths": [r["path"] for r in chosen]})
    inventory["streams"] = streams
    inventory["counts"] = {key: len(inventory[key]) for key in
                          ("solutions", "meshes", "other_hdf", "setup_files", "streams", "errors", "skipped_directory_symlinks")}
    return inventory


def compare(archive, scratch):
    """Only exact file identity or compared conservative arrays establish overlap."""
    by_name = defaultdict(list)
    for r in scratch["solutions"]:
        by_name[Path(r["path"]).name].append(r)
    matches = []
    alias_count = 0
    for a in archive["solutions"]:
        for b in by_name[Path(a["path"]).name]:
            same_inode = (a["device"], a["inode"]) == (b["device"], b["inode"])
            record = {"archive": a["path"], "scratch": b["path"],
                      "same_filesystem_object": same_inode, "same_conservative_payload": None}
            if same_inode:
                alias_count += 1
            if "inspection" in a:
                if "inspection" not in b:
                    b["inspection"] = inspect_hdf(Path(b["path"]), tuple(FIELD_PATHS.values()))
                ah, bh = a["inspection"].get("payload_sha256"), b["inspection"].get("payload_sha256")
                if ah is not None and bh is not None:
                    record["same_conservative_payload"] = ah == bh
            matches.append(record)
    return {"same_basename_candidates": len(matches), "same_inode_pairs": alias_count,
            "compared_payload_pairs": sum(m["same_conservative_payload"] is not None for m in matches),
            "identical_payload_pairs": sum(m["same_conservative_payload"] is True for m in matches),
            "matches": matches}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("/archivenfs/phlusim/HIT_LES_FORCED"))
    p.add_argument("--scratch", type=Path, default=Path("/scratch/coop/theret/HIT_LES_FORCED"))
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    archive = enrich(discover(args.archive))
    print("archive:", archive["counts"], flush=True)
    scratch = enrich(discover(args.scratch))
    print("scratch:", scratch["counts"], flush=True)
    overlap = compare(archive, scratch)
    result = {"provenance": collect_runtime_provenance(),
              "policy": "inventory, representative headers and bounded exact payload hashes; no stationarity or independence certification",
              "archive": archive, "scratch": scratch, "overlap": overlap}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    tmp.replace(args.output)
    print("overlap:", {k:v for k,v in overlap.items() if k != "matches"}, flush=True)


if __name__ == "__main__":
    main()
