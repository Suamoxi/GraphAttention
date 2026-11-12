"""M46: all-snapshot, read-only cross-case HIT similarity audit.

Inventory is metadata-only. Analyze requires explicit M3.3 case reference YAMLs;
source directory labels never supply physical reference values. Analysis never
applies a stationarity mask, reweights by node count, or changes raw data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

import h5py
import numpy as np

from graph_attention.data.avbp import AVBP_FIELD_CATALOG
from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.multicase_avbp import (
    DEFAULT_M43_CASES, _inspect_mesh, _inspect_snapshot,
)
from graph_attention.data.nondimensionalization import ConvectiveNondimensionalizer

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
PATHS = {spec.name: spec.source_path for spec in AVBP_FIELD_CATALOG.require(FIELDS)}
PILOT_SIDES = {"ReT060": 193, "ReT100": 321}
CASE_PATTERN = re.compile(r"^Mt\d+(?:\.\d+)?_Xi\d+(?:\.\d+)?$")
CHUNK = 262_144


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def _rows(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _candidates(archive: Path, pilot: Path):
    for spec in DEFAULT_M43_CASES:
        yield spec.case_id, spec.side, archive / spec.case_id, "legacy_archive"
    for root, label in ((archive, "reT_archive"), (pilot / "data", "reT_pilot")):
        for reynolds, side in PILOT_SIDES.items():
            parent = root / reynolds
            if not parent.is_dir():
                if label == "reT_pilot":
                    raise FileNotFoundError(f"pilot group not found: {parent}")
                continue
            for directory in sorted(parent.iterdir()):
                if directory.is_dir() and CASE_PATTERN.fullmatch(directory.name):
                    yield f"{reynolds}/{directory.name}", side, directory, label


def _m46_snapshots(directory: Path) -> tuple[tuple[int, Path], ...]:
    """Inventory every numbered HDF5 path; M43's strict unique-counter rule is not used.

    Two distinct files in the same source may have the same numeric suffix.
    Both are retained, with the collision listed in inventory.json.
    """
    groups = []
    for location in (directory / "SOLUT", directory / "RUN" / "SOLUT"):
        if location.is_dir():
            entries = tuple(p.absolute() for p in location.glob("*.h5") if p.is_file())
            if entries:
                groups.append(entries)
    if len(groups) != 1:
        raise ValueError(
            f"{directory}: expected one nonempty SOLUT directory; found {len(groups)}"
        )
    numbered = []
    for path in groups[0]:
        match = re.fullmatch(r"(.+?)(\\d+)", path.stem)
        if match is None:
            raise ValueError(f"{directory}: snapshot {path.name!r} has no numeric suffix")
        numbered.append((int(match.group(2)), path))
    return tuple(sorted(numbered, key=lambda pair: (pair[0], pair[1].name)))


def inventory(archive: Path, pilot: Path) -> dict:
    """Retain same-source counter collisions; prefer archive across sources.

    Same-inode aliases are the same physical file and appear once. Distinct
    files sharing a counter *within a source* all enter the empirical dataset.
    Across archive/pilot sources, a matching (case_id, counter) uses archive
    priority, with rejected paths recorded; no byte equality is implied.
    """
    samples: dict[str, dict] = {}
    cases: dict[str, dict] = {}
    duplicates: list[dict] = []
    seen_inodes: dict[tuple[str, str, int, int], dict] = {}
    by_counter: dict[tuple[str, int], list[dict]] = {}
    for case_id, side, directory, source in _candidates(archive, pilot):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        mesh = directory / "MESH" / "mesh.mesh.h5"
        dimensions = _inspect_mesh(mesh, side)
        paths = _m46_snapshots(directory)
        for position in sorted({0, len(paths) // 2, len(paths) - 1}):
            _inspect_snapshot(paths[position][1], dimensions["nodes"])
        case = cases.setdefault(case_id, {
            "case_id": case_id, "side": side, "sources": [], "snapshots": 0,
        })
        if case["side"] != side:
            raise ValueError(f"inconsistent mesh dimensions for {case_id}")
        case["sources"].append({
            "name": source, "path": str(directory), "count": len(paths),
        })
        for counter, snapshot in paths:
            sample_id = f"{case_id}:{snapshot.stem}"
            file_stat = snapshot.stat()
            inode_key = (case_id, source, file_stat.st_dev, file_stat.st_ino)
            if inode_key in seen_inodes:
                duplicates.append({
                    "sample_id": sample_id,
                    "selected": seen_inodes[inode_key]["snapshot"],
                    "discarded": str(snapshot),
                    "policy": "same_filesystem_object_alias",
                })
                continue
            # Archive cases are processed before their pilot counterparts.
            counter_key = (case_id, counter)
            selected = by_counter.get(counter_key, [])
            if selected and any(item["source"] != source for item in selected):
                duplicates.append({
                    "sample_id": sample_id,
                    "selected": selected[0]["snapshot"],
                    "selected_paths": [item["snapshot"] for item in selected],
                    "discarded": str(snapshot),
                    "policy": "archive_preferred_by_case_and_numeric_counter_not_byte_identity",
                })
                continue
            if sample_id in samples:
                raise ValueError(
                    f"{directory}: duplicate sample identifier {sample_id}; "
                    "source naming must uniquely identify each distinct file"
                )
            item = {
                "sample_id": sample_id, "case_id": case_id, "source": source,
                "snapshot": str(snapshot), "mesh": str(mesh.resolve(strict=True)),
                "side": side, "counter": counter,
            }
            samples[sample_id] = item
            seen_inodes[inode_key] = item
            by_counter.setdefault(counter_key, []).append(item)
            case["snapshots"] += 1
    collisions = []
    for (case_id, counter), entries in sorted(by_counter.items()):
        if len(entries) > 1:
            collisions.append({
                "case_id": case_id, "source": entries[0]["source"],
                "counter": counter,
                "retained_paths": [item["snapshot"] for item in entries],
            })
    ordered = sorted(
        samples.values(),
        key=lambda item: (item["case_id"], item["counter"], item["snapshot"]),
    )
    return {
        "version": "m46_all_snapshots_v2", "stationarity_filter": False,
        "source_roots": {"archive": str(archive), "pilot": str(pilot)},
        "deduplication": (
            "same-inode aliases excluded; within-source counter collisions retained; "
            "archive preferred over pilot for matching case ID and numeric counter"
        ),
        "cases": sorted(cases.values(), key=lambda c: c["case_id"]),
        "samples": ordered, "duplicates": duplicates,
        "within_source_counter_collisions": collisions,
        "counts": {
            "cases": len(cases), "snapshots": len(ordered),
            "duplicate_paths": len(duplicates),
            "within_source_counter_collisions": len(collisions),
        },
    }


def _mesh_index(mesh_path: Path, side: int, chunk: int = CHUNK) -> tuple[np.ndarray, np.ndarray]:
    """Validate and index the unique (side-1)^3 periodic lattice, independent of node IDs."""
    count = side**3
    unique_side = side - 1
    occupied = np.zeros(count, dtype=np.bool_)
    unique_to_native = np.full(unique_side**3, -1, dtype=np.int32)
    interior_native_mask = np.zeros(count, dtype=np.bool_)
    native_to_unique = np.full(count, -1, dtype=np.int32)
    with h5py.File(mesh_path, "r") as mesh:
        datasets = [mesh[f"Coordinates/{axis}"] for axis in "xyz"]
        if any(dataset.shape not in ((count,), (count, 1)) for dataset in datasets):
            raise ValueError(f"{mesh_path}: coordinate size differs from {side}^3")
        lower = np.full(3, np.inf)
        upper = np.full(3, -np.inf)
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            xyz = np.column_stack([np.asarray(ds[start:stop]).reshape(-1) for ds in datasets])
            if not np.isfinite(xyz).all():
                raise ValueError(f"{mesh_path}: nonfinite coordinates")
            lower = np.minimum(lower, xyz.min(axis=0))
            upper = np.maximum(upper, xyz.max(axis=0))
        dx = (upper - lower) / unique_side
        if not np.all(dx > 0):
            raise ValueError("degenerate Cartesian grid")
        if not np.allclose(dx, dx[0], rtol=1e-4, atol=0):
            raise ValueError("non-cubic Cartesian grid requires separate spectrum convention")
        for start in range(0, count, chunk):
            stop = min(start + chunk, count)
            xyz = np.column_stack([np.asarray(ds[start:stop]).reshape(-1) for ds in datasets])
            floating = (xyz - lower) / dx
            index = np.rint(floating).astype(np.int32)
            if (np.any(np.abs(floating - index) > 1e-3) or
                    np.any(index < 0) or np.any(index >= side)):
                raise ValueError(f"{mesh_path}: mesh is not a complete Cartesian lattice")
            linear = (index[:, 0].astype(np.int64) * side + index[:, 1]) * side + index[:, 2]
            if len(np.unique(linear)) != len(linear) or np.any(occupied[linear]):
                raise ValueError(f"{mesh_path}: duplicate coordinate lattice point")
            occupied[linear] = True
            interior = np.all(index < unique_side, axis=1)
            interior_native_mask[start:stop] = interior
            i = index[interior]
            position = (i[:, 0].astype(np.int64) * unique_side + i[:, 1]) * unique_side + i[:, 2]
            unique_to_native[position] = np.arange(start, stop, dtype=np.int32)[interior]
            native_to_unique[np.arange(start, stop, dtype=np.int32)[interior]] = position.astype(np.int32)
    if not occupied.all() or np.any(unique_to_native < 0):
        raise ValueError(f"{mesh_path}: missing points from the Cartesian lattice")
    return unique_to_native, interior_native_mask, native_to_unique, float(upper[0] - lower[0])


class Moments:
    """Merge centered moments of bounded chunks (Pébay), without a second I/O pass."""

    def __init__(self):
        self.n = 0
        self.mean = self.m2 = self.m3 = self.m4 = 0.0

    def add(self, values: np.ndarray) -> None:
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        b = x.size
        if not b:
            return
        mean_b = float(x.mean())
        dx = x - mean_b
        m2_b = float(np.sum(dx**2))
        m3_b = float(np.sum(dx**3))
        m4_b = float(np.sum(dx**4))
        if self.n == 0:
            self.n, self.mean, self.m2, self.m3, self.m4 = b, mean_b, m2_b, m3_b, m4_b
            return
        a, n = self.n, self.n + b
        delta = mean_b - self.mean
        m2 = self.m2 + m2_b + delta**2 * a * b / n
        m3 = (self.m3 + m3_b + delta**3 * a * b * (a - b) / n**2
              + 3 * delta * (a * m2_b - b * self.m2) / n)
        m4 = (self.m4 + m4_b + delta**4 * a * b * (a*a - a*b + b*b) / n**3
              + 6 * delta**2 * (a*a * m2_b + b*b * self.m2) / n**2
              + 4 * delta * (a * m3_b - b * self.m3) / n)
        self.mean += delta * b / n
        self.n, self.m2, self.m3, self.m4 = n, m2, m3, m4

    def record(self) -> dict:
        variance = self.m2 / self.n
        return {
            "mean": self.mean, "std": math.sqrt(max(0.0, variance)),
            "skew": self.m3 / (self.n * variance**1.5) if variance > 0 else None,
            "excess_kurtosis": self.m4 / (self.n * variance**2) - 3 if variance > 0 else None,
        }


def _spectral(velocity: np.ndarray) -> dict:
    """FFT of dimensionless u'/Uref: full-cube Helmholtz energy, inscribed-shell spectrum.

    Dimensionless shell spacing is 2*pi in k*L_box; spectrum integral returns
    kinetic energy per reference U^2. Hermitian weights account for rfftn.
    """
    n = velocity.shape[0]
    if velocity.shape != (n, n, n, 3):
        raise ValueError("FFT requires [n,n,n,3] velocity")
    transforms = [np.fft.rfftn(velocity[..., i], norm="forward") for i in range(3)]
    for f in transforms:
        f[0, 0, 0] = 0
    kx = np.fft.fftfreq(n) * n
    ky = np.fft.fftfreq(n) * n
    kz = np.fft.rfftfreq(n) * n
    kxy2 = kx[:, None]**2 + ky[None, :]**2
    shells = np.zeros(n // 2 + 1, dtype=np.float64)
    total_energy = dilatational_energy = 0.0
    for iz, kz_value in enumerate(kz):
        radius = np.sqrt(kxy2 + kz_value**2)
        weight = 1.0 if iz == 0 or (n % 2 == 0 and iz == n // 2) else 2.0
        fx, fy, fz = (f[:, :, iz] for f in transforms)
        power = 0.5 * weight * (np.abs(fx)**2 + np.abs(fy)**2 + np.abs(fz)**2)
        projection = kx[:, None] * fx + ky[None, :] * fy + kz_value * fz
        dilat = np.divide(0.5 * weight * np.abs(projection)**2, radius**2,
                           out=np.zeros_like(power), where=radius > 0)
        total_energy += float(power.sum())
        dilatational_energy += float(dilat.sum())
        shell_index = np.rint(radius).astype(np.int32)
        include = (radius > 0) & (shell_index <= n // 2)
        shells += np.bincount(shell_index[include], weights=power[include],
                              minlength=shells.size)[:shells.size]
    return {
        "kL_box": (2 * np.pi * np.arange(1, n // 2 + 1)).tolist(),
        "E_star": (shells[1:] / (2 * np.pi)).tolist(),
        "TKE_spectral_star": total_energy,
        "dilatational_fraction": dilatational_energy / total_energy if total_energy > 0 else None,
        "spectrum_energy_capture": float(shells.sum() / total_energy) if total_energy > 0 else None,
    }


def _analyze_sample(sample: dict, reference, mesh_cache: dict,
                    samples_per_snapshot: int, spectra: bool, seed: int) -> tuple[dict, dict]:
    case_id = sample["case_id"]
    side = int(sample["side"])
    mesh_key = sample["mesh"]
    if mesh_key not in mesh_cache:
        mesh_cache.clear()  # O(one mesh): essential for 320^3 Cartesian indices
        mesh_cache[mesh_key] = _mesh_index(Path(mesh_key), side)
    to_native, interior_mask, native_to_unique, box_length = mesh_cache[mesh_key]
    definition = load_case_definition(reference)
    normalizer = ConvectiveNondimensionalizer(definition.reference_scales)
    scales = {field: normalizer.field_scale(field) for field in FIELDS}
    l_ref = normalizer.coordinate_scale()
    if not math.isclose(l_ref, box_length, rel_tol=1e-4, abs_tol=0):
        raise ValueError(f"{case_id}: L_ref must be the physical periodic box side for kL comparisons")
    u_ref = definition.reference_scales["U_ref"].value
    rho_ref = definition.reference_scales["rho_ref"].value
    a_ref = (definition.reference_scales["a_ref"].value
             if "a_ref" in definition.reference_scales.names else None)
    if u_ref <= 0 or rho_ref <= 0:
        raise ValueError("positive explicit rho_ref and U_ref required")
    if samples_per_snapshot < 1:
        raise ValueError("samples-per-snapshot must be positive")
    # Equal-size, spatially selected draws per snapshot, independent of native node IDs.
    rng = np.random.default_rng(int(hashlib.sha256(
        f"{seed}:{sample['sample_id']}".encode()).hexdigest()[:16], 16))
    pick = rng.choice(to_native.size, size=min(samples_per_snapshot, to_native.size), replace=False)
    native_pick = np.sort(to_native[pick].astype(np.int64))
    quantile_samples = {}
    field_moments = {name: Moments() for name in FIELDS}
    vel_moments = [Moments() for _ in range(3)]
    vel_cube = np.empty((side-1, side-1, side-1, 3), dtype=np.float32) if spectra else None
    with h5py.File(sample["snapshot"], "r") as data:
        for name in FIELDS:
            if PATHS[name] not in data:
                raise KeyError(f"missing field {PATHS[name]}: {sample['snapshot']}")
            ds = data[PATHS[name]]
            if ds.shape not in ((side**3,), (side**3, 1)):
                raise ValueError(f"unexpected field shape for {name}: {ds.shape}")
            quantile_samples[name] = np.asarray(ds[native_pick] if ds.ndim == 1
                else ds[native_pick, 0], dtype=np.float64) / scales[name]
        for start in range(0, side**3, CHUNK):
            stop = min(start + CHUNK, side**3)
            selection = interior_mask[start:stop]
            if not np.any(selection):
                continue
            raw = {}
            for name in FIELDS:
                ds = data[PATHS[name]]
                raw[name] = np.asarray(ds[start:stop], dtype=np.float64).reshape(-1)[selection]
                if not np.isfinite(raw[name]).all():
                    raise ValueError(f"nonfinite {name}: {sample['snapshot']}")
                field_moments[name].add(raw[name] / scales[name])
            if np.any(raw["rho"] <= 0):
                raise ValueError(f"nonpositive density: {sample['snapshot']}")
            velocity = [raw[name] / raw["rho"] / u_ref for name in FIELDS[1:4]]
            for i, component in enumerate(velocity):
                vel_moments[i].add(component)
            if vel_cube is not None:
                # Native nodes are arbitrary: place velocity by its physical lattice index.
                native = np.flatnonzero(selection).astype(np.int64) + start
                # Native node numbering is irrelevant to the Cartesian FFT order.
                indices = native_to_unique[native]
                for axis, component in enumerate(velocity):
                    vel_cube.reshape(-1, 3)[indices, axis] = component
    if any(m.n != to_native.size for m in field_moments.values()):
        raise RuntimeError("field statistic node count differs from unique grid node count")
    center = np.array([m.mean for m in vel_moments])
    tke = 0.5 * sum(m.m2 / m.n for m in vel_moments)
    record = {
        "sample_id": sample["sample_id"], "case_id": case_id,
        "source": sample["source"], "snapshot": sample["snapshot"],
        "node_count_unique": int(to_native.size),
        "tke_star": tke,
        "velocity_rms_1d_star": math.sqrt(2 * tke / 3),
        "mean_velocity_star": center.tolist(),
        "mach_t_reference_sound_speed": (math.sqrt(2 * tke) * u_ref / a_ref
                                            if a_ref is not None else None),
        "fields": {name: m.record() for name, m in field_moments.items()},
        "reference_file": str(reference), "reference_sha256": _sha256(reference),
        "reference_scheme": definition.reference_scales.scheme,
        "mesh_file_size": Path(mesh_key).stat().st_size,
        "mesh_mtime_ns": Path(mesh_key).stat().st_mtime_ns,
    }
    record["analysis_options"] = {"draws": samples_per_snapshot, "spectra": spectra, "seed": seed}
    record["snapshot_file_size"] = Path(sample["snapshot"]).stat().st_size
    record["snapshot_mtime_ns"] = Path(sample["snapshot"]).stat().st_mtime_ns
    if spectra:
        record["spectral"] = _spectral(vel_cube)
        if not math.isclose(record["spectral"]["TKE_spectral_star"], tke,
                            rel_tol=2e-5, abs_tol=1e-7):
            raise ValueError(f"Fourier Parseval check failed for {sample['sample_id']}")
    return record, quantile_samples


def _combine(samples: list[dict], root: Path, output: Path, draws: int) -> None:
    """Produce equal-case comparisons from per-snapshot cached results."""
    grouped = {}
    for item in samples:
        case = item["case_id"]
        stem = item["sample_id"].split(":", 1)[1]
        stem_path = output / "per_snapshot" / case / stem
        with np.load(stem_path.with_suffix(".npz")) as cache:
            record = json.loads(str(cache["metadata"]))
            grouped.setdefault(case, []).append((record, {name: cache[name].copy()
                                                          for name in FIELDS}))
    case_rows, spectra_rows, distributions = [], [], {}
    for case, entries in sorted(grouped.items()):
        values_by_channel = {}
        for name in FIELDS:
            values = np.concatenate([draw[name] for _, draw in entries])
            values_by_channel[name] = np.sort(values)
            means = np.array([record["fields"][name]["mean"] for record, _ in entries])
            vars_ = np.array([record["fields"][name]["std"]**2 for record, _ in entries])
            pooled_std = math.sqrt(max(0.0, float(np.mean(vars_ + means**2) - np.mean(means)**2)))
            case_rows.append({"case_id": case, "snapshots": len(entries), "field": name,
                              "mean_star": float(np.mean(means)), "std_star": pooled_std,
                              "mean_within_snapshot_std": float(np.mean(np.sqrt(vars_))),
                              "draws_per_snapshot": min(draws, entries[0][0]["node_count_unique"])})
        distributions[case] = values_by_channel
        for rec, _ in entries:
            sp = rec.get("spectral")
            if sp is not None:
                for k, energy in zip(sp["kL_box"], sp["E_star"]):
                    spectra_rows.append({"case_id": case, "sample_id": rec["sample_id"],
                                         "kL_box": k, "E_star": energy})
    _rows(output / "case_field_statistics.csv", case_rows,
          ("case_id", "snapshots", "field", "mean_star", "std_star",
           "mean_within_snapshot_std", "draws_per_snapshot"))
    _rows(output / "snapshot_spectra.csv", spectra_rows,
          ("case_id", "sample_id", "kL_box", "E_star"))
    # Comparable empirical PDFs on identical dimensionless field bins.
    pdf_rows = []
    for field in FIELDS:
        minimum = min(float(distributions[c][field][0]) for c in distributions)
        maximum = max(float(distributions[c][field][-1]) for c in distributions)
        if minimum == maximum:
            maximum = minimum + 1.0
        bins = np.linspace(minimum, maximum, 257)
        for case in sorted(distributions):
            freq, _ = np.histogram(distributions[case][field], bins=bins, density=True)
            for low, high, density in zip(bins[:-1], bins[1:], freq):
                pdf_rows.append({"case_id": case, "field": field, "value_low_star": low,
                                 "value_high_star": high, "pdf_star_inverse": density})
    _rows(output / "case_field_pdfs.csv", pdf_rows,
          ("case_id", "field", "value_low_star", "value_high_star", "pdf_star_inverse"))
    by_case_field = {(r["case_id"], r["field"]): r for r in case_rows}
    cases = sorted(distributions)
    grid = (np.arange(1024, dtype=np.float64) + 0.5) / 1024
    quantiles = {case: {field: np.quantile(distributions[case][field], grid)
                        for field in FIELDS} for case in cases}
    comparisons = []
    for i, left in enumerate(cases):
        for right in cases[i+1:]:
            for field in FIELDS:
                qdiff = np.abs(quantiles[left][field] - quantiles[right][field])
                w1 = float(np.mean(qdiff))
                a = by_case_field[left, field]["std_star"]
                b = by_case_field[right, field]["std_star"]
                pooled = math.hypot(a, b) / math.sqrt(2)
                comparisons.append({"case_a": left, "case_b": right, "field": field,
                                    "w1_star_quantile_estimate": w1,
                                    "w1_over_pooled_std": w1 / pooled if pooled > 0 else None})
    _rows(output / "case_pairwise_wasserstein.csv", comparisons,
          ("case_a", "case_b", "field", "w1_star_quantile_estimate",
           "w1_over_pooled_std"))
    _atomic_json(output / "analysis_summary.json", {
        "status": "complete", "cases": len(cases), "samples": len(samples),
        "weighting": "equal snapshot draw count within case; pairwise comparisons give each case one distribution",
        "statistics": "full unique-periodic-node moments, approximate equal-probability quantile W1",
        "w1_approximation": "1024 equal-probability quantile midpoints on sampled interior nodes",
        "stationarity_filter": False, "source_manifest": str(root),
        "references_required": True,
    })


def run_analyze(manifest: dict, reference_dir: Path, output: Path,
                draws: int, spectra: bool, seed: int) -> None:
    references = {}
    schemes = set()
    missing = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        path = reference_dir / (case_id.replace("/", "_") + ".yaml")
        if not path.is_file():
            missing.append(str(path))
        else:
            definition = load_case_definition(path)
            if definition.case_id != case_id:
                raise ValueError(f"{path}: declared case ID does not match {case_id}")
            schemes.add(definition.reference_scales.scheme)
            normalizer = ConvectiveNondimensionalizer(definition.reference_scales)
            for name in FIELDS:
                normalizer.field_scale(name)
            normalizer.coordinate_scale()
            references[case_id] = path
    if len(schemes) > 1:
        raise ValueError(f"cross-case reference schemes differ: {sorted(schemes)}")
    if missing:
        raise FileNotFoundError("Missing explicit M3.3 case references:\n" + "\n".join(missing))
    regime_rows = []
    for case_id, path in sorted(references.items()):
        definition = load_case_definition(path)
        for name in definition.regime_parameters.names:
            parameter = definition.regime_parameters[name]
            regime_rows.append({"case_id": case_id, "parameter": name,
                                "value": parameter.value, "definition": parameter.definition,
                                "provenance": parameter.provenance,
                                "inference_available": parameter.inference_available})
    _rows(output / "case_regime_parameters.csv", regime_rows,
          ("case_id", "parameter", "value", "definition", "provenance",
           "inference_available"))
    mesh_cache = {}
    for sample in manifest["samples"]:
        out = output / "per_snapshot" / sample["case_id"] / (sample["sample_id"].split(":", 1)[1] + ".npz")
        if out.exists():
            with np.load(out) as existing:
                stored = json.loads(str(existing["metadata"]))
            status = stored.get("analysis_options")
            source = Path(sample["snapshot"])
            if (status != {"draws": draws, "spectra": spectra, "seed": seed}
                    or stored.get("reference_sha256") != _sha256(references[sample["case_id"]])
                    or stored.get("snapshot_file_size") != source.stat().st_size
                    or stored.get("snapshot_mtime_ns") != source.stat().st_mtime_ns
                    or stored.get("mesh_file_size") != Path(sample["mesh"]).stat().st_size
                    or stored.get("mesh_mtime_ns") != Path(sample["mesh"]).stat().st_mtime_ns):
                raise ValueError(f"cached analysis has incompatible inputs; remove {out} to recompute")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        record, values = _analyze_sample(sample, references[sample["case_id"]],
                                         mesh_cache, draws, spectra, seed)
        tmp = out.with_name(out.name + ".tmp")
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, metadata=json.dumps(record, allow_nan=False),
                                **{field: arr.astype(np.float32) for field, arr in values.items()})
        tmp.replace(out)
        print(f"analyzed {sample['sample_id']}", flush=True)
    _combine(manifest["samples"], output / "inventory.json", output, draws)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "analyze"))
    parser.add_argument("--archive-root", type=Path, default=Path("/archive2/cfd-phlusim/HIT_DNS_COMP"))
    parser.add_argument("--pilot-root", type=Path, default=Path("/scratch/ai/HIT_DNS_pilot"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--draws-per-snapshot", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--no-spectra", action="store_true")
    args = parser.parse_args()
    if args.draws_per_snapshot < 1:
        parser.error("--draws-per-snapshot must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_file = args.output / "inventory.json"
    if args.mode == "inventory":
        inventory_data = inventory(args.archive_root, args.pilot_root)
        _atomic_json(manifest_file, inventory_data)
        print(json.dumps(inventory_data["counts"], indent=2))
        return
    if args.reference_dir is None:
        parser.error("analyze requires --reference-dir")
    if not manifest_file.is_file():
        parser.error(f"run inventory first: {manifest_file}")
    manifest = json.loads(manifest_file.read_text())
    run_analyze(manifest, args.reference_dir, args.output, args.draws_per_snapshot,
                not args.no_spectra, args.seed)


if __name__ == "__main__":
    main()