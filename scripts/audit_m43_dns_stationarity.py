"""Read-only M43 DNS stationarity and unresolved-statistics experiment.

Run via python -m scripts.audit_m43_dns_stationarity.
Requires only numpy, h5py, matplotlib and existing GraphAttention modules.
No training checkpoints or dataset splits are read or changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_attention.data.multicase_avbp import DEFAULT_M43_CASES, _find_snapshots
from graph_attention.evaluation.dns_matching import matched_regime_comparison
from graph_attention.evaluation.dns_residual import (
    CONSERVATIVE_FIELDS,
    coarse_descriptors,
    radial_velocity_spectrum,
    spatial_residual_statistics,
    spectral_split,
    stationarity_candidates,
    velocity_from_conservative,
)
from graph_attention.evaluation.spectra_3d import (
    infer_cartesian_grid_3d,
    scatter_to_grid_3d,
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temporary")
    temp.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(k for r in rows for k in r))
    temp = path.with_name(path.name + ".temporary")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temporary.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, path)


def _sha_manifest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def _revision() -> str | None:
    p = subprocess.run(["git", "rev-parse", "HEAD"], check=False,
                       text=True, capture_output=True)
    return p.stdout.strip() if p.returncode == 0 else None


def _case_dir(root: Path, case_id: str) -> Path:
    if case_id not in {s.case_id for s in DEFAULT_M43_CASES}:
        raise ValueError("unknown declared HIT case " + repr(case_id))
    return root.joinpath(*case_id.split("/"))


def _snapshot_files(root: Path, case_id: str) -> tuple[Path, ...]:
    return _find_snapshots(_case_dir(root, case_id))


def _candidate_times(path: Path) -> dict[str, Any]:
    """Only explicit HDF5 timestamps, never inferred from filename counters."""
    found = []
    with h5py.File(path, "r") as h5:
        # Field-level HDF5 attributes are worth checking too: the source
        # preflight inspects group attributes, but actual solvers may attach
        # physical-time tags directly to solution datasets.
        metadata_objects = (
            "/", "GaseousPhase", "Metadata",
            *(f"GaseousPhase/{name}" for name in CONSERVATIVE_FIELDS),
        )
        for group_name in metadata_objects:
            if group_name != "/" and group_name not in h5:
                continue
            group = h5 if group_name == "/" else h5[group_name]
            for key in group.attrs:
                normalized = str(key).lower().replace("_", "").replace(" ", "")
                if normalized not in {"time", "physicaltime", "simulationtime", "simtime"}:
                    continue
                value = np.asarray(group.attrs[key])
                if value.size == 1 and np.issubdtype(value.dtype, np.number):
                    v = float(value.reshape(-1)[0])
                    if np.isfinite(v):
                        found.append({"path": group_name + "/@" + str(key), "value": v})
        for prefix in ("", "GaseousPhase/", "Metadata/"):
            for key in ("time", "Time", "physical_time", "PhysicalTime"):
                full_name = prefix + key
                if full_name in h5 and isinstance(h5[full_name], h5py.Dataset):
                    arr = h5[full_name]
                    if arr.size == 1 and np.issubdtype(arr.dtype, np.number):
                        found.append({
                            "path": "/" + full_name,
                            "value": float(arr[()]),
                        })
    times = [v["value"] for v in found]
    if not times:
        return {"status": "missing", "value": None, "sources": []}
    if not np.isfinite(times).all() or not np.allclose(
        times, times[0], rtol=1.e-9, atol=1.e-12
    ):
        return {"status": "ambiguous", "value": None, "sources": found}
    return {"status": "metadata_value_units_unverified",
            "value": float(times[0]), "sources": found}


def _run_params_save_schedule(directory: Path, files: tuple[Path, ...]) -> dict[str, Any]:
    """Read *declared* AVBP save cadence; do not infer HDF5 times.

    The derived axis is conditional on the declared save schedule being the
    schedule that actually produced these files, with no unrecorded restart.
    In particular t_i = i*dt assumes the first numbered output was written at
    t=dt from a simulation whose time origin was zero.
    """
    params = directory / "RUN" / "run.params"
    if not params.is_file():
        return {"status": "unavailable_no_run_params", "path": str(params)}
    raw = params.read_bytes()
    assignments: dict[str, str] = {}
    for raw_line in raw.decode("utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("!", "#")) or "=" not in line:
            continue
        key, value = line.split("=", 1)
        assignments[key.strip().lower()] = value.split("!", 1)[0].strip()
    required = ("simulation_end_time", "save_solution.time",
                "random_forcing.char_time")
    try:
        end, interval, forcing_time = (
            float(assignments[key].replace("D", "e").replace("d", "e"))
            for key in required
        )
    except (KeyError, ValueError):
        return {"status": "unavailable_missing_or_bad_params", "path": str(params)}
    if not (np.isfinite([end, interval, forcing_time]).all()
            and min(end, interval, forcing_time) > 0):
        return {"status": "unavailable_nonpositive_time", "path": str(params)}
    suffixes = [re.fullmatch(r"(.+?)(\d+)", path.stem) for path in files]
    consistent_names = (
        all(match is not None for match in suffixes)
        and len({match.group(1) for match in suffixes}) == 1
        and [int(match.group(2)) for match in suffixes]
        == list(range(1, len(files) + 1))
    )
    schedule_consistent = (
        consistent_names
        and np.isclose(end, len(files) * interval, rtol=1e-9, atol=0.0)
    )
    return {
        "status": ("configuration_consistent_origin_unverified" if schedule_consistent
                   else "configuration_inconsistent_do_not_infer"),
        "path": str(params),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "simulation_end_time_solver_units": end,
        "save_interval_solver_units": interval,
        "forcing_characteristic_time_solver_units": forcing_time,
        "end_time_over_save_interval": end / interval,
        "save_interval_over_forcing_characteristic_time": interval / forcing_time,
        "output_count": len(files),
        "filename_sequence_starts_at_one_and_is_contiguous": bool(consistent_names),
        "hdf5_time_verified": False,
        "time_origin_verified": False,
        "units_verified": False,
        "assumptions": ("Scheduled saves were produced by this run.params file, "
                        "with no interrupted output sequence, time reset, or "
                        "unrecorded restart. t_i=i*save_interval is only a candidate."),
    }


def _schedule_snapshot_time(schedule: dict[str, Any], index: int) -> dict[str, Any]:
    status = schedule["status"]
    if status != "configuration_consistent_origin_unverified":
        return {"status": status, "elapsed_from_first_output": None,
                "nominal_time_from_zero": None, "elapsed_in_forcing_times": None}
    step = schedule["save_interval_solver_units"]
    forcing_time = schedule["forcing_characteristic_time_solver_units"]
    return {
        "status": status,
        "elapsed_from_first_output": float(index * step),
        "nominal_time_from_zero": float((index + 1) * step),
        "elapsed_in_forcing_times": float(index * step / forcing_time),
        "nominal_time_in_forcing_times": float((index + 1) * step / forcing_time),
        "time_source": "RUN/run.params and consecutively numbered filenames",
        "time_origin_verified": False,
        "hdf5_time_verified": False,
    }


def _snapshot_descriptor(path: Path) -> dict[str, Any]:
    st = path.stat()
    return {"name": path.name, "path": str(path),
            "size": st.st_size, "mtime_ns": st.st_mtime_ns,
            "time": _candidate_times(path)}


def preflight(root: Path, output: Path, selected: list[str]) -> list[dict]:
    report = []
    for case in selected:
        directory = _case_dir(root, case)
        mesh = directory / "MESH" / "mesh.mesh.h5"
        files = _snapshot_files(root, case)
        with h5py.File(mesh, "r") as m:
            sizes = [int(m[f"Coordinates/{ax}"].size) for ax in "xyz"]
            conn = {"shape": list(m["Connectivity/hex->node"].shape),
                    "dtype": str(m["Connectivity/hex->node"].dtype)}
            attrs = sorted(str(k) for k in m.attrs)
        with h5py.File(files[0], "r") as h:
            fields = {
                name: {"shape": list(h[f"GaseousPhase/{name}"].shape),
                       "dtype": str(h[f"GaseousPhase/{name}"].dtype)}
                for name in CONSERVATIVE_FIELDS
            }
            root_attrs = sorted(str(k) for k in h.attrs)
            fluid_attrs = sorted(str(k) for k in h["GaseousPhase"].attrs)
        provenance = {}
        for relative in ("case_info.json", "RUN/run.params"):
            source = directory / relative
            if source.is_file():
                text_content = source.read_text(errors="replace")
                provenance[relative] = {
                    "path": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "time_related_lines_for_review": [
                        line.strip()[:300] for line in text_content.splitlines()
                        if any(token in line.lower()
                               for token in ("time", "timestep", "forcing", "viscos", "gamma"))
                    ][:30],
                }
            else:
                provenance[relative] = {"status": "missing", "path": str(source)}
        save_schedule = _run_params_save_schedule(directory, files)
        descriptors = [_snapshot_descriptor(p) for p in files]
        for index, descriptor in enumerate(descriptors):
            descriptor["scheduled_time_candidate"] = _schedule_snapshot_time(
                save_schedule, index
            )
        times = [s["time"]["value"] for s in descriptors]
        physical_time_known = all(t is not None for t in times)
        monotonic = bool(physical_time_known and np.all(np.diff(times) > 0))
        record = {
            "case_id": case,
            "snapshot_count": len(files),
            "run_params_save_schedule": save_schedule,
            "mesh": str(mesh),
            "mesh_coordinates_sizes": sizes,
            "mesh_connectivity": conn,
            "mesh_root_attribute_names": attrs,
            "snapshot_root_attribute_names": root_attrs,
            "snapshot_fluid_attribute_names": fluid_attrs,
            "fields": fields,
            "physical_provenance_text_files": provenance,
            "physical_time_metadata_monotonic": monotonic,
            "time_units_authoritative": False,
            "snapshots": descriptors,
            "note": "filename counters sort snapshots but do not give physical time",
        }
        report.append(record)
        print(f"{case}: {len(files)} snapshots, metadata time monotonic={monotonic}",
              flush=True)
    _write_json(output / "preflight.json", {"cases": report})
    return report


def _mesh_grid(mesh: Path):
    with h5py.File(mesh, "r") as h:
        coords = np.column_stack([
            np.asarray(h[f"Coordinates/{a}"]).reshape(-1) for a in "xyz"
        ])
    grid = infer_cartesian_grid_3d(coords, periodic_endpoint_mode="drop_max")
    if len(set(grid.shape)) != 1 or any(n % 8 for n in grid.shape):
        raise ValueError("audit requires an endpoint-inclusive periodic cubic grid "
                         "with unique side divisible by 8")
    return grid


def _read_field_grid(path: Path, grid) -> tuple[np.ndarray, dict[str, float]]:
    all_fields = []
    seam_errors: dict[str, float] = {}
    with h5py.File(path, "r") as h:
        for name in CONSERVATIVE_FIELDS:
            field = np.asarray(h[f"GaseousPhase/{name}"]).reshape(-1)
            # grid.linear_indices are *flat* Cartesian indices.
            # Indexing a 3-D ndarray with these values indexes axis 0 only.
            flat = np.empty(int(np.prod(grid.source_shape)), dtype=np.float64)
            flat[grid.linear_indices] = field
            full = flat.reshape(grid.source_shape)
            deviations = [
                np.max(np.abs(np.take(full, 0, axis=axis) -
                              np.take(full, -1, axis=axis)))
                for axis in range(3)
            ]
            discrepancy = float(max(deviations))
            seam_errors[name] = discrepancy
            scale = max(1.0, float(np.max(np.abs(full))))
            if discrepancy > max(1.e-8, 5.e-4 * scale):
                raise ValueError(f"{path.name}: {name} does not satisfy assumed "
                                 "periodic endpoint equivalence "
                                 f"(max_diff={discrepancy:.4g})")
            all_fields.append(scatter_to_grid_3d(field, grid))
    q = np.stack(all_fields, axis=-1)
    if np.any(q[..., 0] <= 0) or not np.isfinite(q).all():
        raise ValueError("invalid conservative fluid state")
    return q, seam_errors


def _base_metrics(q: np.ndarray, spacing: tuple[float, float, float]) -> dict:
    from graph_attention.evaluation.dns_residual import periodic_gradient
    u = velocity_from_conservative(q)
    mean = u.mean(axis=(0, 1, 2), keepdims=True)
    fluct = u - mean
    grad = periodic_gradient(u, spacing)
    curl = np.stack([
        grad[..., 2, 1] - grad[..., 1, 2],
        grad[..., 0, 2] - grad[..., 2, 0],
        grad[..., 1, 0] - grad[..., 0, 1],
    ], axis=-1)
    return {
        "mean_rho": float(np.mean(q[..., 0])),
        "std_rho": float(np.std(q[..., 0])),
        "u_rms_one_component": float(np.sqrt(np.mean(np.sum(fluct**2, axis=-1))/3)),
        "turbulent_k": float(0.5 * np.mean(np.sum(fluct**2, axis=-1))),
        "enstrophy": float(0.5 * np.mean(np.sum(curl**2, axis=-1))),
        "mean_temperature": None,
        "turbulent_mach": None,
        "taylor_reynolds": None,
        "dissipation": None,
        "unavailable": {
            "temperature": "no authoritative EOS/thermodynamic interpretation verified",
            "turbulent_mach": "requires verified sound-speed/EOS reference",
            "taylor_reynolds": "requires validated compressible definition and viscosity",
            "dissipation": "requires validated constitutive model and viscosity",
        },
    }


def _case_cache_root(output: Path, case: str) -> Path:
    return output / "cases" / case.replace("/", "_")


def _sample_snapshot(
    path: Path, grid, factors: tuple[int, ...], max_lag: int,
    cells_per_snapshot: int, seed: int, snapshot_index: int, case_output: Path,
    schedule: dict[str, Any] | None = None,
) -> dict[str, Any]:
    q, seams = _read_field_grid(path, grid)
    u = velocity_from_conservative(q)
    k, ek = radial_velocity_spectrum(u, grid.spacing)
    entry = {
        "snapshot_index": snapshot_index,
        "source_path": str(path),
        "source_stat": {"bytes": path.stat().st_size,
                        "mtime_ns": path.stat().st_mtime_ns},
        "hdf5_time": _candidate_times(path),
        "scheduled_time_candidate": _schedule_snapshot_time(
            schedule or {"status": "unavailable_no_run_params"}, snapshot_index
        ),
        "periodic_seam_max_abs": seams,
        "metrics": _base_metrics(q, grid.spacing),
        "filters": {},
    }
    with h5py.File(path, "r") as h5:
        for optional_name in ("temperature", "T"):
            key = "GaseousPhase/" + optional_name
            if key in h5 and h5[key].size == np.prod(grid.source_shape):
                measured = np.asarray(h5[key]).reshape(-1)
                if np.isfinite(measured).all():
                    entry["metrics"]["mean_temperature"] = float(np.mean(
                        scatter_to_grid_3d(measured, grid)
                    ))
                    entry["metrics"]["unavailable"].pop("temperature", None)
                    entry["metrics"]["temperature_source"] = key
                break
    spec_arrays = {"k": k, "E": ek}
    for factor in factors:
        feats, summary, residual_u = coarse_descriptors(q, factor, grid.spacing)
        low, high, orth = spectral_split(u, grid.spacing, factor)
        del low
        summary["spectral_unresolved_velocity_k"] = float(
            0.5 * np.mean(np.sum(high**2, axis=-1))
        )
        summary["spectral_parseval_relative_error"] = orth["parseval_relative_error"]
        summary["spectral_k_cutoff"] = orth["cutoff_rad_per_length"]
        kinetic_k = entry["metrics"]["turbulent_k"]
        summary["spectral_unresolved_energy_fraction"] = (
            summary["spectral_unresolved_velocity_k"] / kinetic_k
            if kinetic_k > 1.e-20 else None
        )
        # Not additive: residual velocity is referenced to Favre coarse velocity.
        summary["favre_residual_to_total_k_ratio_nonadditive"] = (
            summary["unresolved_velocity_k"] / kinetic_k
            if kinetic_k > 1.e-20 else None
        )
        rk, rE = radial_velocity_spectrum(residual_u, grid.spacing)
        spec_arrays[f"k_residual_f{factor}"] = rk
        spec_arrays[f"E_residual_f{factor}"] = rE
        lag = spatial_residual_statistics(residual_u, grid.spacing, max_lag)
        lag["r_over_delta"] = lag["r"] / (factor * float(np.mean(grid.spacing)))
        fp = case_output / "lag" / f"snapshot_{snapshot_index:03d}_f{factor}.npz"
        _save_npz(fp, lag)
        coarse_shape = feats["log_strain"].shape
        total = int(np.prod(coarse_shape))
        rng = np.random.default_rng(seed + snapshot_index * 10007 + factor)
        selected = rng.choice(total, size=min(total, cells_per_snapshot), replace=False)
        sample = {name: values.reshape(-1)[selected].astype(np.float64)
                  for name, values in feats.items()}
        sample["snapshot_index"] = np.full(len(selected), snapshot_index, dtype=np.int32)
        _save_npz(case_output / "cells" /
                  f"snapshot_{snapshot_index:03d}_f{factor}.npz", sample)
        entry["filters"][str(factor)] = summary
        entry["filters"][str(factor)]["num_coarse_cells"] = total
        entry["filters"][str(factor)]["sampled_coarse_cells"] = len(selected)
    _save_npz(case_output / "spectra" / f"snapshot_{snapshot_index:03d}.npz",
              spec_arrays)
    return entry


def _load_cutoffs(path: Path | None) -> dict:
    if path is None:
        return {}
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("reviewed cutoff JSON must map case IDs to index bounds")
    return value


def _read_cell_samples(
    case_output: Path, factor: int, indices: list[int]
) -> dict[str, np.ndarray]:
    collected: dict[str, list[np.ndarray]] = {}
    for index in indices:
        filename = case_output / "cells" / f"snapshot_{index:03d}_f{factor}.npz"
        with np.load(filename) as data:
            for name in data.files:
                collected.setdefault(name, []).append(data[name])
    return {key: np.concatenate(parts) for key, parts in collected.items()}


def _plot_case(
    case_output: Path, case: str, records: list[dict], factors: tuple[int, ...]
) -> None:
    figures = case_output / "figures"
    figures.mkdir(exist_ok=True)
    timeline = [row.get("scheduled_time_candidate", {}) for row in records]
    if all(
        row.get("status") == "configuration_consistent_origin_unverified"
        for row in timeline
    ):
        index = np.asarray(
            [row["elapsed_in_forcing_times"] for row in timeline], dtype=float
        )
        axis_label = (
            "Elapsed from first output / forcing char. time "
            "(configuration-derived, unverified)"
        )
    else:
        index = np.array([row["snapshot_index"] for row in records])
        axis_label = "Snapshot index (physical time unverified)"
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (title, key) in zip(axes.ravel(), (
        ("Turbulent kinetic energy", "turbulent_k"),
        ("Velocity RMS (one component)", "u_rms_one_component"),
        ("Density RMS", "std_rho"),
        ("Enstrophy", "enstrophy"),
    )):
        ax.plot(index, [row["metrics"][key] for row in records], lw=2)
        ax.set_xlabel(axis_label)
        ax.set_ylabel(title)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.25)
    fig.suptitle(case + " — stationarity diagnostics", fontsize=15)
    fig.savefig(figures / "time_history.pdf")
    plt.close(fig)

    chosen = [0, len(records)//2, len(records)-1]
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for index_value in chosen:
        with np.load(case_output / "spectra" /
                     f"snapshot_{records[index_value]['snapshot_index']:03d}.npz") as values:
            ax.loglog(values["k"], values["E"], lw=2,
                      label=f"snapshot {records[index_value]['snapshot_index']}")
    for factor in factors:
        cutoff = records[0]["filters"][str(factor)]["spectral_k_cutoff"]
        ax.axvline(cutoff, linestyle="--", linewidth=1.5, alpha=0.65,
                   label=f"filter f={factor} cutoff")
    nyquist = 2.0 * records[0]["filters"][str(2)]["spectral_k_cutoff"] \
        if 2 in factors else min(
            records[0]["filters"][str(f)]["spectral_k_cutoff"] * f
            for f in factors
        )
    ax.axvline(nyquist, color="black", linestyle=":", linewidth=2,
               label="axis Nyquist")
    ax.set_xlabel("Physical wavenumber k [rad / length]", fontsize=13)
    ax.set_ylabel("Velocity kinetic-energy spectrum E(k)", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / "early_middle_late_spectra.pdf")
    plt.close(fig)

    for factor in factors:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for index_value in chosen:
            with np.load(case_output / "spectra" /
                         f"snapshot_{records[index_value]['snapshot_index']:03d}.npz") as values:
                ax.loglog(values[f"k_residual_f{factor}"],
                          values[f"E_residual_f{factor}"], linewidth=2,
                          label=f"snapshot {records[index_value]['snapshot_index']}")
        ax.set_xlabel("Physical wavenumber k [rad / length]", fontsize=13)
        ax.set_ylabel("Residual velocity spectrum E_delta(k)", fontsize=13)
        ax.tick_params(labelsize=11)
        ax.legend()
        ax.grid(alpha=0.2)
        fig.savefig(figures / f"residual_spectra_factor_{factor}.pdf")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for factor in factors:
        ax.plot(index, [r["filters"][str(factor)]["unresolved_velocity_k"]
                        for r in records], lw=2, label=f"discrete coarse factor {factor}")
    ax.set_xlabel(axis_label, fontsize=13)
    ax.set_ylabel("Favre-relative unresolved velocity energy", fontsize=13)
    ax.tick_params(labelsize=11)
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(figures / "unresolved_energy_history.pdf")
    plt.close(fig)


def _postprocess_case(
    output: Path, case: str, factors: tuple[int, ...], cutoffs: dict,
    bootstrap: int, seed: int
) -> dict:
    base = _case_cache_root(output, case)
    records = [json.loads(path.read_text())
               for path in sorted((base / "snapshots").glob("snapshot_*.json"))]
    if not records:
        raise RuntimeError("no finished snapshots to summarize for " + case)
    _write_csv(base / "time_history.csv", [
        {"snapshot_index": r["snapshot_index"],
         "time_metadata_value": r["hdf5_time"]["value"],
         "schedule_elapsed_from_first_output": r.get(
             "scheduled_time_candidate", {}
         ).get("elapsed_from_first_output"),
         "schedule_nominal_time_from_zero": r.get(
             "scheduled_time_candidate", {}
         ).get("nominal_time_from_zero"),
         "schedule_elapsed_in_forcing_times": r.get(
             "scheduled_time_candidate", {}
         ).get("elapsed_in_forcing_times"),
         **{k: v for k, v in r["metrics"].items()
                                     if not isinstance(v, dict)},
         **{f"f{f}_{key}": value
            for f in factors for key, value in r["filters"][str(f)].items()
            if isinstance(value, (int, float))}
         } for r in records
    ])
    _plot_case(base, case, records, factors)
    candidate_keys = ("turbulent_k", "enstrophy", "u_rms_one_component")
    signals = np.array([[r["metrics"][key] for key in candidate_keys]
                        for r in records], dtype=float)
    candidate = stationarity_candidates(signals, window=min(12, max(4,len(records)//4)))
    metric_corr = {}
    for metric in ("turbulent_k", "enstrophy", "u_rms_one_component"):
        values = np.array([r["metrics"][metric] for r in records], dtype=float)
        centered = values - values.mean()
        denom = float(centered @ centered)
        metric_corr[metric] = [
            float(centered[:-lag] @ centered[lag:] / denom)
            if denom > 0 else None
            for lag in range(1, min(11, len(centered)//2))
        ]
    manifest = json.loads((base / "source_signature.json").read_text())
    report = {
        "case_id": case, "snapshots_completed": len(records),
        "run_params_save_schedule": manifest.get("run_params_save_schedule", {}),
        "time_axis_interpretation": (
            "nominal save-cadence elapsed time, not HDF5 verified time; "
            "forcing char. time is not the large-eddy turnover time"
        ),
        "candidate": candidate.__dict__,
        "index_lag_autocorrelation": metric_corr,
        "autocorrelation_warning": (
            "snapshot-index autocorrelation only; physical time and statistical "
            "independence must be verified before interpreting bootstrap intervals"
        ),
        "important": "stationarity onset is NOT auto-selected; review spectra and time histories",
        "comparison": {},
    }
    if case not in cutoffs:
        report["comparison_status"] = "awaiting_reviewed_cutoff"
    else:
        bounds = cutoffs[case]
        stop = int(bounds["transient_end_index"])
        start = int(bounds["stationary_start_index"])
        all_indices = sorted(r["snapshot_index"] for r in records)
        if not (0 <= stop < start <= max(all_indices)):
            raise ValueError(f"{case}: cutoff indices invalid")
        transient = [x for x in all_indices if x <= stop]
        stationary = [x for x in all_indices if x >= start]
        if len(transient) < 2 or len(stationary) < 2:
            raise ValueError("at least two snapshots in each reviewed regime required")
        transient_records = [r for r in records if r["snapshot_index"] in transient]
        stationary_records = [r for r in records if r["snapshot_index"] in stationary]
        report["unconditional"] = {}
        for name, get_value in {
            "turbulent_k": lambda r: r["metrics"]["turbulent_k"],
            "u_rms": lambda r: r["metrics"]["u_rms_one_component"],
            "enstrophy": lambda r: r["metrics"]["enstrophy"],
            **{
                f"unresolved_velocity_k_factor_{f}":
                (lambda r, fac=f: r["filters"][str(fac)]["unresolved_velocity_k"])
                for f in factors
            },
            **{
                f"spectral_unresolved_k_factor_{f}":
                (lambda r, fac=f: r["filters"][str(fac)]["spectral_unresolved_velocity_k"])
                for f in factors
            },
        }.items():
            left = np.array([get_value(r) for r in transient_records])
            right = np.array([get_value(r) for r in stationary_records])
            report["unconditional"][name] = {
                "transient_mean": float(np.mean(left)),
                "stationary_mean": float(np.mean(right)),
                "difference_stationary_minus_transient": float(np.mean(right) - np.mean(left)),
                "transient_snapshot_std": float(np.std(left, ddof=1)),
                "stationary_snapshot_std": float(np.std(right, ddof=1)),
            }
        for factor in factors:
            t = _read_cell_samples(base, factor, transient)
            s = _read_cell_samples(base, factor, stationary)
            result = matched_regime_comparison(
                t, s, min_rows_per_bin=20, min_snapshots_per_bin=2,
                bootstrap_repeats=bootstrap, seed=seed + factor
            )
            _write_json(base / f"matched_factor_{factor}.json", result)
            report["comparison"][str(factor)] = {
                "status": result["status"],
                "common_bins": result.get("common_bins"),
                "fraction_transient_retained": result.get("transient", {}).get(
                    "fraction_retained"),
                "fraction_stationary_retained": result.get("stationary", {}).get(
                    "fraction_retained"),
            }
        report["comparison_status"] = "computed_with_reviewed_index_cutoffs"
        report["reviewed_cutoff"] = bounds
    _write_json(base / "summary.json", report)
    _write_json(base / "candidate_cutoff_for_review.json", {
        "case_id": case,
        "candidate_interval_not_a_final_cutoff": candidate.__dict__,
        "transient_end_index": None,
        "stationary_start_index": None,
        "warning": "fill these indices only after manually reviewing stationarity plots",
    })
    narrative = [
        f"# {case}: DNS-only stationarity and residual audit",
        "",
        f"Completed snapshots: {len(records)}.",
        "",
        f"Stationarity heuristic: {candidate.status}.",
        candidate.explanation,
        "",
        "The heuristic is NOT a definitive transient cutoff.",
        "RUN/run.params-derived times assume a contiguous output schedule and ",
        "an unverified time origin; forcing characteristic time is not an",
        "independently measured large-eddy turnover time.",
        "Temperature, turbulent Mach, Taylor Reynolds and dissipation are unavailable",
        "until authoritative EOS/viscosity information is verified.",
        "Spectral filtering is orthogonal for velocity; block restrictions are",
        "equal-voxel discrete conservative, not a verified AVBP cell-volume integral.",
        "Matching compares a finite local coarse feature set, not full LES fields.",
        "Uncertainty uses snapshot-block resampling; spatial points are not",
        "independent replicates. Review physical-time spacing before inference.",
        "",
        f"Comparison status: {report.get('comparison_status','unknown')}.",
        "",
    ]
    if report.get("unconditional"):
        narrative += ["## Unconditional transient/stationary differences", ""]
        for name, results in report["unconditional"].items():
            narrative.append(
                f"- {name}: transient {results['transient_mean']:.6g}; "
                f"stationary {results['stationary_mean']:.6g}; "
                f"difference {results['difference_stationary_minus_transient']:.6g}"
            )
    if report.get("comparison"):
        narrative += ["", "## Coarse-feature-matched comparisons", ""]
        for factor, overview in report["comparison"].items():
            narrative.append(f"- factor {factor}: {overview['status']}; "
                             f"shared bins {overview.get('common_bins')}; "
                             "see matched_factor_N.json for effects and confidence intervals")
    (base / "scientific_report.md").write_text("\n".join(narrative))
    return report


def _run_case(args, case: str, cutoffs: dict) -> None:
    base = _case_cache_root(args.output_dir, case)
    base.mkdir(parents=True, exist_ok=True)
    all_files = _snapshot_files(args.root, case)
    schedule = _run_params_save_schedule(_case_dir(args.root, case), all_files)
    files = all_files
    if args.max_snapshots is not None:
        files = files[:args.max_snapshots]
    mesh = _case_dir(args.root, case) / "MESH" / "mesh.mesh.h5"
    mesh_stat = mesh.stat()
    signature = {
        "case": case, "root": str(args.root.resolve()),
        "mesh": str(mesh.resolve()), "mesh_size": mesh_stat.st_size,
        "mesh_mtime_ns": mesh_stat.st_mtime_ns,
        "snapshots": [
            (str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns)
            for path in files
        ],
        "factors": args.factors, "max_lag": args.max_lag,
        "sample_cells": args.sample_cells, "seed": args.seed,
        "code_revision": _revision(),
        "run_params_save_schedule": schedule,
    }
    signature["hash"] = _sha_manifest(signature)
    signature_path = base / "source_signature.json"
    if signature_path.exists():
        prior = json.loads(signature_path.read_text())
        if prior["hash"] != signature["hash"]:
            raise RuntimeError(f"{case}: stale/incompatible cache signature; "
                               "use another output directory, do not overwrite")
    else:
        if any((base / "snapshots").glob("snapshot_*.json")):
            raise RuntimeError("unversioned partial results found, refusing reuse")
        _write_json(signature_path, signature)
    grid = _mesh_grid(mesh)
    print(f"{case}: verified unique periodic grid {grid.shape}", flush=True)
    for index, path in enumerate(files):
        destination = base / "snapshots" / f"snapshot_{index:03d}.json"
        if destination.is_file():
            continue
        record = _sample_snapshot(
            path, grid, args.factors, args.max_lag,
            args.sample_cells, args.seed, index, base, schedule
        )
        _write_json(destination, record)
        print(f"  {index+1}/{len(files)} {path.name}: "
              f"K={record['metrics']['turbulent_k']:.6g}", flush=True)
    _postprocess_case(args.output_dir, case, args.factors,
                      cutoffs, args.bootstrap, args.seed)


def _compare_only(args, case: str, cutoffs: dict) -> None:
    if case not in cutoffs:
        raise ValueError("compare mode requires reviewed cutoffs for each selected case")
    base = _case_cache_root(args.output_dir, case)
    signature = json.loads((base / "source_signature.json").read_text())
    if tuple(signature["factors"]) != args.factors:
        raise ValueError("compare mode factors do not match cached audit")
    _postprocess_case(args.output_dir, case, args.factors,
                      cutoffs, args.bootstrap, args.seed)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "case", "all", "compare"),
                   default="preflight")
    p.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--case", choices=[x.case_id for x in DEFAULT_M43_CASES])
    p.add_argument("--factors", type=int, nargs="+", default=[2, 4, 8])
    p.add_argument("--max-lag", type=int, default=8)
    p.add_argument("--sample-cells", type=int, default=2048)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--max-snapshots", type=int)
    p.add_argument("--cutoffs-json", type=Path)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.mode == "case" and not args.case:
        p.error("--mode case requires --case")
    if args.max_lag < 1 or args.sample_cells < 1 or args.bootstrap < 0:
        p.error("invalid lag, sample-cell count or bootstrap count")
    if len(set(args.factors)) != len(args.factors) or any(
        f < 2 or f & (f - 1) for f in args.factors
    ):
        p.error("factors must be distinct powers of two >=2")
    args.factors = tuple(args.factors)
    return args


def main() -> None:
    args = _args()
    cases = ([args.case] if args.case else [c.case_id for c in DEFAULT_M43_CASES])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        preflight(args.root, args.output_dir, cases)
        return
    cutoffs = _load_cutoffs(args.cutoffs_json)
    for case in cases:
        if args.mode == "compare":
            _compare_only(args, case, cutoffs)
        else:
            _run_case(args, case, cutoffs)
    _write_json(args.output_dir / "audit_status.json", {
        "mode": args.mode, "processed_cases": cases, "revision": _revision(),
        "cutoffs_provided": bool(cutoffs),
        "note": "All outputs are descriptive; no training data or checkpoint changed",
    })


if __name__ == "__main__":
    main()
