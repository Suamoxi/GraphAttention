"""M43 read-only stationarity qualification and case-normalized DNS comparisons.

Modes:
  qualify: consume precomputed audit CSV/NPZ and propose, NEVER approve, cutoffs
  sample:  require reviewed cutoffs and M3.3 YAML, sample unique DNS mesh nodes
  compare: require reviewed cutoffs, samples, and YAML; compare six cases

No training script, split, normalizer, original HDF5 or checkpoint is modified.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_attention.evaluation.m43_qualification import (
    QualificationConfig,
    block_bootstrap_mean,
    effective_block_length,
    qualify_case,
    snapshot_labels,
    spectral_energy_fractions,
    validate_reviewed_cutoffs,
    wasserstein_quantile_1d,
)
from graph_attention.data.multicase_avbp import DEFAULT_M43_CASES, reference_filename
from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.nondimensionalization import ConvectiveNondimensionalizer
from graph_attention.evaluation.dns_residual import CONSERVATIVE_FIELDS
from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d

ALL_CASES = tuple(spec.case_id for spec in DEFAULT_M43_CASES)
CHANNELS = (*CONSERVATIVE_FIELDS, "u", "v", "w")
OUT_VERSION = "m43_qualification_v1"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temporary")
    temp.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("refuse empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".temporary")
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _case_path(root: Path, case: str) -> Path:
    if case not in ALL_CASES:
        raise ValueError(f"unknown case: {case}")
    return root / "cases" / case.replace("/", "_")


def _read_history(root: Path, case: str) -> dict[str, np.ndarray]:
    filename = _case_path(root, case) / "time_history.csv"
    with filename.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 100 or [int(r["snapshot_index"]) for r in rows] != list(range(100)):
        raise ValueError(f"{case}: require complete indexed history of 100 snapshots")
    required = ("turbulent_k", "enstrophy", "std_rho", "mean_rho",
                "u_rms_one_component", "f4_spectral_unresolved_energy_fraction")
    result = {key: np.array([float(r[key]) for r in rows], dtype=float)
              for key in required}
    if any(not np.isfinite(x).all() for x in result.values()):
        raise ValueError(f"{case}: nonfinite history")
    return result


def _read_spectra(root: Path, case: str, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    wave, energy = None, []
    for index in indices:
        filename = _case_path(root, case) / "spectra" / f"snapshot_{index:03d}.npz"
        with np.load(filename, allow_pickle=False) as data:
            current_wave, current_energy = (np.asarray(data[k], dtype=float) for k in ("k", "E"))
        if wave is None:
            wave = current_wave
        elif not np.array_equal(wave, current_wave):
            raise ValueError(f"{case}: inconsistent spectral wavenumber support")
        energy.append(current_energy)
    if wave is None:
        raise ValueError(f"{case}: no spectra")
    return wave, np.stack(energy)


def _reviewed_bounds(path: Path | None) -> dict[str, dict[str, int]]:
    if path is None:
        return {}
    raw = json.loads(path.read_text())
    if not isinstance(raw, dict) or set(raw) != set(ALL_CASES):
        raise ValueError("reviewed-cutoffs JSON must provide exactly all six case IDs")
    for case, bounds in raw.items():
        validate_reviewed_cutoffs(bounds, 100)
    return raw


def _read_reference_scales(reference_dir: Path) -> tuple[dict[str, dict[str, Any]], str]:
    references: dict[str, dict[str, Any]] = {}
    schemes = set()
    for case in ALL_CASES:
        filename = reference_dir / reference_filename(case)
        definition = load_case_definition(filename)
        if definition.case_id != case:
            raise ValueError(f"{case}: reference case ID mismatch")
        nondim = ConvectiveNondimensionalizer(definition.reference_scales)
        scale = {key: nondim.field_scale(key) for key in CONSERVATIVE_FIELDS}
        # Reuse M3.3's exact physical scale rules for every channel.
        uref = definition.reference_scales["U_ref"].value
        lref = nondim.coordinate_scale()
        rref = definition.reference_scales["rho_ref"].value
        if not all(np.isfinite([uref, lref, rref])):
            raise ValueError(f"{case}: invalid M3.3 reference values")
        schemes.add(definition.reference_scales.scheme)
        references[case] = {
            "field_scales": scale, "U_ref": uref, "L_ref": lref,
            "rho_ref": rref, "source": str(filename.resolve()),
            "sha256": _sha256(filename), "scheme": definition.reference_scales.scheme,
        }
    if len(schemes) != 1:
        raise ValueError("all M43 cases require the SAME M3.3 reference scheme")
    return references, next(iter(schemes))


def run_qualify(audit_root: Path, destination: Path, bounds: dict, conf: QualificationConfig) -> None:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "version": OUT_VERSION,
        "reviewed_cutoffs_supplied": bool(bounds),
        "warning": "No proposed cutoff is authoritative; supervised review is required.",
        "cases": {},
    }
    proposals: dict[str, Any] = {}
    for case in ALL_CASES:
        hist = _read_history(audit_root, case)
        wave, spectra = _read_spectra(audit_root, case, list(range(100)))
        band_records = [spectral_energy_fractions(wave, row) for row in spectra]
        bands = {name: np.array([r[name] for r in band_records])
                 for name in band_records[0]}
        result = qualify_case({**hist, **bands}, conf)
        labels = snapshot_labels(100, result, bounds.get(case))
        audit_case = _case_path(audit_root, case)
        source = json.loads((audit_case / "source_signature.json").read_text())
        if len(source.get("snapshots", [])) != 100:
            raise ValueError(f"{case}: incompatible DNS source signature")
        summary["cases"][case] = {
            **result,
            "source_signature_sha256": _sha256(audit_case / "source_signature.json"),
            "history_sha256": _sha256(audit_case / "time_history.csv"),
            "classification_reviewed": case in bounds,
            "reviewed_cutoffs": bounds.get(case),
            "counts": {label: labels.count(label) for label in sorted(set(labels))},
        }
        start = result["candidate_stationary_start_index"]
        transition = result["candidate_transition_start_index"]
        proposals[case] = {
            "candidate_transient_end_index": transition - 1 if transition is not None else None,
            "candidate_stationary_start_index": start,
            "reviewed_transient_end_index": None,
            "reviewed_stationary_start_index": None,
            "status": result["status"],
        }
        for index, (label, record) in enumerate(zip(labels, band_records, strict=True)):
            rows.append({
                "case_id": case, "snapshot_index": index,
                "snapshot_filename": Path(source["snapshots"][index][0]).name,
                "classification": label,
                "reviewed": case in bounds,
                "configuration_derived_elapsed_forcing_times_unverified": (
                    index * source["run_params_save_schedule"].get(
                        "save_interval_over_forcing_characteristic_time", float("nan"))
                ),
                "time_axis_status": source["run_params_save_schedule"].get("status", "unverified"),
                "turbulent_k_dimensional": hist["turbulent_k"][index],
                "enstrophy_dimensional": hist["enstrophy"][index],
                "std_rho_dimensional": hist["std_rho"][index],
                **record,
            })
        print(f"{case}: {result['status']}; candidate index={start}; "
              f"sensitivity={result['sensitivity_range_indices']}", flush=True)
    _write_json(destination / "qualification_summary.json", summary)
    _write_json(destination / "cutoff_proposals_NOT_REVIEWED.json", proposals)
    _write_csv(destination / "snapshot_classification_manifest.csv", rows)
    (destination / "qualification_readme.txt").write_text(
        "All suggested cutoffs are provisional. Do not use for model training.\n"
        "For subsequent field sampling and comparison, manually create a JSON\n"
        "mapping each case ID to the two ZERO-BASED integer keys:\n"
        "transient_end_index and stationary_start_index.\n"
    )


def _unique_periodic_source_nodes(mesh_path: Path) -> np.ndarray:
    with h5py.File(mesh_path, "r") as h5:
        xyz = np.column_stack([np.asarray(h5[f"Coordinates/{axis}"]).reshape(-1)
                               for axis in "xyz"])
    grid = infer_cartesian_grid_3d(xyz, periodic_endpoint_mode="drop_max")
    if any(length not in (64, 128) for length in grid.shape):
        raise ValueError(f"not a supported M43 periodic grid: {grid.shape}")
    # Convert canonical FFT-flat indices back to original AVBP node numbers.
    inverse = np.empty(xyz.shape[0], dtype=np.int64)
    inverse[grid.linear_indices] = np.arange(xyz.shape[0])
    unique = inverse.reshape(grid.source_shape)[:-1, :-1, :-1].ravel()
    if len(np.unique(unique)) != len(unique):
        raise ValueError("duplicate unique-periodic source-node index")
    return unique


def _sample_signature(audit_root: Path, bounds: dict, refs: dict, n: int, seed: int) -> dict:
    return {
        "version": OUT_VERSION,
        "mode": "physical_dns_field_samples",
        "cases": {case: {
            "source_signature_sha256": _sha256(_case_path(audit_root, case) / "source_signature.json"),
            "reference_sha256": refs[case]["sha256"],
            "cutoff": bounds[case],
        } for case in ALL_CASES},
        "nodes_per_snapshot": n,
        "seed": seed,
        "fields": list(CHANNELS),
        "periodic_endpoint_mode": "drop_max",
        "normalization": "M3.3 convective nondimensionalization, no per-case z-score",
    }


def _verify_signature(path: Path, expected: dict) -> None:
    if path.exists() and json.loads(path.read_text()) != expected:
        raise RuntimeError(f"incompatible existing sample cache: {path}; use new output dir")


def run_sample(
    audit_root: Path, dest: Path, bounds: dict, refs: dict, *,
    nodes_per_snapshot: int, seed: int,
) -> None:
    if nodes_per_snapshot < 16:
        raise ValueError("nodes_per_snapshot must be >= 16")
    signature = _sample_signature(audit_root, bounds, refs, nodes_per_snapshot, seed)
    signature_path = dest / "fields" / "source_signature.json"
    _verify_signature(signature_path, signature)
    if not signature_path.exists() and (dest / "fields").exists():
        if any((dest / "fields").rglob("*.npz")):
            raise RuntimeError("unversioned field cache already exists")
    _write_json(signature_path, signature)
    for case_i, case in enumerate(ALL_CASES):
        signature_case = json.loads((_case_path(audit_root, case) / "source_signature.json").read_text())
        if len(signature_case["snapshots"]) != 100:
            raise ValueError("incomplete audit source signature")
        mesh = Path(signature_case["mesh"])
        stat = mesh.stat()
        if stat.st_size != signature_case["mesh_size"] or stat.st_mtime_ns != signature_case["mesh_mtime_ns"]:
            raise RuntimeError(f"{case}: mesh metadata changed since audit")
        available_nodes = _unique_periodic_source_nodes(mesh)
        start = bounds[case]["stationary_start_index"]
        directory = dest / "fields" / case.replace("/", "_")
        directory.mkdir(parents=True, exist_ok=True)
        for index in range(start, 100):
            source, size, timestamp = signature_case["snapshots"][index]
            source_path = Path(source)
            stat = source_path.stat()
            if stat.st_size != size or stat.st_mtime_ns != timestamp:
                raise RuntimeError(f"{case}: source snapshot changed since audit: {source}")
            out = directory / f"snapshot_{index:03d}.npz"
            if out.exists():
                with np.load(out, allow_pickle=False) as saved:
                    if tuple(saved["values"].shape) != (min(nodes_per_snapshot, len(available_nodes)), len(CHANNELS)):
                        raise RuntimeError(f"incompatible cached snapshot: {out}")
                continue
            rng = np.random.default_rng(seed + case_i * 100003 + index * 10007)
            count = min(nodes_per_snapshot, len(available_nodes))
            selected = np.sort(rng.choice(available_nodes, size=count, replace=False))
            arrays = []
            with h5py.File(source_path, "r") as h5:
                for name in CONSERVATIVE_FIELDS:
                    dataset = h5[f"GaseousPhase/{name}"]
                    if dataset.ndim == 1:
                        values = np.asarray(dataset[selected], dtype=np.float64)
                    elif dataset.ndim == 2 and dataset.shape[1] == 1:
                        values = np.asarray(dataset[selected, 0], dtype=np.float64)
                    else:
                        raise ValueError(f"{source}: unexpected field shape {dataset.shape}")
                    arrays.append(values / refs[case]["field_scales"][name])
            conservative = np.stack(arrays, axis=-1)
            if not np.isfinite(conservative).all() or np.any(conservative[:, 0] <= 0):
                raise ValueError(f"{case}: invalid nondimensional conservative sample")
            # u* = rhou* / rho*, since M3.3 divides momentum by rho_ref * U_ref.
            velocity = conservative[:, 1:4] / conservative[:, 0, None]
            state = np.column_stack((conservative, velocity))
            tmp = out.with_name(out.name + ".temporary.npz")
            np.savez_compressed(tmp, values=state, snapshot_index=index, source=str(source_path))
            os.replace(tmp, out)
        print(f"{case}: sampled stationary interval {start}..99", flush=True)


def _case_values(dest: Path, case: str, indices: list[int]) -> np.ndarray:
    root = dest / "fields" / case.replace("/", "_")
    arrays = []
    for i in indices:
        with np.load(root / f"snapshot_{i:03d}.npz", allow_pickle=False) as data:
            values = np.asarray(data["values"], dtype=float)
            if values.ndim != 2 or values.shape[1] != len(CHANNELS) or not np.isfinite(values).all():
                raise ValueError(f"{case}: invalid cached dimensionless field samples")
            arrays.append(values)
    return np.stack(arrays)


def _spectra_on_common_grid(
    audit_root: Path, case: str, indices: list[int], refs: dict,
) -> tuple[np.ndarray, np.ndarray]:
    k, spectra = _read_spectra(audit_root, case, indices)
    uref, lref = refs[case]["U_ref"], refs[case]["L_ref"]
    return k * lref, np.mean(spectra, axis=0) / (uref * uref * lref)


def _spectral_distance(k: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    dk = np.diff(k)
    integral = lambda y: float(np.sum(0.5 * (y[:-1] + y[1:]) * dk))
    return integral(np.abs(left - right)) / max(0.5 * (integral(left) + integral(right)), 1e-20)


def _pair_type(left: str, right: str) -> str:
    re_a, ma_a = left.split("/")
    re_b, ma_b = right.split("/")
    if re_a == re_b:
        return "same_Re_resolution_vary_Ma"
    if ma_a == ma_b:
        return "same_Ma_vary_Re_and_resolution_confounded"
    return "both_Re_resolution_and_Ma_differ"


def _lag_statistics(
    audit_root: Path, case: str, indices: list[int], refs: dict, factor: int,
) -> dict[str, Any]:
    """Cache-derived Favre residual spatial statistics, all in M3.3 units."""
    collected: dict[str, list[np.ndarray]] = {}
    expected_r = None
    for index in indices:
        path = _case_path(audit_root, case) / "lag" / f"snapshot_{index:03d}_f{factor}.npz"
        with np.load(path, allow_pickle=False) as data:
            radius = np.asarray(data["r"], dtype=float)
            if expected_r is None:
                expected_r = radius
            elif not np.allclose(radius, expected_r, atol=0, rtol=1e-12):
                raise ValueError(f"{case}: inconsistent lag-distance grid")
            for name in ("r_ll", "s2_ll", "s3_ll", "s4_ll"):
                collected.setdefault(name, []).append(np.asarray(data[name], dtype=float))
    if expected_r is None:
        raise ValueError(f"{case}: no lag samples")
    uref, lref = refs[case]["U_ref"], refs[case]["L_ref"]
    result = {"r_star": expected_r / lref,
              "delta_star": float(factor * expected_r[0] / lref), "factor": factor}
    for name, pieces in collected.items():
        order = {"r_ll": 0, "s2_ll": 2, "s3_ll": 3, "s4_ll": 4}[name]
        result[name + "_star"] = np.mean(pieces, axis=0) / (uref**order)
    return result


def _lag_pairwise(audit_root: Path, dest: Path, bounds: dict, refs: dict) -> list[dict]:
    cache = {}
    for case in ALL_CASES:
        indices = list(range(bounds[case]["stationary_start_index"], 100))
        for factor in (2, 4, 8):
            cache[(case, factor)] = _lag_statistics(audit_root, case, indices, refs, factor)
    output = []
    for i, left in enumerate(ALL_CASES):
        for right in ALL_CASES[i + 1:]:
            # Compare only similar dimensionless FILTER width; same numerical f
            # across 64^3 and 128^3 is generally not equivalent.
            options = [(f, g) for f in (2, 4, 8) for g in (2, 4, 8)]
            a_factor, b_factor = min(options, key=lambda fg: abs(
                np.log(cache[left, fg[0]]["delta_star"] /
                       cache[right, fg[1]]["delta_star"])
            ))
            a = cache[left, a_factor]
            b = cache[right, b_factor]
            mismatch = abs(a["delta_star"] - b["delta_star"]) / max(
                a["delta_star"], b["delta_star"])
            status = "matched_filter_width" if mismatch < 0.10 else "unmatched_filter_width"
            record = {
                "case_a": left, "case_b": right,
                "comparison": _pair_type(left, right),
                "factor_a": a_factor, "factor_b": b_factor,
                "delta_star_a": a["delta_star"], "delta_star_b": b["delta_star"],
                "filter_width_relative_mismatch": mismatch,
                "status": status,
                "residual_S2_star_relative_L1": None,
                "residual_longitudinal_correlation_L1": None,
            }
            rmin = max(a["r_star"].min(), b["r_star"].min())
            rmax = min(a["r_star"].max(), b["r_star"].max())
            if status == "matched_filter_width" and rmax > 1.2 * rmin:
                shared_r = np.linspace(rmin, rmax, 64)
                left_s2 = np.interp(shared_r, a["r_star"], a["s2_ll_star"])
                right_s2 = np.interp(shared_r, b["r_star"], b["s2_ll_star"])
                left_corr = np.interp(shared_r, a["r_star"], a["r_ll_star"])
                right_corr = np.interp(shared_r, b["r_star"], b["r_ll_star"])
                norm = max(0.5 * (np.mean(np.abs(left_s2)) +
                                  np.mean(np.abs(right_s2))), 1e-20)
                record["residual_S2_star_relative_L1"] = float(
                    np.mean(np.abs(left_s2 - right_s2)) / norm)
                record["residual_longitudinal_correlation_L1"] = float(
                    np.mean(np.abs(left_corr - right_corr)))
            output.append(record)
    _write_csv(dest / "pairwise_m3_3_residual_lag.csv", output)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for case in ALL_CASES:
        item = cache[(case, 4)]
        ax.plot(item["r_star"], item["s2_ll_star"], lw=2,
                label=f"{case}, f=4")
    ax.set(xlabel=r"Dimensionless separation $r^*=r/L_{ref}$",
           ylabel=r"Favre residual longitudinal $S_2^*(r^*)$ [f=4]")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25)
    fig.savefig(dest / "stationary_m3_3_residual_S2.pdf")
    plt.close(fig)
    return output


def run_compare(audit_root: Path, dest: Path, bounds: dict, refs: dict, scheme: str, seed: int) -> None:
    expected = _sample_signature(
        audit_root, bounds, refs,
        json.loads((dest / "fields" / "source_signature.json").read_text())["nodes_per_snapshot"],
        json.loads((dest / "fields" / "source_signature.json").read_text())["seed"],
    )
    _verify_signature(dest / "fields" / "source_signature.json", expected)
    all_samples, case_spectra, summaries = {}, {}, {}
    for case in ALL_CASES:
        start = bounds[case]["stationary_start_index"]
        indices = list(range(start, 100))
        values = _case_values(dest, case, indices)
        all_samples[case] = values
        wave, spectrum = _spectra_on_common_grid(audit_root, case, indices, refs)
        case_spectra[case] = (wave, spectrum)
        hist = _read_history(audit_root, case)
        norms = {
            "turbulent_k_over_U_ref_squared": refs[case]["U_ref"] ** 2,
            "enstrophy_times_L_ref_squared_over_U_ref_squared": (
                refs[case]["U_ref"] / refs[case]["L_ref"]
            ) ** 2,
            "rho_rms_over_rho_ref": refs[case]["rho_ref"],
        }
        names = {"turbulent_k_star": ("turbulent_k", norms["turbulent_k_over_U_ref_squared"]),
                 "enstrophy_star": ("enstrophy", norms["enstrophy_times_L_ref_squared_over_U_ref_squared"]),
                 "rho_rms_star": ("std_rho", norms["rho_rms_over_rho_ref"])}
        stats = {}
        for metric, (source, divisor) in names.items():
            a = hist[source][start:] / divisor
            length = effective_block_length(a)
            mean, low, high = block_bootstrap_mean(a, length=length, repeats=400,
                                                    seed=seed + len(stats))
            stats[metric] = {"mean": mean, "ci95": [low, high],
                             "block_length_snapshots": length}
        field_marginals = {}
        for j, name in enumerate(CHANNELS):
            pool = values[..., j].ravel()
            mu = float(pool.mean())
            sigma = float(pool.std())
            z = (pool - mu) / sigma if sigma > 0 else np.zeros_like(pool)
            field_marginals[name] = {
                "mean": mu, "std": sigma,
                "skewness": float(np.mean(z**3)),
                "excess_kurtosis": float(np.mean(z**4) - 3),
                "quantiles_01_50_99": np.quantile(pool, [0.01, 0.5, 0.99]).tolist(),
            }
        summaries[case] = {
            "snapshot_count": len(indices),
            "first_stationary_snapshot_1_based": start + 1,
            "mesh": 65 if "Re54" in case else 129,
            "reference_file_sha256": refs[case]["sha256"],
            "normalized_turbulence_statistics": stats,
            "m3_3_marginals": field_marginals,
        }
    kmin = max(k.min() for k, _ in case_spectra.values())
    kmax = min(k.max() for k, _ in case_spectra.values())
    if kmax <= 2 * kmin:
        raise ValueError("insufficient shared dimensionless wavenumber support")
    shared_k = np.geomspace(kmin, kmax, 128)
    shared_spectra = {case: np.interp(shared_k, wave, spectrum)
                      for case, (wave, spectrum) in case_spectra.items()}
    distances = []
    for i, left in enumerate(ALL_CASES):
        for right in ALL_CASES[i + 1:]:
            a = all_samples[left]
            b = all_samples[right]
            comparison = _pair_type(left, right)
            spectral = _spectral_distance(shared_k, shared_spectra[left], shared_spectra[right])
            for j, channel in enumerate(CHANNELS):
                xa = a[..., j].ravel()
                xb = b[..., j].ravel()
                sigma = float(np.sqrt((np.var(xa) + np.var(xb)) / 2))
                w1 = wasserstein_quantile_1d(xa, xb)
                # Half-trajectory discrepancy is an uncertainty diagnostic, not a null p-value.
                mid_a = a.shape[0] // 2
                mid_b = b.shape[0] // 2
                baseline_a = wasserstein_quantile_1d(a[:mid_a, :, j].ravel(),
                                                       a[mid_a:, :, j].ravel())
                baseline_b = wasserstein_quantile_1d(b[:mid_b, :, j].ravel(),
                                                       b[mid_b:, :, j].ravel())
                distances.append({
                    "case_a": left, "case_b": right, "comparison": comparison,
                    "field": channel, "W1_m3_3_dimensionless": w1,
                    "W1_over_pooled_sigma": w1 / sigma if sigma > 0 else 0.0,
                    "W1_within_half_split_case_a": baseline_a,
                    "W1_within_half_split_case_b": baseline_b,
                    "spectral_L1_common_k_star": spectral,
                    "common_k_star_min": kmin, "common_k_star_max": kmax,
                })
    _write_json(dest / "cross_case_summary.json", {
        "version": OUT_VERSION, "reference_scheme": scheme,
        "cutoffs": bounds, "cases": summaries,
        "common_k_star": {"min": kmin, "max": kmax, "points": len(shared_k)},
        "warnings": [
            "Comparisons use reviewed stationary intervals; review does not prove stationarity.",
            "Re54 uses 65^3 and Re94 uses 129^3: Reynolds and resolution are confounded.",
            "The original per-case reference YAML is required; no case-specific z-score is fitted.",
            "Only the common resolved k* range is compared; spectral interpolation is approximate.",
            "Residual lag comparisons require matched dimensionless filter widths; mismatches are marked unavailable.",
            "Within-case half-trajectory W1 is descriptive and can be biased by temporal correlation.",
            "Time-based CIs use moving-block bootstrap; single-realization uncertainty remains.",
            "M3.3 rhoE* retains physical compressibility/Mach dependence.",
            "Node sampling uses equal discrete voxel weights, not validated AVBP dual volumes.",
        ],
    })
    _write_csv(dest / "pairwise_m3_3_wasserstein.csv", distances)
    _lag_pairwise(audit_root, dest, bounds, refs)
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for case in ALL_CASES:
        ax.loglog(shared_k, shared_spectra[case], label=case, lw=2)
    ax.set(xlabel=r"Dimensionless physical wavenumber $k^*=kL_{ref}$",
           ylabel=r"Dimensionless velocity energy spectrum $E^*(k^*)$")
    ax.tick_params(labelsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.25, which="both")
    fig.savefig(dest / "stationary_m3_3_energy_spectra.pdf")
    plt.close(fig)
    report = ["# M43: M3.3 stationary cross-case comparison", "",
              f"Reference scheme: `{scheme}`. Human-reviewed cutoffs supplied.", "",
              "| Case | Stationary first snapshot (1-based) | Count | K* mean | Enstrophy* mean |",
              "|---|---:|---:|---:|---:|"]
    for case in ALL_CASES:
        row = summaries[case]
        r = row["normalized_turbulence_statistics"]
        report.append(
            f"| {case} | {row['first_stationary_snapshot_1_based']} | {row['snapshot_count']} | "
            f"{r['turbulent_k_star']['mean']:.4g} | {r['enstrophy_star']['mean']:.4g} |"
        )
    report += ["", "This is descriptive, not a decision that conditioning is unnecessary.",
               "See cross_case_summary.json, pairwise_m3_3_wasserstein.csv, and pairwise_m3_3_residual_lag.csv.", ""]
    (dest / "cross_case_report.md").write_text("\n".join(report))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("qualify", "sample", "compare"), required=True)
    parser.add_argument("--audit-root", type=Path,
                        default=Path("/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--reviewed-cutoffs", type=Path)
    parser.add_argument("--reference-dir", type=Path,
                        default=Path("/scratch/coop/theret/HIT_DNS_COMP_case_references"))
    parser.add_argument("--nodes-per-snapshot", type=int, default=512)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def main() -> None:
    args = _args()
    audit = args.audit_root.resolve(strict=True)
    output = (args.output_dir or audit / OUT_VERSION).resolve()
    cutoffs = _reviewed_bounds(args.reviewed_cutoffs)
    if args.mode in {"sample", "compare"} and len(cutoffs) != len(ALL_CASES):
        raise ValueError("field sampling and cross-case comparison require six reviewed cutoffs")
    refs, scheme = _read_reference_scales(args.reference_dir) if args.mode != "qualify" else ({}, "")
    output.mkdir(parents=True, exist_ok=True)
    if args.mode == "qualify":
        run_qualify(audit, output, cutoffs, QualificationConfig(seed=args.seed))
    elif args.mode == "sample":
        run_sample(audit, output, cutoffs, refs,
                   nodes_per_snapshot=args.nodes_per_snapshot, seed=args.seed)
    else:
        run_compare(audit, output, cutoffs, refs, scheme, args.seed)
    print(f"M43 {args.mode} completed: {output}", flush=True)


if __name__ == "__main__":
    main()
