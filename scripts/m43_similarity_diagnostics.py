"""M43 reference-state similarity test based on cached, review-gated DNS output.

Modes:
  inspect: inventory immutable M3.3 references, create EMPTY thermodynamic templates
  compare: use explicitly provided physical reference states and reviewed
           41..100-style snapshots; no DNS read, model or normalizer writes

The initial M3.3 reference YAML and its hash are NEVER changed by this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.contracts import ReferenceScope
from graph_attention.data.multicase_avbp import reference_filename
from graph_attention.evaluation.m43_qualification import (
    block_bootstrap_mean, effective_block_length, wasserstein_quantile_1d,
)
from graph_attention.evaluation.m43_similarity import (
    DIAGNOSTIC_CHANNELS, SimilarityReferences, circular_block_indices,
    common_shell_spectra, similarity_inverse, similarity_transform,
    spectral_relative_l1,
)
from scripts import m43_qualify_compare as qualification

VERSION = "m43_reference_similarity_v1"
EXTRA_REFERENCES = ("rho0", "a_ref", "e_ref")
_EXPECTED_UNITS = {
    "rho0": {"kg/m^3", "kg/m3"},
    "a_ref": {"m/s"},
    "e_ref": {"j/kg", "m^2/s^2", "m2/s2"},
}
CHANNEL_SOURCES = ("rho", "u", "v", "w", "rhoE")


def _json(path: Path) -> dict:
    obj = json.loads(path.read_text())
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object in {path}")
    return obj


def _reference_value(value, case: str, key: str) -> float:
    if (value is None or value.provenance is None or value.units is None
        or value.inference_available is not True or value.scope is ReferenceScope.SNAPSHOT
        or not np.isfinite(value.value) or value.value <= 0):
        raise ValueError(f"{case}: {key} requires positive finite units, source provenance, "
                         "case-level scope and availability at inference")
    if value.provenance.startswith("REPLACE_"):
        raise ValueError(f"{case}: {key} still has a template provenance placeholder")
    if value.units.strip().lower().replace(" ", "") not in _EXPECTED_UNITS[key]:
        raise ValueError(f"{case}: {key} requires documented SI units compatible with "
                         f"{sorted(_EXPECTED_UNITS[key])}, got {value.units!r}")
    return float(value.value)


def _sidecar_template(case: str, scheme: str) -> str:
    lines = [
        "# TEMPLATE ONLY. Replace null values with authoritative simulation-setup values.",
        "# Do not derive rho0, a_ref or e_ref from the stationary DNS snapshots.",
        f"case_id: {case}", f"reference_scheme: {scheme}", "references:",
    ]
    definitions = {
        "rho0": ("kg/m^3", "prescribed_conserved_mean_density"),
        "a_ref": ("m/s", "reference_thermodynamic_sound_speed"),
        "e_ref": ("J/kg", "reference_thermodynamic_specific_internal_energy"),
    }
    for name, (units, definition) in definitions.items():
        lines += [
            f"  {name}:", "    value: null", f"    units: {units}",
            f"    definition: {definition}",
            "    provenance: REPLACE_WITH_VERIFIABLE_SIMULATION_SETUP_SOURCE",
            "    inference_available: true", "    scope: case",
        ]
    return "\n".join(lines) + "\n"


def read_reference_states(reference_dir: Path, state_dir: Path,
                          refs_m33: dict, *, allow_incomplete: bool) -> tuple[dict, dict]:
    """Preserve the old SHA while reading additional authoritative sidecar values."""
    result, audit = {}, {}
    for case in qualification.ALL_CASES:
        base = load_case_definition(reference_dir / reference_filename(case))
        if base.case_id != case or base.reference_scales.scheme != refs_m33[case]["scheme"]:
            raise ValueError(f"{case}: M3.3 case definition/scheme mismatch")
        extra_path = state_dir / reference_filename(case)
        extra = load_case_definition(extra_path) if extra_path.is_file() else None
        if extra is not None and (extra.case_id != case or
                                 extra.reference_scales.scheme != base.reference_scales.scheme):
            raise ValueError(f"{case}: sidecar must use same exact case/scheme as M3.3")
        scales = {}
        missing = []
        meta = {}
        for name in EXTRA_REFERENCES:
            old = base.reference_scales
            new = extra.reference_scales if extra else None
            value = (new[name] if new and name in new.names
                     else old[name] if name in old.names else None)
            if value is None:
                missing.append(name)
                continue
            scales[name] = _reference_value(value, case, name)
            meta[name] = {"value": scales[name], "units": value.units,
                          "definition": value.definition, "provenance": value.provenance,
                          "source": str(extra_path if new and name in new.names
                                        else base.source_path)}
        if missing and not allow_incomplete:
            raise ValueError(f"{case}: missing {missing} from explicit state reference YAML "
                             f"{extra_path}; run --mode inspect and complete the template")
        m33 = refs_m33[case]
        audit[case] = {
            "m3_3_reference_scheme": m33["scheme"],
            "m3_3_reference_file": m33["source"],
            "m3_3_reference_sha256": m33["sha256"],
            "state_reference_file": str(extra_path) if extra else None,
            "state_reference_sha256": qualification._sha256(extra_path) if extra else None,
            "missing_explicit_state_values": missing, "state_references": meta,
        }
        if not missing:
            obj = SimilarityReferences(
                m33["rho_ref"], m33["U_ref"], m33["L_ref"],
                scales["rho0"], scales["a_ref"], scales["e_ref"],
            )
            nominal = float(case.rsplit("Ma", maxsplit=1)[-1])
            audit[case]["mach_from_references"] = obj.mach
            audit[case]["nominal_mach_label"] = nominal
            audit[case]["relative_mach_label_mismatch"] = abs(obj.mach / nominal - 1.0)
            audit[case]["warning"] = (
                "Reference-based Mach differs from the nominal case label; check whether "
                "the two use different RMS conventions. No label value is used for scaling."
                if abs(obj.mach / nominal - 1.0) > 0.05 else None
            )
            result[case] = obj
    return result, audit


def run_inspect(audit_root: Path, destination: Path, reference_dir: Path,
                state_dir: Path) -> None:
    original, scheme = qualification._read_reference_scales(reference_dir)
    states, report = read_reference_states(reference_dir, state_dir, original,
                                           allow_incomplete=True)
    for case in qualification.ALL_CASES:
        hist = qualification._read_history(audit_root, case)
        observed_mean = float(np.mean(hist["mean_rho"][40:]))
        report[case]["observed_mean_rho_snapshot_41_to_100_diagnostic_only"] = observed_mean
        if case in states:
            refs = states[case]
            drho = float(np.mean(hist["std_rho"][40:]))
            report[case]["mean_density_offset_over_rms"] = (
                abs(observed_mean - refs.rho0) / max(drho, 1e-30))
            report[case]["mean_scaled_density_offset"] = (
                observed_mean - refs.rho0) / (refs.rho0 * refs.mach**2)
            if report[case]["mean_density_offset_over_rms"] > 0.2:
                report[case]["rho0_centering_warning"] = (
                    "Chosen physical rho0 differs from conserved mean by >20% of the "
                    "observed density RMS. Do not silently fit rho0 to these snapshots."
                )
        if len(report[case]["missing_explicit_state_values"]):
            template_dir = destination / "reference_templates_NOT_VALIDATED"
            template_dir.mkdir(parents=True, exist_ok=True)
            (template_dir / reference_filename(case)).write_text(
                _sidecar_template(case, scheme)
            )
    qualification._write_json(destination / "reference_audit.json", {
        "version": VERSION, "reference_scheme": scheme,
        "status": "ready_for_comparison" if len(states) == len(qualification.ALL_CASES)
                  else "missing_verified_thermodynamic_reference_values",
        "warning": "Observed DNS mean is a diagnostic, NEVER an automatically chosen rho0.",
        "cases": report,
    })
    print(f"Reference audit: {len(states)}/{len(qualification.ALL_CASES)} cases ready", flush=True)


def _trusted_inputs(audit_root: Path, qualification_root: Path,
                    original: dict, bounds: dict) -> dict:
    signature_path = qualification_root / "fields" / "source_signature.json"
    actual = _json(signature_path)
    expected = qualification._sample_signature(
        audit_root, bounds, original,
        int(actual["nodes_per_snapshot"]), int(actual["seed"]),
    )
    qualification._verify_signature(signature_path, expected)
    # The qualification manifest must confirm all six reviewed cutoffs.
    reviewed = _json(qualification_root / "qualification_summary.json")
    if not reviewed.get("reviewed_cutoffs_supplied"):
        raise ValueError("M43 qualification summary lacks six reviewed cutoff labels")
    for case in qualification.ALL_CASES:
        if (not reviewed["cases"][case]["classification_reviewed"] or
            reviewed["cases"][case]["reviewed_cutoffs"] != bounds[case]):
            raise ValueError(f"{case}: qualification label/cutoff mismatch")
    return actual


def _read_inputs(audit_root: Path, qualification_root: Path,
                 bounds: dict, refs: dict) -> tuple[dict, dict, dict, dict]:
    states, original, spectra, history = {}, {}, {}, {}
    for case in qualification.ALL_CASES:
        indices = list(range(bounds[case]["stationary_start_index"], 100))
        old = qualification._case_values(qualification_root, case, indices)
        new = similarity_transform(old, refs[case])
        if not np.allclose(similarity_inverse(new, refs[case]), old[..., :5],
                           rtol=5e-11, atol=1e-9):
            raise ValueError(f"{case}: similarity transformation failed inverse check")
        states[case], original[case] = new, old
        spectra[case] = qualification._read_spectra(audit_root, case, indices)
        history[case] = qualification._read_history(audit_root, case)
    return states, original, spectra, history


def _block_lengths(history: dict, bounds: dict) -> dict[str, int]:
    return {case: effective_block_length(
        history[case]["turbulent_k"][bounds[case]["stationary_start_index"]:])
        for case in qualification.ALL_CASES}


def _safe_ratio(numer: float, denom: float) -> float | None:
    return float(numer / denom) if abs(denom) > 1e-14 else None


def _distribution_rows(states: dict, original: dict, lengths: dict,
                       repeats: int, seed: int) -> tuple[list[dict], list[dict]]:
    stats, pairwise = [], []
    # 1) Snapshot-block uncertainty for per-case moments, not spatial-node iid CI.
    for case in qualification.ALL_CASES:
        for index, field in enumerate(DIAGNOSTIC_CHANNELS):
            values = states[case][..., index]
            snapshot_mean = np.mean(values, axis=1)
            snapshot_rms = np.std(values, axis=1)
            mean, ci_low, ci_high = block_bootstrap_mean(
                snapshot_mean, length=lengths[case], repeats=repeats,
                seed=seed + 43 * len(stats),
            )
            stats.append({"case_id": case, "field": field,
                          "mean": mean, "std_pooled": float(np.std(values)),
                          "mean_ci95_low": ci_low, "mean_ci95_high": ci_high,
                          "mean_snapshot_std": float(np.mean(snapshot_rms)),
                          "p01": float(np.quantile(values, 0.01)),
                          "p50": float(np.quantile(values, 0.5)),
                          "p99": float(np.quantile(values, 0.99)),
                          "block_length_snapshots": lengths[case]})
    # 2) Existing M43 W1 + half-trajectory baseline; compare before vs after.
    for i, left in enumerate(qualification.ALL_CASES):
        for right in qualification.ALL_CASES[i + 1:]:
            case_a, case_b = states[left], states[right]
            old_a, old_b = original[left], original[right]
            for j, field in enumerate(DIAGNOSTIC_CHANNELS):
                xa, xb = case_a[..., j], case_b[..., j]
                baseline_a = wasserstein_quantile_1d(
                    xa[:len(xa)//2].ravel(), xa[len(xa)//2:].ravel(), bins=256)
                baseline_b = wasserstein_quantile_1d(
                    xb[:len(xb)//2].ravel(), xb[len(xb)//2:].ravel(), bins=256)
                distance = wasserstein_quantile_1d(xa.ravel(), xb.ravel(), bins=256)
                sigma = float(np.sqrt(0.5 * (np.var(xa) + np.var(xb))))
                baseline = 0.5 * (baseline_a + baseline_b)
                # Existing source field indices: rho=0; u/v/w=5/6/7; rhoE=4.
                source_index = (0, 5, 6, 7, 4)[j]
                old_x, old_y = old_a[..., source_index], old_b[..., source_index]
                old_d = wasserstein_quantile_1d(old_x.ravel(), old_y.ravel(), bins=256)
                old_s = float(np.sqrt(0.5 * (np.var(old_x) + np.var(old_y))))
                rng = np.random.default_rng(seed + 997 * i + 71 * j + len(right))
                boot = np.empty(repeats, dtype=float)
                for b in range(repeats):
                    a_ix = circular_block_indices(len(xa), lengths[left], rng)
                    b_ix = circular_block_indices(len(xb), lengths[right], rng)
                    a_r, b_r = xa[a_ix].ravel(), xb[b_ix].ravel()
                    d = wasserstein_quantile_1d(a_r, b_r, bins=128)
                    s = np.sqrt(0.5 * (np.var(a_r) + np.var(b_r)))
                    boot[b] = d / s if s > 1e-20 else 0.0
                low, high = np.quantile(boot, [0.025, 0.975])
                pairwise.append({
                    "case_a": left, "case_b": right,
                    "comparison": qualification._pair_type(left, right),
                    "field": field, "source_field": CHANNEL_SOURCES[j],
                    "new_W1": distance,
                    "new_W1_over_pooled_sigma": _safe_ratio(distance, sigma),
                    "new_normalized_W1_CI95_low": float(low),
                    "new_normalized_W1_CI95_high": float(high),
                    "within_half_W1_a": baseline_a,
                    "within_half_W1_b": baseline_b,
                    "new_W1_over_within_half_baseline": _safe_ratio(distance, baseline),
                    "old_W1_over_pooled_sigma": _safe_ratio(old_d, old_s),
                    "old_W1": old_d,
                })
    return stats, pairwise


def _spectral_rows(audit_root: Path, spectra: dict, refs: dict,
                   lengths: dict, repeats: int, seed: int,
                   nyquist_fraction: float) -> tuple[np.ndarray, dict, list, dict]:
    grids = {case: 64 if case.startswith("Re54/") else 128
             for case in qualification.ALL_CASES}
    k_shared, data, supports = common_shell_spectra(
        spectra, refs, grids, nyquist_fraction=nyquist_fraction,
    )
    # Split *common resolved* k into equal log-span bands. Total spectral L1
    # emphasizes energy-bearing scales and can hide differences at small scales.
    edges = np.geomspace(k_shared[0], k_shared[-1], 4)
    band_masks = {
        label: ((k_shared >= edges[j]) & (k_shared <= edges[j+1]))
        for j, label in enumerate(("low", "mid", "high"))
    }
    out = []
    for i, left in enumerate(qualification.ALL_CASES):
        for right in qualification.ALL_CASES[i+1:]:
            a, b = data[left], data[right]
            mean_a, mean_b = np.mean(a, axis=0), np.mean(b, axis=0)
            dist = spectral_relative_l1(k_shared, mean_a, mean_b)
            by_band = {f"spectral_relative_L1_{label}_common_k": (
                spectral_relative_l1(k_shared[mask], mean_a[mask], mean_b[mask])
                if np.count_nonzero(mask) >= 2 else None
            ) for label, mask in band_masks.items()}
            within_a = spectral_relative_l1(k_shared,
                np.mean(a[:len(a)//2],axis=0), np.mean(a[len(a)//2:],axis=0))
            within_b = spectral_relative_l1(k_shared,
                np.mean(b[:len(b)//2],axis=0), np.mean(b[len(b)//2:],axis=0))
            rng = np.random.default_rng(seed + 37 * i + len(right))
            boot = []
            boot_bands = {label: [] for label in band_masks}
            for _ in range(repeats):
                ai = circular_block_indices(len(a), lengths[left], rng)
                bi = circular_block_indices(len(b), lengths[right], rng)
                mean_ai, mean_bi = np.mean(a[ai],axis=0), np.mean(b[bi],axis=0)
                boot.append(spectral_relative_l1(k_shared, mean_ai, mean_bi))
                for label, mask in band_masks.items():
                    if np.count_nonzero(mask) >= 2:
                        boot_bands[label].append(spectral_relative_l1(
                            k_shared[mask], mean_ai[mask], mean_bi[mask]))
            low, high = np.quantile(boot, [0.025, 0.975])
            band_uncertainty = {}
            for label, values in boot_bands.items():
                if values:
                    ll, hh = np.quantile(values, [0.025, 0.975])
                    band_uncertainty[f"spectral_{label}_L1_CI95_low"] = float(ll)
                    band_uncertainty[f"spectral_{label}_L1_CI95_high"] = float(hh)
                else:
                    band_uncertainty[f"spectral_{label}_L1_CI95_low"] = None
                    band_uncertainty[f"spectral_{label}_L1_CI95_high"] = None
            out.append({
                "case_a": left, "case_b": right,
                "comparison": qualification._pair_type(left, right),
                "spectral_relative_L1_on_common_resolved_k": dist,
                "spectral_L1_block_CI95_low": float(low),
                "spectral_L1_block_CI95_high": float(high),
                "within_half_spectral_L1_case_a": within_a,
                "within_half_spectral_L1_case_b": within_b,
                "spectral_L1_over_within_half_baseline":
                    _safe_ratio(dist, 0.5*(within_a+within_b)),
                "common_kstar_min": float(k_shared[0]),
                "common_kstar_max": float(k_shared[-1]),
                "conservative_nyquist_fraction": nyquist_fraction,
                **by_band, **band_uncertainty,
            })
    return k_shared, data, out, supports


def _matched_lag_rows(audit_root: Path, bounds: dict, refs_m33: dict,
                      lengths: dict, repeats: int, seed: int) -> tuple[list[dict], dict]:
    cache = {}
    for case in qualification.ALL_CASES:
        indices = list(range(bounds[case]["stationary_start_index"], 100))
        for factor in (2, 4, 8):
            cache[case, factor] = qualification._lag_statistics(
                audit_root, case, indices, refs_m33, factor)
    def snapshot_s2(case: str, factor: int) -> np.ndarray:
        # Same cache as _lag_statistics; retain the time axis for bootstrap.
        indices = range(bounds[case]["stationary_start_index"], 100)
        values = []
        for index in indices:
            filename = (qualification._case_path(audit_root, case) / "lag" /
                        f"snapshot_{index:03d}_f{factor}.npz")
            with np.load(filename, allow_pickle=False) as raw:
                values.append(np.asarray(raw["s2_ll"], dtype=float))
        return np.stack(values) / refs_m33[case]["U_ref"]**2

    series_cache = {}
    rows = []
    for i, left in enumerate(qualification.ALL_CASES):
        for right in qualification.ALL_CASES[i + 1:]:
            fa, fb = min(((f, g) for f in (2, 4, 8) for g in (2, 4, 8)),
                key=lambda fg: abs(np.log(
                    cache[left, fg[0]]["delta_star"] /
                    cache[right, fg[1]]["delta_star"])))
            a, b = cache[left, fa], cache[right, fb]
            mismatch = abs(a["delta_star"] - b["delta_star"]) / max(
                a["delta_star"], b["delta_star"])
            rmin = max(a["r_star"].min(), b["r_star"].min())
            rmax = min(a["r_star"].max(), b["r_star"].max())
            status = ("matched_filter_width" if mismatch < 0.10 and rmax > 1.2*rmin
                      else "unmatched_filter_or_insufficient_separation")
            record = {"case_a": left, "case_b": right,
                "comparison": qualification._pair_type(left,right),
                "factor_a": fa, "factor_b": fb,
                "delta_star_a": a["delta_star"], "delta_star_b": b["delta_star"],
                "relative_filter_width_mismatch": mismatch,
                "status": status,
                "residual_S2_relative_L1": None,
                "residual_S2_block_CI95_low": None,
                "residual_S2_block_CI95_high": None,
                "residual_correlation_mean_absolute_difference": None}
            if status == "matched_filter_width":
                rstar = np.linspace(rmin, rmax, 64)
                aa = np.interp(rstar, a["r_star"], a["s2_ll_star"])
                bb = np.interp(rstar, b["r_star"], b["s2_ll_star"])
                denom = 0.5 * (np.mean(np.abs(aa)) + np.mean(np.abs(bb)))
                record["residual_S2_relative_L1"] = _safe_ratio(np.mean(np.abs(aa-bb)), denom)
                ra = np.interp(rstar, a["r_star"], a["r_ll_star"])
                rb = np.interp(rstar, b["r_star"], b["r_ll_star"])
                record["residual_correlation_mean_absolute_difference"] = float(
                    np.mean(np.abs(ra-rb)))
                for case, factor in ((left, fa), (right, fb)):
                    if (case, factor) not in series_cache:
                        series_cache[case, factor] = snapshot_s2(case, factor)
                rng = np.random.default_rng(seed + 941*i + 7*fa + fb)
                boot = []
                for _ in range(repeats):
                    x = series_cache[left, fa]
                    y = series_cache[right, fb]
                    xi = circular_block_indices(len(x), lengths[left], rng)
                    yi = circular_block_indices(len(y), lengths[right], rng)
                    xx = np.interp(rstar, a["r_star"], np.mean(x[xi],axis=0))
                    yy = np.interp(rstar, b["r_star"], np.mean(y[yi],axis=0))
                    denom_boot = .5*(np.mean(np.abs(xx))+np.mean(np.abs(yy)))
                    boot.append(float(np.mean(np.abs(xx-yy))/max(denom_boot,1e-20)))
                low, high = np.quantile(boot, [0.025, 0.975])
                record["residual_S2_block_CI95_low"] = float(low)
                record["residual_S2_block_CI95_high"] = float(high)
            rows.append(record)
    return rows, cache


def _plots(destination: Path, original: dict, samples: dict, k: np.ndarray,
           spectra: dict, lag: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    specs = [(0, 0, "rho* (original M3.3)", False),
             (0, 0, "Mach-scaled density fluctuation", True),
             (4, 4, "(rho E)* (original M3.3)", False),
             (4, 4, "Reference-subtracted energy*", True)]
    for ax, (old_index, new_index, xlabel, transformed) in zip(axes.ravel(), specs):
        per_case = [samples[c][..., new_index].ravel() if transformed
                    else original[c][..., old_index].ravel()
                    for c in qualification.ALL_CASES]
        # Use the same x-support for all cases on each axis, without a z-score.
        low = min(float(np.quantile(v, 0.002)) for v in per_case)
        high = max(float(np.quantile(v, 0.998)) for v in per_case)
        bins = np.linspace(low, high, 110) if high > low else np.linspace(low-1,high+1,110)
        for case, values in zip(qualification.ALL_CASES, per_case, strict=True):
            ax.hist(values, bins=bins, density=True, histtype="step",
                    linewidth=1.8, label=case)
        ax.set(xlabel=xlabel, ylabel="Probability density")
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.2)
    axes[0,0].legend(fontsize=9)
    fig.savefig(destination / "reference_state_one_point_pdfs.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    for case in qualification.ALL_CASES:
        ax.loglog(k, np.mean(spectra[case], axis=0), lw=2.1, label=case)
    ax.set(xlabel=r"Common resolved dimensionless $k^*=kL_{ref}$",
           ylabel=r"Velocity spectrum $E^*(k^*)$")
    ax.set_xlim(float(k[0]), float(k[-1]))
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=10)
    fig.savefig(destination / "well_resolved_common_k_spectra.pdf")
    plt.close(fig)

    # Match *dimensionless filter width* across cases, not the filter factor.
    target = lag[qualification.ALL_CASES[0], 2]["delta_star"]
    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)
    plotted = []
    for case in qualification.ALL_CASES:
        fac = min((2, 4, 8), key=lambda f: abs(np.log(lag[case,f]["delta_star"]/target)))
        record = lag[case, fac]
        mismatch = abs(record["delta_star"]-target)/max(record["delta_star"],target)
        if mismatch >= 0.10:
            continue
        ax.plot(record["r_star"], record["s2_ll_star"], lw=2.1,
                label=f"{case}, f={fac}")
        plotted.append(case)
    ax.set(xlabel=r"Dimensionless separation $r^*=r/L_{ref}$",
           ylabel=r"Favre residual longitudinal $S_2^*(r^*)$")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.25)
    if plotted:
        ax.legend(fontsize=10)
    ax.text(0.02, 0.03, "Equal coarse width does not equalize DNS fine-scale bandwidth",
            transform=ax.transAxes, fontsize=10)
    fig.savefig(destination / "matched_delta_residual_S2.pdf")
    plt.close(fig)


def _rows_summary(rows: list[dict], field: str, key: str) -> dict:
    out = {}
    for category in sorted(set(row["comparison"] for row in rows)):
        values = [row[key] for row in rows if row.get("field") == field
                  and row["comparison"] == category and row[key] is not None]
        if values:
            out[category] = {"median": float(np.median(values)),
                             "min": float(min(values)), "max": float(max(values))}
    return out


def run_compare(audit_root: Path, qualification_root: Path, destination: Path,
                reference_dir: Path, state_dir: Path, cutoff_path: Path,
                *, repeats: int, seed: int, nyquist_fraction: float) -> None:
    if repeats < 16:
        raise ValueError("at least 16 bootstrap repeats required")
    bounds = qualification._reviewed_bounds(cutoff_path)
    old_refs, scheme = qualification._read_reference_scales(reference_dir)
    state_refs, audit = read_reference_states(reference_dir, state_dir, old_refs,
                                              allow_incomplete=False)
    cache = _trusted_inputs(audit_root, qualification_root, old_refs, bounds)
    sig = {"version": VERSION, "old_sample_cache_signature": cache,
        "reviewed_cutoffs_sha256": qualification._sha256(cutoff_path),
        "reference_scheme": scheme,
        "thermodynamic_reference_sha256": {case: audit[case]["state_reference_sha256"]
                         for case in qualification.ALL_CASES},
        "audit_cache_sha256": {case: qualification._sha256(
            qualification._case_path(audit_root,case) / "source_signature.json")
            for case in qualification.ALL_CASES},
        "bootstrap_repeats": repeats, "seed": seed,
        "nyquist_fraction": nyquist_fraction,
    }
    qualification._verify_signature(destination / "input_signature.json", sig)
    qualification._write_json(destination / "input_signature.json", sig)
    states, originals, raw_spectra, history = _read_inputs(
        audit_root, qualification_root, bounds, state_refs)
    lengths = _block_lengths(history, bounds)
    marginal, pairwise = _distribution_rows(states, originals, lengths, repeats, seed)
    kstar, spectral, spectral_pairs, support = _spectral_rows(
        audit_root, raw_spectra, state_refs, lengths, repeats, seed,
        nyquist_fraction)
    residual_pairs, lag = _matched_lag_rows(
        audit_root, bounds, old_refs, lengths, repeats, seed)
    qualification._write_csv(destination / "reference_state_marginals.csv", marginal)
    qualification._write_csv(destination / "pairwise_reference_state_wasserstein.csv", pairwise)
    qualification._write_csv(destination / "pairwise_well_resolved_spectra.csv", spectral_pairs)
    qualification._write_csv(destination / "pairwise_matched_delta_residual.csv", residual_pairs)
    for case in qualification.ALL_CASES:
        rho_obs = float(np.mean(history[case]["mean_rho"][40:]))
        density_rms = float(np.mean(history[case]["std_rho"][40:]))
        refs = state_refs[case]
        audit[case]["mean_rho_from_audit_diagnostic_only"] = rho_obs
        audit[case]["density_center_offset_over_rms"] = (
            abs(rho_obs - refs.rho0) / max(density_rms, 1e-20))
        audit[case]["mach_scaled_density_rms"] = density_rms/(refs.rho0*refs.mach**2)
        audit[case]["density_center_offset_in_mach_scaled_units"] = (
            rho_obs-refs.rho0)/(refs.rho0*refs.mach**2)
        audit[case]["e0_over_U_ref_squared"] = refs.e0_over_u0_squared
        audit[case]["block_length_snapshots"] = lengths[case]
    summary = {
        "version": VERSION, "reference_scheme": scheme,
        "status": "exploratory_nonstationarity_not_excluded",
        "source": "reviewed stationary 60-snapshot M43 sample cache, no DNS reread",
        "reviewed_cutoffs": bounds, "reference_audit": audit,
        "spectral_support": {"kstar_min": float(kstar[0]),
            "kstar_max": float(kstar[-1]), "points": len(kstar),
            "nyquist_fraction": nyquist_fraction,
            "individual_kstar_upper_limits": support,
            "three_equal_log_span_band_edges": np.geomspace(kstar[0], kstar[-1], 4).tolist()},
        "median_pairwise_normalized_w1_by_group": {
            field: _rows_summary(pairwise, field, "new_W1_over_pooled_sigma")
            for field in DIAGNOSTIC_CHANNELS},
        "median_pairwise_W1_over_temporal_baseline_by_group": {
            field: _rows_summary(pairwise, field, "new_W1_over_within_half_baseline")
            for field in DIAGNOSTIC_CHANNELS},
        "warnings": [
            "DNS stationary cutoffs are exploratory and not proof of stationarity.",
            "The nominal case Mach may differ from U_ref/a_ref if RMS conventions differ.",
            "Reference rho0, a_ref and e_ref are prescribed input values, never snapshot-fitted.",
            "W1 half-trajectory baseline and time-block bootstrap are descriptive, not p-values.",
            "Spatial nodes are not independent Monte Carlo samples of DNS realizations.",
            "Reynolds number and grid resolution are confounded in this six-case dataset.",
            "Common k support stays below 70% (or selected fraction) of minimum "
            "directional Nyquist.",
            "Equal dimensionless coarse-filter width does not match fine-grid bandwidth.",
            "Reference-state similarity is diagnostic, NOT a new model preprocessing choice.",
        ],
    }
    qualification._write_json(destination / "similarity_summary.json", summary)
    _plots(destination, originals, states, kstar, spectral, lag)
    lines = ["# M43 reference-state similarity diagnostics", "",
        f"Scheme: `{scheme}`. All six cases use the reviewed exploratory bounds.",
        f"Conservative shared spectral range: {kstar[0]:.5g} <= k* <= {kstar[-1]:.5g}.",
        "", "## Mach-scaled density RMS (not case-fit)", "",
        "| Case | U_ref / a_ref | rho_rms/(rho0 Ma_ref^2) | density mean offset / rms |",
        "|---|---:|---:|---:|"]
    for case in qualification.ALL_CASES:
        a = audit[case]
        lines.append(f"| {case} | {a['mach_from_references']:.5g} | "
                     f"{a['mach_scaled_density_rms']:.4g} | "
                     f"{a['density_center_offset_over_rms']:.3g} |")
    lines += ["", "## Descriptive W1 / pooled sigma by regime contrast", "",
              "| Diagnostic | Comparison category | Median normalized W1 |",
              "|---|---|---:|"]
    for field in DIAGNOSTIC_CHANNELS:
        for cat, val in summary["median_pairwise_normalized_w1_by_group"][field].items():
            lines.append(f"| {field} | {cat} | {val['median']:.4g} |")
    lines += ["", "All results are exploratory. Review marginal and spatial plots. "
              "The cutoff/temporal uncertainty and Re-grid confounding remain.", ""]
    (destination / "similarity_report.md").write_text("\n".join(lines))
    print(f"M43 similarity comparison complete: {destination}", flush=True)


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("inspect", "compare"), default="inspect")
    p.add_argument("--audit-root", type=Path, default=Path(
        "/scratch/coop/theret/GraphAttention_runs/m43_dns_stationarity_audit"))
    p.add_argument("--qualification-root", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/HIT_DNS_COMP_case_references"))
    p.add_argument("--state-reference-dir", type=Path, default=Path(
        "/scratch/coop/theret/HIT_DNS_COMP_similarity_references"))
    p.add_argument("--reviewed-cutoffs", type=Path)
    p.add_argument("--bootstrap-repeats", type=int, default=64)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--nyquist-fraction", type=float, default=0.70)
    return p.parse_args()


def main() -> None:
    args = _args()
    root = args.audit_root.expanduser().resolve(strict=True)
    qual = (args.qualification_root or root / "m43_qualification_v1").resolve()
    destination = (args.output_dir or root / VERSION).resolve()
    refs = args.reference_dir.expanduser().resolve(strict=True)
    sidecars = args.state_reference_dir.expanduser().resolve()
    if args.mode == "compare":
        if args.reviewed_cutoffs is None:
            raise ValueError("--reviewed-cutoffs must explicitly name the reviewed JSON")
    destination.mkdir(parents=True, exist_ok=True)
    if args.mode == "inspect":
        run_inspect(root, destination, refs, sidecars)
    else:
        run_compare(root, qual, destination, refs, sidecars,
                    args.reviewed_cutoffs.expanduser().resolve(strict=True),
                    repeats=args.bootstrap_repeats, seed=args.seed,
                    nyquist_fraction=args.nyquist_fraction)


if __name__ == "__main__":
    main()
