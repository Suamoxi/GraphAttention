"""End-to-end M43 similarity cache test; synthetic fields and reference states."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import m43_qualify_compare as q
from scripts import m43_similarity_diagnostics as similarity


_CASES = ("Re54/Ma0.05", "Re94/Ma0.05")


def _reference_yaml(case: str, *, sidecar: bool) -> str:
    scales = {"rho0": (1.2, "kg/m^3"), "a_ref": (40.0, "m/s"),
              "e_ref": (80.0, "J/kg")} if sidecar else {
                  "rho_ref": (1.2, "kg/m^3"), "U_ref": (2.0, "m/s"),
                  "L_ref": (1.0, "m")}
    rows = [f"case_id: {case}", "reference_scheme: test_convective", "references:"]
    for key, (value, unit) in scales.items():
        rows += [f"  {key}:", f"    value: {value}", f"    units: {unit}",
                 "    definition: declared_from_synthetic_setup",
                 "    provenance: test_simulation_parameters", "    inference_available: true"]
    return "\n".join(rows) + "\n"


@pytest.fixture
def cache(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(q, "ALL_CASES", _CASES)
    audit = tmp_path / "audit"
    qual = tmp_path / "qual"
    ref_dir = tmp_path / "refs"
    state_dir = tmp_path / "additional_refs"
    ref_dir.mkdir()
    rng = np.random.default_rng(28)
    bounds = {case: {"transient_end_index": 9, "stationary_start_index": 40}
              for case in _CASES}
    for ci, case in enumerate(_CASES):
        (ref_dir / (case.replace("/", "_") + ".yaml")).write_text(
            _reference_yaml(case, sidecar=False))
        root = audit / "cases" / case.replace("/", "_")
        (root / "spectra").mkdir(parents=True)
        (root / "lag").mkdir()
        (root / "source_signature.json").write_text(json.dumps({
            "snapshots": [[f"solution_{i}.h5", 1234, 1111] for i in range(100)]}))
        history = []
        for i in range(100):
            history.append({"snapshot_index": i, "turbulent_k": 2 + .03*np.sin(i/7),
                            "enstrophy": 4 + .05*np.cos(i/5),
                            "std_rho": 0.01, "mean_rho": 1.2,
                            "u_rms_one_component": 1.0,
                            "f4_spectral_unresolved_energy_fraction": 0.04})
            if i < 40:
                continue
            wave = 2*np.pi*np.arange(1, 115)
            energy = np.exp(-wave/(70+8*ci)) * (1+.02*np.sin(i/5))
            np.savez_compressed(root / "spectra" / f"snapshot_{i:03d}.npz",
                                k=wave, E=energy)
            dx = 1/(64 if ci == 0 else 128)
            for factor in (2, 4, 8):
                r = np.arange(1, 9, dtype=float) * dx
                np.savez_compressed(root / "lag" / f"snapshot_{i:03d}_f{factor}.npz",
                    r=r, r_ll=np.exp(-20*r), s2_ll=(1 + .05*ci)*r**0.6,
                    s3_ll=-r, s4_ll=r**1.2)
        with (root / "time_history.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        saved = qual / "fields" / case.replace("/", "_")
        saved.mkdir(parents=True)
        for i in range(40, 100):
            rho_star = 1.0 + rng.normal(0, 0.005, 16)
            vel = rng.normal(0, 1, (16, 3))
            rhoe_star = rho_star * 20 + 0.5*np.sum(vel**2, axis=1)*rho_star
            x = np.column_stack((rho_star, rho_star[:, None]*vel,
                                 rhoe_star, vel))
            np.savez_compressed(saved / f"snapshot_{i:03d}.npz", values=x)
    old, _ = q._read_reference_scales(ref_dir)
    signature = q._sample_signature(audit, bounds, old, 16, 13)
    (qual / "fields" / "source_signature.json").write_text(json.dumps(signature))
    (qual / "qualification_summary.json").write_text(json.dumps({
        "reviewed_cutoffs_supplied": True,
        "cases": {case: {"classification_reviewed": True,
                          "reviewed_cutoffs": bounds[case]} for case in _CASES}}))
    cutoff = tmp_path / "reviewed.json"
    cutoff.write_text(json.dumps(bounds))
    return audit, qual, ref_dir, state_dir, cutoff, bounds


def test_inspect_generates_missing_templates_no_fabricated_values(cache, tmp_path):
    audit, _, refs, state, _, _ = cache
    out = tmp_path / "inspect"
    similarity.run_inspect(audit, out, refs, state)
    report = json.loads((out / "reference_audit.json").read_text())
    assert report["status"] == "missing_verified_thermodynamic_reference_values"
    template = (out / "reference_templates_NOT_VALIDATED" / "Re54_Ma0.05.yaml").read_text()
    assert "value: null" in template
    assert "REPLACE_WITH_VERIFIABLE" in template
    assert "case_id: Re54/Ma0.05" in template


def test_valid_state_comparison_uses_original_immutable_cache(cache, tmp_path):
    audit, qual, refs, state, cutoff, _ = cache
    state.mkdir()
    for case in _CASES:
        (state / (case.replace("/", "_") + ".yaml")).write_text(
            _reference_yaml(case, sidecar=True))
    ref_hashes = [(refs / (case.replace("/", "_") + ".yaml")).read_bytes()
                  for case in _CASES]
    out = tmp_path / "comparison"
    similarity.run_compare(audit, qual, out, refs, state, cutoff,
                           repeats=16, seed=16, nyquist_fraction=0.70)
    s = json.loads((out / "similarity_summary.json").read_text())
    assert s["spectral_support"]["kstar_max"] < np.pi * 64
    assert s["spectral_support"]["kstar_max"] == pytest.approx(.70*np.pi*64)
    assert s["reference_audit"][_CASES[0]]["mach_from_references"] == pytest.approx(.05)
    with (out / "pairwise_reference_state_wasserstein.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {r["field"] for r in rows} == {
        "density_fluctuation", "u", "v", "w", "energy_shifted"}
    assert len(rows) == 5
    with (out / "pairwise_matched_delta_residual.csv").open() as f:
        lag = list(csv.DictReader(f))
    assert lag[0]["status"] == "matched_filter_width"
    assert (int(lag[0]["factor_a"]), int(lag[0]["factor_b"])) == (2, 4)
    for case, contents in zip(_CASES, ref_hashes, strict=True):
        assert (refs / (case.replace("/", "_") + ".yaml")).read_bytes() == contents
    assert (out / "reference_state_one_point_pdfs.pdf").is_file()
    assert (out / "well_resolved_common_k_spectra.pdf").is_file()
    assert (out / "matched_delta_residual_S2.pdf").is_file()
    # Changed state-reference inputs cannot silently overwrite the existing analysis.
    sidecar = state / "Re54_Ma0.05.yaml"
    sidecar.write_text(sidecar.read_text().replace("e_ref:\n    value: 80.0",
                                                  "e_ref:\n    value: 90.0"))
    with pytest.raises(RuntimeError, match="incompatible"):
        similarity.run_compare(audit, qual, out, refs, state, cutoff,
                               repeats=16, seed=16, nyquist_fraction=0.70)


def test_refuse_missing_reference_value(cache, tmp_path):
    audit, qual, refs, state, cutoff, _ = cache
    with pytest.raises(ValueError, match="missing"):
        similarity.run_compare(audit, qual, tmp_path/"failed", refs, state, cutoff,
                               repeats=16, seed=16, nyquist_fraction=0.70)
