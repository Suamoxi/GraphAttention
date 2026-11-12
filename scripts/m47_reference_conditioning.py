"""Prepare M47 global physics conditions from audited case-level references.

No quantities are inferred from Re*/Ma* folder names, DNS snapshots, or
training-set statistics. The immutable M3.3 reference YAMLs are never edited.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path

from omegaconf import OmegaConf

from graph_attention.data.case_definition import load_case_definition
from graph_attention.data.multicase_avbp import DEFAULT_M43_CASES, reference_filename

VERSION = "m47_physics_conditions_v1"
CONDITION_NAMES = {
    "none": (),
    "mach": ("Ma_condition",),
    "mach_re": ("Ma_condition", "log_Re_condition"),
}
UNITS = {"a_ref": "m/s", "mu_ref": "kg/(m*s)", "nu_ref": "m^2/s"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _literal_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} requires an explicit numerical literal")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _load_mapping(path: Path) -> dict:
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected YAML mapping")
    return raw


def _read_audited_source(
    case_id: str, name: str, supplied: dict, root: Path,
) -> tuple[dict, dict]:
    required = {"value", "units", "definition", "provenance", "source_path"}
    if not isinstance(supplied, dict) or set(supplied) != required:
        raise ValueError(f"{case_id}.{name}: supply exactly {sorted(required)}")
    value = _literal_positive(supplied["value"], f"{case_id}.{name}")
    if supplied["units"] != UNITS[name]:
        raise ValueError(f"{case_id}.{name}: expected units {UNITS[name]!r}")
    for key in ("definition", "provenance", "source_path"):
        if (not isinstance(supplied[key], str) or not supplied[key].strip()
            or supplied[key].strip().upper().startswith("TODO")):
            raise ValueError(f"{case_id}.{name}: audited {key} is required")
    # A generic AVBP database can be supplied once and tracked inside the
    # repository. Other source paths retain their original per-case/absolute
    # meaning; do not require a nonexistent RUN/mixture_database.dat copy.
    source_spec = supplied["source_path"].strip()
    if source_spec.startswith("repo:"):
        relative = Path(source_spec.removeprefix("repo:"))
        if (not relative.parts or relative.is_absolute()
            or ".." in relative.parts):
            raise ValueError(f"{case_id}.{name}: invalid repository-relative source")
        repository = Path(__file__).resolve().parents[1]
        source = (repository / relative).resolve(strict=True)
        if not source.is_relative_to(repository):
            raise ValueError(f"{case_id}.{name}: source escapes repository")
    else:
        source = Path(source_spec).expanduser()
        if not source.is_absolute():
            source = root / case_id / source
        source = source.resolve(strict=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    evidence = {
        "source_path": str(source), "source_sha256": _sha256(source),
        "value": value, "provenance": supplied["provenance"],
        "verification": "source-file hash checked; numeric interpretation must be manually audited",
    }
    data = {
        "value": value, "units": supplied["units"],
        "definition": supplied["definition"],
        "provenance": f"{supplied['provenance']} [source sha256:{evidence['source_sha256']}]",
        "inference_available": True,
    }
    return data, evidence


def _existing_reference(name: str, path: Path) -> dict | None:
    raw = _load_mapping(path)
    ref = raw.get("references", {}).get(name)
    if ref is None:
        return None
    _literal_positive(ref["value"], f"{path}:{name}")
    if ref["units"] != UNITS[name]:
        raise ValueError(f"{path}: existing {name} units must be {UNITS[name]}")
    if ref["inference_available"] is not True or not ref.get("provenance"):
        raise ValueError(f"{path}: {name} unavailable at inference or missing provenance")
    return ref


def _regime(value: float, definition: str, derivation: str, provenance: str) -> dict:
    return {"value": float(value), "definition": definition,
            "provenance": provenance, "inference_available": True,
            "derivation": derivation}


def build_reference_cases(
    root: Path, references: Path, physical_input: Path | None,
) -> tuple[dict[str, dict], dict]:
    """Return fully validated sidecar YAML payloads and non-mutating audit report.

    Missing physical references are reported, not guessed. Existing a_ref or
    mu_ref/nu_ref in M3.3 YAMLs may be reused when genuinely declared.
    """
    raw_input = _load_mapping(physical_input) if physical_input and physical_input.is_file() else {}
    if raw_input and (raw_input.get("version") != VERSION or not isinstance(raw_input.get("cases"), dict)):
        raise ValueError("unsupported physical-input schema; expected m47_physics_conditions_v1")
    inputs = raw_input.get("cases", {})
    expected_ids = {case.case_id for case in DEFAULT_M43_CASES}
    if set(inputs) - expected_ids:
        raise ValueError(f"physical-input has unknown cases: {sorted(set(inputs)-expected_ids)}")
    prepared, observations, missing = {}, {}, {}
    schemes = set()
    for spec in DEFAULT_M43_CASES:
        case_id = spec.case_id
        base_path = references / reference_filename(case_id)
        base = load_case_definition(base_path)
        if base.case_id != case_id:
            raise ValueError(f"{base_path}: incorrect declared case ID")
        schemes.add(base.reference_scales.scheme)
        raw = _load_mapping(base_path)
        prescribed = inputs.get(case_id, {})
        if not isinstance(prescribed, dict) or set(prescribed) - set(UNITS):
            raise ValueError(f"{case_id}: invalid supplemental physical input")
        refs = dict(raw["references"])
        chosen, evidence, need = {}, {}, []
        for name in ("a_ref", "mu_ref", "nu_ref"):
            existing = _existing_reference(name, base_path)
            supplied = prescribed.get(name)
            if existing is not None:
                if supplied is not None:
                    other, proof = _read_audited_source(case_id, name, supplied, root)
                    if not math.isclose(existing["value"], other["value"], rel_tol=1e-10):
                        raise ValueError(f"{case_id}: conflicting original and supplemental {name}")
                    evidence[name] = proof
                chosen[name] = existing
                evidence.setdefault(name, {"original_reference_yaml": str(base_path),
                                           "original_reference_sha256": _sha256(base_path)})
            elif supplied is not None:
                chosen[name], evidence[name] = _read_audited_source(case_id, name, supplied, root)
        if "a_ref" not in chosen:
            need.append("a_ref")
        if ("mu_ref" in chosen) == ("nu_ref" in chosen):
            if not ("mu_ref" in chosen):
                need.append("mu_ref or nu_ref")
            else:
                raise ValueError(f"{case_id}: choose exactly one of mu_ref or nu_ref")
        if need:
            missing[case_id] = need
            candidates = {}
            for source in ("RUN/run.params", "RUN/mixture_database.dat",
                           "RUN/species_database.dat", "case_info.json"):
                path = root / case_id / source
                if path.is_file():
                    candidates[source] = {"path": str(path), "sha256": _sha256(path)}
            observations[case_id] = {
                "status": "missing", "missing": need,
                "source_candidates": candidates,
                "original_reference_sha256": _sha256(base_path),
            }
            continue
        rho = base.reference_scales["rho_ref"].value
        u = base.reference_scales["U_ref"].value
        length = base.reference_scales["L_ref"].value
        _literal_positive(rho, "rho_ref")
        _literal_positive(u, "U_ref")
        _literal_positive(length, "L_ref")
        mu = (chosen["mu_ref"]["value"] if "mu_ref" in chosen
              else rho * chosen["nu_ref"]["value"])
        re = rho * u * length / mu
        ma = u / chosen["a_ref"]["value"]
        if not math.isfinite(re) or not math.isfinite(ma) or re <= 0 or ma <= 0:
            raise ValueError(f"{case_id}: derived Re/Ma invalid")
        for name, obj in chosen.items():
            refs.setdefault(name, obj)
        # Existing descriptors must never silently be overwritten.
        regime = dict(raw.get("regime") or {})
        if set(regime) & {"Re_ref", "Ma_ref", "Ma_condition", "log_Re_condition"}:
            raise ValueError(f"{case_id}: derived descriptor already exists; manually reconcile")
        description = (f"M47 setup-derived from {base_path} sha256={_sha256(base_path)}; "
                       "physical source provenance in reference_audit.json")
        regime.update({
            "Re_ref": _regime(re, "rho_ref_U_ref_L_ref_over_mu_ref", "rho_ref * U_ref * L_ref / mu_ref", description),
            "Ma_ref": _regime(ma, "U_ref_over_a_ref", "U_ref / a_ref", description),
            "Ma_condition": _regime(ma / 0.1, "Ma_ref_divided_by_fixed_0.1", "Ma_ref / 0.1", description),
            "log_Re_condition": _regime(math.log(re), "natural_log_Re_ref", "ln(Re_ref)", description),
        })
        prepared[case_id] = {**raw, "references": refs, "regime": regime}
        observations[case_id] = {
            "status": "ready", "Re_ref": re, "Ma_ref": ma,
            "rho_ref": rho, "U_ref": u, "L_ref": length, "mu_ref": mu,
            "a_ref": chosen["a_ref"]["value"], "evidence": evidence,
            "original_reference_sha256": _sha256(base_path),
            "previous_regime_descriptors": {
                key: value.get("value") for key, value in (raw.get("regime") or {}).items()
            },
        }
    if len(schemes) != 1:
        raise ValueError(f"mixed physical reference schemes: {sorted(schemes)}")
    return prepared, {
        "version": VERSION, "status": "ready" if not missing else "missing_physical_inputs",
        "reference_scheme": next(iter(schemes)), "cases": observations,
        "missing": missing, "physical_input_file": str(physical_input) if physical_input else None,
        "physical_input_sha256": (_sha256(physical_input) if physical_input and physical_input.is_file() else None),
        "warning": "source checksums verify provenance, not the thermodynamic/viscosity calculation; audit source values manually",
    }


def _template() -> dict:
    return {"version": VERSION, "cases": {
        c.case_id: {
            "a_ref": {"value": None, "units": UNITS["a_ref"],
                      "definition": "reference_sound_speed_from_original_setup",
                      "provenance": "TODO: EOS and reference state", "source_path": "TODO: path in case"},
            "mu_ref": {"value": None, "units": UNITS["mu_ref"],
                       "definition": "molecular_dynamic_viscosity_at_reference_state",
                       "provenance": "TODO: molecular viscosity at reference state",
                       "source_path": "TODO: path in case"},
        } for c in DEFAULT_M43_CASES
    }}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("inspect", "prepare"))
    p.add_argument("--root", type=Path, default=Path("/archivenfs/phlusim/HIT_DNS_COMP"))
    p.add_argument("--reference-dir", type=Path, default=Path("/scratch/coop/theret/HIT_DNS_COMP_case_references"))
    p.add_argument("--physical-input", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    prepared, report = build_reference_cases(args.root, args.reference_dir, args.physical_input)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == "inspect":
        report_path = (args.output / "reference_inspection_latest.json" if
                       (args.output / "case_references").exists() else
                       args.output / "reference_audit.json")
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        template = args.output / "physical_inputs_TEMPLATE_NOT_VALIDATED.yaml"
        if not template.exists():
            OmegaConf.save(OmegaConf.create(_template()), template)
        print(json.dumps(report, indent=2), flush=True)
        return
    if report["missing"] or len(prepared) != len(DEFAULT_M43_CASES):
        (args.output / "reference_prepare_missing.json").write_text(
            json.dumps(report, indent=2) + "\n")
        raise ValueError("cannot prepare model references until all 6 physical conditions are audited")
    target = args.output / "case_references"
    if target.exists():
        raise FileExistsError(f"refusing overwrite of derived references: {target}")
    staging = args.output / "case_references.staging"
    if staging.exists():
        raise FileExistsError(f"incomplete prior staging directory: {staging}")
    staging.mkdir()
    try:
        generated_hashes = {}
        for case_id, payload in sorted(prepared.items()):
            path = staging / reference_filename(case_id)
            OmegaConf.save(OmegaConf.create(payload), path)
            verified = load_case_definition(path)
            if verified.case_id != case_id:
                raise AssertionError("generated case-definition mismatch")
            generated_hashes[case_id] = _sha256(path)
        staging.rename(target)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    report["generated_reference_hashes"] = generated_hashes
    report["generated_reference_dir"] = str(target.resolve())
    (args.output / "reference_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()