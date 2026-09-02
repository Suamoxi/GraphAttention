"""Explicit file-backed physical case definitions for M3.3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf

from .contracts import ReferenceScale, ReferenceScales, ReferenceScope

_ROOT_KEYS = {"case_id", "reference_scheme", "references"}
_REFERENCE_REQUIRED_KEYS = {
    "value",
    "units",
    "definition",
    "provenance",
    "inference_available",
}
_REFERENCE_OPTIONAL_KEYS = {"scope", "derivation"}


@dataclass(frozen=True, slots=True)
class CaseDefinition:
    """One declared physical case and its authoritative reference scales."""

    case_id: str
    reference_scales: ReferenceScales
    source_path: Path


def load_case_definition(path_value: str | Path) -> CaseDefinition:
    """Load one explicit YAML case definition without deriving reference values."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"case definition file does not exist: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"case definition '{path}' must contain a mapping at the document root")

    keys = set(raw)
    missing = _ROOT_KEYS - keys
    extra = keys - _ROOT_KEYS
    if missing:
        raise ValueError(f"case definition '{path}' is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"case definition '{path}' has unsupported keys: {sorted(extra)}")

    case_id = _required_text(raw["case_id"], "case_id", path)
    scheme = _required_text(raw["reference_scheme"], "reference_scheme", path)
    references = raw["references"]
    if not isinstance(references, Mapping) or not references:
        raise ValueError(f"case definition '{path}' references must be a non-empty mapping")

    scales: list[ReferenceScale] = []
    for raw_name, raw_reference in references.items():
        name = _required_text(raw_name, "reference name", path)
        if not isinstance(raw_reference, Mapping):
            raise TypeError(f"reference '{name}' in '{path}' must be a mapping")

        reference_keys = set(raw_reference)
        missing_reference_keys = _REFERENCE_REQUIRED_KEYS - reference_keys
        extra_reference_keys = reference_keys - _REFERENCE_REQUIRED_KEYS - _REFERENCE_OPTIONAL_KEYS
        if missing_reference_keys:
            raise ValueError(
                f"reference '{name}' in '{path}' is missing keys: "
                f"{sorted(missing_reference_keys)}"
            )
        if extra_reference_keys:
            raise ValueError(
                f"reference '{name}' in '{path}' has unsupported keys: "
                f"{sorted(extra_reference_keys)}"
            )

        value = raw_reference["value"]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"reference '{name}' in '{path}' value must be an explicit numeric literal"
            )
        inference_available = raw_reference["inference_available"]
        if not isinstance(inference_available, bool):
            raise TypeError(
                f"reference '{name}' in '{path}' inference_available must be a bool"
            )

        derivation = raw_reference.get("derivation")
        if derivation is not None:
            derivation = _required_text(derivation, f"reference '{name}' derivation", path)

        scales.append(
            ReferenceScale(
                name=name,
                value=float(value),
                units=_required_text(raw_reference["units"], f"reference '{name}' units", path),
                definition=_required_text(
                    raw_reference["definition"], f"reference '{name}' definition", path
                ),
                provenance=_required_text(
                    raw_reference["provenance"], f"reference '{name}' provenance", path
                ),
                scope=raw_reference.get("scope", ReferenceScope.CASE),
                inference_available=inference_available,
                derivation=derivation,
            )
        )

    return CaseDefinition(
        case_id=case_id,
        reference_scales=ReferenceScales(tuple(scales), scheme=scheme),
        source_path=path,
    )


def load_case_definitions(
    case_files: Mapping[str, str | Path] | None,
) -> dict[str, CaseDefinition]:
    """Load configured case files once and verify their declared identities."""

    if case_files is None:
        return {}
    if not isinstance(case_files, Mapping):
        raise TypeError("case_files must be a mapping from case_id to case-definition file")

    definitions: dict[str, CaseDefinition] = {}
    for raw_case_id, path_value in case_files.items():
        case_id = _required_text(raw_case_id, "case_files key", None)
        if not isinstance(path_value, (str, Path)):
            raise TypeError(f"case_files['{case_id}'] must be a filesystem path")
        definition = load_case_definition(path_value)
        if definition.case_id != case_id:
            raise ValueError(
                f"case_files key '{case_id}' does not match declared case_id "
                f"'{definition.case_id}' in '{definition.source_path}'"
            )
        definitions[case_id] = definition
    return definitions


def _required_text(value: object, name: str, path: Path | None) -> str:
    if not isinstance(value, str) or not value.strip():
        location = f" in '{path}'" if path is not None else ""
        raise ValueError(f"{name}{location} must be a non-empty string")
    return value
