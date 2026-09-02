"""Explicit file-backed physical case definitions for M3.3."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from .contracts import (
    ReferenceScale,
    ReferenceScales,
    ReferenceScope,
    RegimeParameter,
    RegimeParameters,
)

_ROOT_REQUIRED_KEYS = {"case_id", "reference_scheme", "references"}
_ROOT_OPTIONAL_KEYS = {"regime"}
_REFERENCE_REQUIRED_KEYS = {
    "value",
    "units",
    "definition",
    "provenance",
    "inference_available",
}
_REFERENCE_OPTIONAL_KEYS = {"scope", "derivation"}
_REGIME_REQUIRED_KEYS = {"value", "definition", "provenance", "inference_available"}
_REGIME_OPTIONAL_KEYS = {"scope", "derivation"}


@dataclass(frozen=True, slots=True)
class CaseDefinition:
    """One declared physical case and its authoritative physical descriptors."""

    case_id: str
    reference_scales: ReferenceScales
    source_path: Path
    regime_parameters: RegimeParameters = field(default_factory=RegimeParameters)


def load_case_definition(path_value: str | Path) -> CaseDefinition:
    """Load one explicit YAML case definition without deriving physical values."""

    path = Path(path_value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"case definition file does not exist: {path}")

    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
    if not isinstance(raw, Mapping):
        raise TypeError(f"case definition '{path}' must contain a mapping at the document root")

    keys = set(raw)
    missing = _ROOT_REQUIRED_KEYS - keys
    extra = keys - _ROOT_REQUIRED_KEYS - _ROOT_OPTIONAL_KEYS
    if missing:
        raise ValueError(f"case definition '{path}' is missing keys: {sorted(missing)}")
    if extra:
        raise ValueError(f"case definition '{path}' has unsupported keys: {sorted(extra)}")

    case_id = _required_text(raw["case_id"], "case_id", path)
    scheme = _required_text(raw["reference_scheme"], "reference_scheme", path)
    reference_scales = _load_reference_scales(raw["references"], scheme, path)
    regime_parameters = _load_regime_parameters(raw.get("regime"), path)

    return CaseDefinition(
        case_id=case_id,
        reference_scales=reference_scales,
        source_path=path,
        regime_parameters=regime_parameters,
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


def _load_reference_scales(
    raw_references: object,
    scheme: str,
    path: Path,
) -> ReferenceScales:
    if not isinstance(raw_references, Mapping) or not raw_references:
        raise ValueError(f"case definition '{path}' references must be a non-empty mapping")

    scales: list[ReferenceScale] = []
    for raw_name, raw_reference in raw_references.items():
        name = _required_text(raw_name, "reference name", path)
        if not isinstance(raw_reference, Mapping):
            raise TypeError(f"reference '{name}' in '{path}' must be a mapping")

        keys = set(raw_reference)
        missing = _REFERENCE_REQUIRED_KEYS - keys
        extra = keys - _REFERENCE_REQUIRED_KEYS - _REFERENCE_OPTIONAL_KEYS
        if missing:
            raise ValueError(f"reference '{name}' in '{path}' is missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"reference '{name}' in '{path}' has unsupported keys: {sorted(extra)}")

        inference_available = raw_reference["inference_available"]
        if not isinstance(inference_available, bool):
            raise TypeError(f"reference '{name}' in '{path}' inference_available must be a bool")
        derivation = _optional_text(
            raw_reference.get("derivation"),
            f"reference '{name}' derivation",
            path,
        )
        scales.append(
            ReferenceScale(
                name=name,
                value=_explicit_numeric_literal(raw_reference["value"], "reference", name, path),
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
    return ReferenceScales(tuple(scales), scheme=scheme)


def _load_regime_parameters(raw_regime: object, path: Path) -> RegimeParameters:
    if raw_regime is None:
        return RegimeParameters()
    if not isinstance(raw_regime, Mapping):
        raise TypeError(f"case definition '{path}' regime must be a mapping")

    parameters: list[RegimeParameter] = []
    for raw_name, raw_parameter in raw_regime.items():
        name = _required_text(raw_name, "regime parameter name", path)
        if not isinstance(raw_parameter, Mapping):
            raise TypeError(f"regime parameter '{name}' in '{path}' must be a mapping")

        keys = set(raw_parameter)
        missing = _REGIME_REQUIRED_KEYS - keys
        extra = keys - _REGIME_REQUIRED_KEYS - _REGIME_OPTIONAL_KEYS
        if missing:
            raise ValueError(
                f"regime parameter '{name}' in '{path}' is missing keys: {sorted(missing)}"
            )
        if extra:
            raise ValueError(
                f"regime parameter '{name}' in '{path}' has unsupported keys: {sorted(extra)}"
            )

        inference_available = raw_parameter["inference_available"]
        if not isinstance(inference_available, bool):
            raise TypeError(
                f"regime parameter '{name}' in '{path}' inference_available must be a bool"
            )
        derivation = _optional_text(
            raw_parameter.get("derivation"),
            f"regime parameter '{name}' derivation",
            path,
        )
        parameters.append(
            RegimeParameter(
                name=name,
                value=_explicit_numeric_literal(
                    raw_parameter["value"], "regime parameter", name, path
                ),
                definition=_required_text(
                    raw_parameter["definition"], f"regime parameter '{name}' definition", path
                ),
                provenance=_required_text(
                    raw_parameter["provenance"], f"regime parameter '{name}' provenance", path
                ),
                scope=raw_parameter.get("scope", ReferenceScope.CASE),
                inference_available=inference_available,
                derivation=derivation,
            )
        )
    return RegimeParameters(tuple(parameters))


def _explicit_numeric_literal(value: object, kind: str, name: str, path: Path) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{kind} '{name}' in '{path}' value must be an explicit numeric literal")
    return float(value)


def _optional_text(value: object, name: str, path: Path) -> str | None:
    if value is None:
        return None
    return _required_text(value, name, path)


def _required_text(value: object, name: str, path: Path | None) -> str:
    if not isinstance(value, str) or not value.strip():
        location = f" in '{path}'" if path is not None else ""
        raise ValueError(f"{name}{location} must be a non-empty string")
    return value
