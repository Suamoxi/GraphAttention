"""Read-only adapter from the published MS_SR CEDAR campaign to GraphAttention.

MS_SR remains authoritative for source roles, HDF5 identities, physical
conservative components and its exact joint CEDAR inverse. This module calls
MS_SR's own CampaignDataset and TorchCEDAR; it never modifies the source
manifest or HDF5 files. All CEDAR operations act on complete periodic
fields, never on independently periodized training patches.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

FIELDS = ("rho", "rhou", "rhov", "rhow", "rhoE")
FILTERS = {"w1p25_l1": 1.25, "w1p5_l1": 1.5}
MANIFEST_SHA256 = "993f791f88d801e44e5feb0cee927c4c48eaa459bd6171e9e364b002bb01fdca"
CEDAR_REVISION = "ae96eece81204b7eb55365eade4838ae1e337b38"


@dataclass(frozen=True)
class MSRPair:
    coarse: torch.Tensor        # float64 [1,5,Nc,Nc,Nc], entire periodic domain
    fine: torch.Tensor          # float64 [1,5,Nf,Nf,Nf]
    baseline: torch.Tensor      # P(Uc) on the whole fine domain
    residual: torch.Tensor      # fine - P(Uc), C residual approximately zero
    metadata: dict
    manifest_sha256: str
    operator_sha256: str
    periodic_length: float
    refinement: int


def _import_from_file(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import MS_SR module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _add_external_sources(root: Path) -> None:
    """Use MS_SR's own code and a verified CEDAR source or installed package.

    A deployed MS_SR directory may have no cedar/src checkout because
    cedar-cfd is provided by the runtime Python environment. Validate the
    imported package against MS_SR's pinned file hashes in either case.
    """
    import os

    code = root / "code"
    if not code.is_dir():
        raise FileNotFoundError(f"required MS_SR source missing: {code}")
    code_path = str(code.resolve())
    if code_path not in sys.path:
        sys.path.insert(0, code_path)

    explicit = os.environ.get("M48_CEDAR_SRC")
    cedar = Path(explicit) if explicit else root / "cedar" / "src"
    if explicit and not cedar.is_dir():
        raise FileNotFoundError(f"M48_CEDAR_SRC does not exist: {cedar}")
    if cedar.is_dir():
        value = str(cedar.resolve())
        if value not in sys.path:
            sys.path.insert(0, value)

    try:
        import cedar
        from preprocess import check_cedar
    except ImportError as exc:
        raise ImportError(
            "CEDAR is unavailable in the M48 Python runtime. Supply the "
            "pinned source with M48_CEDAR_SRC, or use a Python environment "
            "containing the pinned cedar-cfd package."
        ) from exc
    check_cedar()  # MS_SR's audited source-file SHA-256 hashes.


def select_pilot_record(
    publication: dict, *, role: str = "teacher", filter_id: str = "w1p25_l1",
    fine_factor: int = 4, coarse_factor: int = 8, case_prefix: str = "ReT060/",
) -> dict:
    """Select one source *before* opening data; never substitute a test record."""
    if publication.get("schema") != "ms-sr.campaign.v1":
        raise ValueError("not an MS_SR campaign manifest")
    if role not in ("teacher", "validation"):
        raise ValueError("locked_in_regime is excluded from the M48 qualification/training pilot")
    if filter_id not in FILTERS or not (0 < fine_factor < coarse_factor):
        raise ValueError("invalid filter or fine/coarse source factors")
    records = [
        r for r in publication["pairs"]
        if r["role"] == role and r["source_role"] == role
        and r["filter_id"] == filter_id
        and int(r["fine_factor"]) == fine_factor
        and int(r["coarse_factor"]) == coarse_factor
        and r["case"].startswith(case_prefix)
    ]
    if not records:
        raise ValueError("no eligible MS_SR pair matches the requested role, case, filter and factors")
    records.sort(key=lambda r: (r["case"], float(r["physical_time_s"]), int(r["snapshot_id"])))
    return records[0]


def _isotropic_box(record: dict) -> tuple[float, int]:
    coarse_shape = tuple(record["input_shape"])
    fine_shape = tuple(record["target_shape"])
    if (len(coarse_shape) != 3 or len(fine_shape) != 3
            or len(set(coarse_shape)) != 1 or len(set(fine_shape)) != 1):
        raise ValueError("first M48 qualification supports cubic periodic fields only")
    n, nf = coarse_shape[0], fine_shape[0]
    if nf % n or nf // n < 2:
        raise ValueError("coarse/fine shape is not a valid integer refinement")
    lengths_c = [float(a) * b for a, b in zip(record["coarse_spacing"], coarse_shape)]
    lengths_f = [float(a) * b for a, b in zip(record["fine_spacing"], fine_shape)]
    length = lengths_c[0]
    if (not math.isfinite(length) or length <= 0
            or any(not math.isclose(v, length, rel_tol=1e-7, abs_tol=1e-12)
                   for v in lengths_c + lengths_f)):
        raise ValueError("M45 attention geometry currently requires a cubic, isotropic physical box")
    return length, nf // n


def flat_channels(tensor: torch.Tensor) -> torch.Tensor:
    """[1,5,Nx,Ny,Nz] -> [Nx*Ny*Nz,5] in M45 periodic xyz order."""
    if tensor.ndim != 5 or tensor.shape[:2] != (1, 5):
        raise ValueError("expected an MS_SR batch [1,5,Nx,Ny,Nz]")
    return tensor.permute(0, 2, 3, 4, 1).reshape(-1, 5).contiguous()


def component_max_relative(error: torch.Tensor, reference: torch.Tensor) -> dict:
    """Physical per-component absolute and scale-relative max norms."""
    if error.shape != reference.shape or error.ndim != 5 or error.shape[1] != 5:
        raise ValueError("expected matching full-field conservative tensors")
    errors = error.abs().amax(dim=(0, 2, 3, 4))
    scales = reference.abs().amax(dim=(0, 2, 3, 4)).clamp_min(1e-12)
    return {
        "absolute_linf": errors.tolist(),
        "component_scaled_linf": (errors / scales).tolist(),
        "max_component_scaled_linf": float((errors / scales).max()),
    }


def qualify_full_pair(
    coarse: torch.Tensor, fine: torch.Tensor, cedar, *,
    source_seed: int = 481001, tolerance: float = 1e-9,
) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Qualify C(PUc)=Uc, C(R)=0, exact field recovery, projected source."""
    if coarse.ndim != 5 or fine.ndim != 5 or coarse.shape[:2] != (1, 5) or fine.shape[:2] != (1, 5):
        raise ValueError("coarse/fine must be [1,5,X,Y,Z]")
    if coarse.dtype != torch.float64 or fine.dtype != torch.float64:
        raise TypeError("M48 full-field qualification requires float64")
    if not torch.isfinite(coarse).all() or not torch.isfinite(fine).all():
        raise ValueError("nonfinite real conservative field")
    baseline = cedar.prolong(coarse, tuple(fine.shape[-3:]))
    residual = fine - baseline
    coarse_from_fine = cedar.restrict(fine)
    checks = {
        "C_fine_minus_input": component_max_relative(coarse_from_fine - coarse, coarse),
        "CP_input_minus_input": component_max_relative(cedar.restrict(baseline) - coarse, coarse),
        "C_residual": component_max_relative(cedar.restrict(residual), coarse),
        "reconstruct_fine": component_max_relative(baseline + residual - fine, fine),
    }
    generator = torch.Generator(device="cpu").manual_seed(source_seed)
    noise = torch.randn(fine.shape, dtype=fine.dtype, generator=generator)
    projected = cedar.project(noise)
    checks["C_projected_source"] = component_max_relative(
        cedar.restrict(projected), coarse.new_ones(coarse.shape)
    )
    checks["projection_idempotence"] = component_max_relative(
        cedar.project(projected) - projected, fine.new_ones(fine.shape)
    )
    if any(value["max_component_scaled_linf"] > tolerance for value in checks.values()):
        raise RuntimeError(
            "M48 full-field CEDAR identities exceed component tolerance "
            f"{tolerance}: {checks}"
        )
    return baseline, residual, checks


def load_ms_sr_pair(
    ms_sr_root: Path, *, role: str = "teacher", filter_id: str = "w1p25_l1",
    fine_factor: int = 4, coarse_factor: int = 8,
    case_prefix: str = "ReT060/",
    expected_manifest_sha256: str = MANIFEST_SHA256,
) -> tuple[MSRPair, object, dict]:
    """Select one published pair and qualify it with MS_SR's TorchCEDAR.

    Calls MS_SR's CampaignDataset and own TorchCEDAR. The source HDF5 bytes
    and all other role records remain read-only. No new MS_SR data product.
    """
    root = ms_sr_root.resolve()
    manifest = root / "dataset" / "manifest.json"
    content = manifest.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    if digest != expected_manifest_sha256:
        raise ValueError(
            f"MS_SR manifest sha256 mismatch: {digest}; expected {expected_manifest_sha256}"
        )
    publication = json.loads(content)
    record = select_pilot_record(
        publication, role=role, filter_id=filter_id,
        fine_factor=fine_factor, coarse_factor=coarse_factor,
        case_prefix=case_prefix,
    )
    length, r = _isotropic_box(record)
    _add_external_sources(root)
    # Delegate provenance validation, HDF5 reads and role checking to MS_SR.
    from campaign_data import CampaignDataset, _load_records
    # Use the same canonical _load_records rather than parsing file paths
    # without source-role validation. Load only one eligible HDF5 pair.
    records, manifest_sha = _load_records(
        manifest, filters={filter_id}, cases={role: {record["case"]}},
        snapshot_ids={role: {str(record["snapshot_id"])}},
        pairs={(fine_factor, coarse_factor)},
    )
    selected = [
        r for r in records if r["role"] == role
        and r["case"] == record["case"] and str(r["snapshot_id"]) == str(record["snapshot_id"])
        and r["filter_id"] == filter_id
    ]
    if len(selected) != 1 or manifest_sha != digest:
        raise RuntimeError("MS_SR loader selected unexpected pair or manifest")
    pair = CampaignDataset(selected, partitions=1, halo=0)[0]
    coarse = pair["coarse"].unsqueeze(0).to(dtype=torch.float64)
    fine = pair["target"].unsqueeze(0).to(dtype=torch.float64)
    if tuple(coarse.shape[-3:]) != tuple(record["input_shape"]) or tuple(fine.shape[-3:]) != tuple(record["target_shape"]):
        raise RuntimeError("MS_SR reader returned inconsistent pair geometry")

    # Import original MS_SR source code; do not duplicate or approximate CEDAR.
    from cedar.plugins.nodal_operator_design.operators import NodalOperator
    operator = NodalOperator.builtin(
        2, family="optimized", omega=FILTERS[filter_id], lambda_=1,
    )
    stages = len(record["stages"])
    if stages != int(math.log2(r)) or 2**stages != r:
        raise ValueError("CEDAR composite stages do not match refinement ratio")
    expected_hash = record["operator_sha256"]
    actual_hash = operator.repeat(stages).hash
    if actual_hash != expected_hash or any(x != operator.hash for x in record["stages"]):
        raise ValueError("pinned MS_SR CEDAR operator hash mismatch")
    from pathlib import Path as _Path
    ms_models = _import_from_file("m48_ms_sr_original_models", root / "code" / "models.py")
    cedar = ms_models.TorchCEDAR(operator, stages=stages)
    baseline, residual, checks = qualify_full_pair(coarse, fine, cedar)
    return MSRPair(
        coarse=coarse, fine=fine, baseline=baseline, residual=residual,
        metadata=pair["metadata"], manifest_sha256=digest,
        operator_sha256=expected_hash, periodic_length=length, refinement=r,
    ), cedar, checks
