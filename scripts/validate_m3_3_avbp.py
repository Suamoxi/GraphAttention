"""Validate M3.3 physical nondimensionalization on one real AVBP sample."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from graph_attention.data import (
    AVBPHDF5Dataset,
    AVBPSampleSpec,
    ConvectiveNondimensionalizer,
)

_FIELD_NAMES = ("rho", "rhou", "rhov", "rhow", "rhoE")


def _tensor_stats(tensor: torch.Tensor) -> tuple[float, float, float]:
    if tensor.numel() == 0:
        raise ValueError("real-data validation cannot summarize an empty tensor")
    return float(tensor.min()), float(tensor.max()), float(tensor.mean())


def _round_trip_metrics(
    original: torch.Tensor,
    restored: torch.Tensor,
) -> tuple[float, float]:
    error = (restored - original).abs()
    max_abs_error = float(error.max()) if error.numel() else 0.0
    scale = float(original.abs().max()) if original.numel() else 0.0
    scale_relative_error = max_abs_error / scale if scale > 0.0 else max_abs_error
    return max_abs_error, scale_relative_error


def _format_stats(tensor: torch.Tensor) -> str:
    minimum, maximum, mean = _tensor_stats(tensor)
    return f"min={minimum:.12e} max={maximum:.12e} mean={mean:.12e}"


def _validate_round_trip(
    name: str,
    original: torch.Tensor,
    restored: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> tuple[float, float]:
    max_abs_error, scale_relative_error = _round_trip_metrics(original, restored)
    if not torch.allclose(restored, original, rtol=rtol, atol=atol):
        raise RuntimeError(
            f"round-trip validation failed for {name}: "
            f"max_abs_error={max_abs_error:.12e}, "
            f"scale_relative_error={scale_relative_error:.12e}, "
            f"rtol={rtol:.3e}, atol={atol:.3e}"
        )
    return max_abs_error, scale_relative_error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the real-data M3.3 transform on one AVBP snapshot/mesh/case triple."
    )
    parser.add_argument("--snapshot", type=Path, required=True, help="AVBP solution HDF5 file")
    parser.add_argument("--mesh", type=Path, required=True, help="AVBP mesh HDF5 file")
    parser.add_argument("--case-file", type=Path, required=True, help="authoritative case YAML file")
    parser.add_argument("--case-id", default="HIT_LES_FORCED")
    parser.add_argument("--mesh-id", default="HIT_LES_FORCED")
    parser.add_argument("--sample-id", default="m3_3_real_validation")
    parser.add_argument(
        "--connectivity-indexing",
        choices=("auto", "zero", "one"),
        default="auto",
    )
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    parser.add_argument("--atol", type=float, default=1.0e-12)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.rtol < 0.0 or args.atol < 0.0:
        raise ValueError("rtol and atol must be non-negative")

    dataset = AVBPHDF5Dataset(
        samples=[
            AVBPSampleSpec(
                sample_id=args.sample_id,
                snapshot_file=args.snapshot,
                mesh_id=args.mesh_id,
                mesh_file=args.mesh,
                case_id=args.case_id,
            )
        ],
        case_files={args.case_id: args.case_file},
        field_names=_FIELD_NAMES,
        connectivity_indexing=args.connectivity_indexing,
    )
    sample = dataset[0]
    transform = ConvectiveNondimensionalizer(sample.reference_scales)

    dimensionless_fields = transform.nondimensionalize(sample.fields)
    restored_fields = transform.dimensionalize(dimensionless_fields)
    dimensionless_coords = transform.nondimensionalize_coordinates(sample.mesh.coords)
    restored_coords = transform.dimensionalize_coordinates(dimensionless_coords)

    print("M3.3 real AVBP validation")
    print(f"sample_id: {sample.sample_id}")
    print(f"case_id: {sample.case_id}")
    print(f"mesh_id: {sample.mesh.mesh_id}")
    print(f"num_nodes: {sample.mesh.num_nodes}")
    print(f"spatial_dim: {sample.mesh.spatial_dim}")
    if sample.mesh.cell_connectivity is not None:
        print(f"num_cells: {sample.mesh.cell_connectivity.shape[0]}")
    print(f"reference_scheme: {sample.reference_scales.scheme}")

    print("\nReference scales:")
    for reference in sample.reference_scales.scales:
        print(
            f"  {reference.name}: value={reference.value:.12e} units={reference.units} "
            f"definition={reference.definition}"
        )

    print("\nRegime parameters:")
    if sample.regime_parameters.parameters:
        for parameter in sample.regime_parameters.parameters:
            print(
                f"  {parameter.name}: value={parameter.value:.12e} "
                f"definition={parameter.definition}"
            )
    else:
        print("  none")

    print("\nFields:")
    for name in _FIELD_NAMES:
        original = sample.fields[name]
        dimensionless = dimensionless_fields[name]
        max_abs_error, scale_relative_error = _validate_round_trip(
            name,
            original,
            restored_fields[name],
            rtol=args.rtol,
            atol=args.atol,
        )
        print(f"  {name}")
        print(f"    dimensional:   {_format_stats(original)}")
        print(f"    dimensionless: {_format_stats(dimensionless)}")
        print(
            f"    round_trip: max_abs={max_abs_error:.12e} "
            f"scale_relative={scale_relative_error:.12e}"
        )

    print("\nCoordinates:")
    for axis in range(sample.mesh.spatial_dim):
        original = sample.mesh.coords[:, axis]
        dimensionless = dimensionless_coords[:, axis]
        print(f"  axis_{axis}")
        print(f"    dimensional:   {_format_stats(original)}")
        print(f"    dimensionless: {_format_stats(dimensionless)}")
        print(f"    dimensionless_span: {float(dimensionless.max() - dimensionless.min()):.12e}")

    max_abs_error, scale_relative_error = _validate_round_trip(
        "coordinates",
        sample.mesh.coords,
        restored_coords,
        rtol=args.rtol,
        atol=args.atol,
    )
    print(
        f"  round_trip: max_abs={max_abs_error:.12e} "
        f"scale_relative={scale_relative_error:.12e}"
    )
    print("\nRESULT: PASS")


if __name__ == "__main__":
    main()
