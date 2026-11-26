"""M47.9B read-only periodic-boundary audit of archived HIT_LES_FORCED fields.

No retraining, resampling, periodification, clipping, or projection. Unlike an
FFT-only audit, tests both duplicate stored periodic nodes and the one-cell
wrap on the unique Fourier grid. Uses the same centered div/curl stencil as
M47.6 to measure how the divergence defect is spatially localized.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess

import numpy as np
from omegaconf import OmegaConf

from scripts.m47_9b_boundary_metrics import (
    AXES, divergence_localization, face_jump_statistics,
)

PROTOCOL = "m47_9b_hit_les_forced_boundary_localization_v1"
CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")
FACE_FIELDS = (
    "stored_duplicate_endpoint_rms", "periodic_one_cell_wrap_rms",
    "ordinary_one_cell_interior_rms", "face_adjacent_one_cell_rms",
    "duplicate_over_interior", "duplicate_over_face_adjacent",
    "wrap_over_interior", "wrap_over_face_adjacent",
)


def _statistics(values):
    a = np.asarray([x for x in values if x is not None], dtype=np.float64)
    a = a[np.isfinite(a)]
    return {
        "count": int(a.size),
        "mean": float(np.mean(a)) if a.size else None,
        "std": float(np.std(a, ddof=1)) if a.size > 1 else None,
        "q10": float(np.quantile(a, .1)) if a.size else None,
        "q90": float(np.quantile(a, .9)) if a.size else None,
    }


def _population(samples, ids, channels, grid, u_ref, core_trim):
    """Audit actual saved (n+1)^3 conservative states in original node order."""
    from scripts.m47_6_spatial_metrics import interior_divcurl

    if samples.ndim != 3 or len(samples) != len(ids):
        raise ValueError("population samples and IDs are not aligned")
    bases = tuple(name.split(".")[0] for name in channels)
    if len(bases) != 5 or set(bases) != set(CHANNELS):
        raise ValueError("expected five conservative channels exactly once")
    order = [bases.index(name) for name in CHANNELS]
    expected_nodes = int(np.prod(grid.source_shape))
    if samples.shape[1:] != (expected_nodes, len(order)):
        raise ValueError("source node count/channel count differs from saved archive")
    if grid.periodic_endpoint_mode != "drop_max" or any(
        s != n + 1 for s, n in zip(grid.source_shape, grid.shape)
    ):
        raise ValueError("requires stored duplicate periodic endpoints")

    records = []
    failures = []
    for sample_id, state in zip(ids, samples, strict=True):
        q = np.asarray(state[:, order], dtype=np.float64)
        if not np.isfinite(q).all() or np.any(q[:, 0] <= 0):
            failures.append({
                "sample_id": sample_id,
                "nonfinite_fraction": float(np.mean(~np.isfinite(q))),
                "nonpositive_density_fraction": float(np.mean(q[:, 0] <= 0)),
            })
            continue
        reordered = np.empty_like(q)
        reordered[grid.linear_indices] = q
        full_state = reordered.reshape((*grid.source_shape, 5))
        full_velocity = u_ref * full_state[..., 1:4] / full_state[..., :1]
        velocity_faces = face_jump_statistics(full_velocity, velocity_vector=True)
        conservative_faces = {
            name: face_jump_statistics(full_state[..., i:i + 1])
            for i, name in enumerate(CHANNELS)
        }
        unique_velocity = full_velocity[:-1, :-1, :-1]
        depth = divergence_localization(
            unique_velocity, grid.spacing, core_trim=core_trim,
        )
        if not records:
            old = interior_divcurl(
                unique_velocity, grid.spacing, trims=(0, core_trim),
            )
            for trim, new_value in (
                (0, depth["full_divergence_to_vorticity_rms"]),
                (core_trim, depth["core_divergence_to_vorticity_rms"]),
            ):
                old_value = old[str(trim)]["divergence_to_vorticity_rms"]
                if old_value is None or new_value is None or not np.isclose(
                    old_value, new_value, rtol=1e-10, atol=1e-12,
                ):
                    raise ValueError("new divergence stencil differs from M47.6")
        records.append({
            "sample_id": str(sample_id),
            "velocity_faces": velocity_faces,
            "conservative_faces": conservative_faces,
            "divergence": depth,
        })
    if not records:
        raise ValueError("no valid positive-density fields for boundary audit")

    velocity_axes = {
        axis: {
            name: _statistics(r["velocity_faces"][axis][name] for r in records)
            for name in FACE_FIELDS
        }
        for axis in AXES
    }
    normal_axes = {
        axis: {
            name: _statistics(
                r["velocity_faces"][axis]["normal_component"][name]
                for r in records
            )
            for name in (
                "stored_duplicate_endpoint_rms", "periodic_one_cell_wrap_rms",
                "ordinary_one_cell_interior_rms", "duplicate_over_interior",
                "wrap_over_interior",
            )
        }
        for axis in AXES
    }
    channel_axes = {
        ch: {
            axis: {
                name: _statistics(
                    r["conservative_faces"][ch][axis][name]
                    for r in records
                )
                for name in (
                    "stored_duplicate_endpoint_rms", "duplicate_over_interior",
                    "wrap_over_interior",
                )
            }
            for axis in AXES
        }
        for ch in CHANNELS
    }
    divergence_stats = {
        name: _statistics(r["divergence"][name] for r in records)
        for name in (
            "full_divergence_rms_s_minus_1", "full_vorticity_rms_s_minus_1",
            "full_divergence_to_vorticity_rms", "core_divergence_to_vorticity_rms",
            "outer_shell_divergence_energy_fraction",
        )
    }
    shells = {}
    for shell in [str(i) for i in range(core_trim)] + [f"{core_trim}_plus"]:
        shells[shell] = {
            name: _statistics(
                r["divergence"]["shells"][shell][name]
                for r in records
            )
            for name in (
                "voxel_fraction", "divergence_energy_fraction",
                "vorticity_energy_fraction", "divergence_enrichment_over_uniform",
                "divergence_to_vorticity_rms",
            )
        }
    profiles = {}
    for axis in AXES:
        profiles[axis] = {}
        for key in (
            "divergence_enrichment_over_uniform", "vorticity_enrichment_over_uniform",
        ):
            curves = [r["divergence"]["axis_profiles"][axis][key]
                      for r in records]
            good = np.asarray([v for v in curves if v is not None], dtype=float)
            profiles[axis][key] = {
                "mean": np.mean(good, axis=0).tolist() if good.size else None,
                "q10": np.quantile(good, .1, axis=0).tolist() if good.size else None,
                "q90": np.quantile(good, .9, axis=0).tolist() if good.size else None,
            }

    return {
        "requested_count": len(ids),
        "valid_count": len(records),
        "invalid_count": len(failures),
        "invalid_fields": failures,
        "summary": {
            "velocity_faces_by_axis": velocity_axes,
            "velocity_normal_component_by_axis": normal_axes,
            "conservative_faces_by_channel_axis": channel_axes,
            "divergence": divergence_stats,
            "boundary_depth_shells": shells,
            "divergence_axis_profiles": profiles,
        },
        "per_sample": records,
    }


def _csv_rows(populations, directory, core_trim):
    with (directory / "per_field.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "population", "sample_id", "divcurl_full", "divcurl_core",
            "divergence_outer_shell_fraction", "boundary_shell0_enrichment",
        ))
        writer.writeheader()
        for label, pop in populations.items():
            for r in pop["per_sample"]:
                d = r["divergence"]
                writer.writerow({
                    "population": label, "sample_id": r["sample_id"],
                    "divcurl_full": d["full_divergence_to_vorticity_rms"],
                    "divcurl_core": d["core_divergence_to_vorticity_rms"],
                    "divergence_outer_shell_fraction": d["outer_shell_divergence_energy_fraction"],
                    "boundary_shell0_enrichment": d["shells"]["0"]["divergence_enrichment_over_uniform"],
                })
    with (directory / "per_axis.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "population", "sample_id", "axis", "endpoint_rms_m_per_s",
            "wrap_rms_m_per_s", "interior_rms_m_per_s", "face_adjacent_rms_m_per_s",
            "endpoint_over_interior", "endpoint_over_face_adjacent",
            "wrap_over_interior", "wrap_over_face_adjacent",
            "normal_wrap_over_interior",
        ))
        writer.writeheader()
        for label, pop in populations.items():
            for r in pop["per_sample"]:
                for axis in AXES:
                    v = r["velocity_faces"][axis]
                    writer.writerow({
                        "population": label, "sample_id": r["sample_id"],
                        "axis": axis,
                        "endpoint_rms_m_per_s": v["stored_duplicate_endpoint_rms"],
                        "wrap_rms_m_per_s": v["periodic_one_cell_wrap_rms"],
                        "interior_rms_m_per_s": v["ordinary_one_cell_interior_rms"],
                        "face_adjacent_rms_m_per_s": v["face_adjacent_one_cell_rms"],
                        "endpoint_over_interior": v["duplicate_over_interior"],
                        "endpoint_over_face_adjacent": v["duplicate_over_face_adjacent"],
                        "wrap_over_interior": v["wrap_over_interior"],
                        "wrap_over_face_adjacent": v["wrap_over_face_adjacent"],
                        "normal_wrap_over_interior": v["normal_component"]["wrap_over_interior"],
                    })
    with (directory / "per_shell.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "population", "sample_id", "shell", "voxel_fraction",
            "divergence_energy_fraction", "vorticity_energy_fraction",
            "divergence_enrichment", "divcurl_rms",
        ))
        writer.writeheader()
        for label, pop in populations.items():
            for r in pop["per_sample"]:
                for shell, values in r["divergence"]["shells"].items():
                    writer.writerow({
                        "population": label, "sample_id": r["sample_id"],
                        "shell": shell, "voxel_fraction": values["voxel_fraction"],
                        "divergence_energy_fraction": values["divergence_energy_fraction"],
                        "vorticity_energy_fraction": values["vorticity_energy_fraction"],
                        "divergence_enrichment": values["divergence_enrichment_over_uniform"],
                        "divcurl_rms": values["divergence_to_vorticity_rms"],
                    })


def _plot_summaries(populations, outdir, core_trim):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    labels = tuple(populations)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    width = .78 / len(labels)
    x = np.arange(3)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for i, label in enumerate(labels):
        summary = populations[label]["summary"]["velocity_faces_by_axis"]
        position = x + (i - (len(labels) - 1) / 2) * width
        for ax, key in zip(axs, ("duplicate_over_face_adjacent", "wrap_over_face_adjacent")):
            values = [summary[a][key]["mean"] for a in AXES]
            ax.bar(position, [np.nan if v is None else v for v in values],
                   width=width, label=label, color=colors[i % len(colors)])
    for ax, title in zip(axs, (
        "Equal-coordinate duplicated endpoints", "Periodic one-cell wrap",
    )):
        ax.set_title(title)
        ax.set_xticks(x, AXES)
        ax.set_xlabel("Axis")
        ax.set_ylabel("RMS jump / adjacent face one-cell RMS")
        ax.grid(axis="y", alpha=.25)
    axs[1].legend()
    fig.savefig(outdir / "face_mismatch.png", dpi=160)
    plt.close(fig)

    depth = list(range(core_trim + 1))
    names = [str(i) for i in range(core_trim)] + [f"{core_trim}_plus"]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for i, label in enumerate(labels):
        data = populations[label]["summary"]["boundary_depth_shells"]
        mean = np.array([data[n]["divergence_enrichment_over_uniform"]["mean"] for n in names])
        low = np.array([data[n]["divergence_enrichment_over_uniform"]["q10"] for n in names])
        high = np.array([data[n]["divergence_enrichment_over_uniform"]["q90"] for n in names])
        ax.plot(depth, mean, "o-", label=label, color=colors[i % len(colors)])
        ax.fill_between(depth, low, high, alpha=.12, color=colors[i % len(colors)])
    ax.axhline(1, ls="--", color="gray", linewidth=1)
    ax.set_xticks(depth, names)
    ax.set_xlabel("Minimum distance to any unique-grid face (cells)")
    ax.set_ylabel("Mean div(u)^2 / full-volume mean div(u)^2")
    ax.set_title("Boundary localization of divergence (q10-q90 across fields)")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(outdir / "divergence_by_depth.png", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, axis in zip(axs, AXES):
        for i, label in enumerate(labels):
            p = populations[label]["summary"]["divergence_axis_profiles"][axis]
            mean = p["divergence_enrichment_over_uniform"]["mean"]
            if mean is not None:
                ax.plot(np.arange(len(mean)), mean, label=label,
                        color=colors[i % len(colors)])
        ax.axhline(1, ls="--", color="gray", linewidth=1)
        ax.set_title(f"Axis {axis}")
        ax.set_xlabel("Unique-grid index")
        ax.grid(alpha=.2)
    axs[0].set_ylabel("Mean div(u)^2 / full-volume mean div(u)^2")
    axs[-1].legend()
    fig.savefig(outdir / "divergence_slice_profiles.png", dpi=160)
    plt.close(fig)


def run(args):
    from scripts.m47_9_hit_les_forced_helmholtz import (
        _fingerprint, _generation, _load, _mesh_signature,
    )

    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory {args.output_dir}")
    generations = dict(_generation(spec) for spec in args.generation)
    if len(generations) != len(args.generation) or not {"M37", "M38"}.issubset(generations):
        raise ValueError("provide unique M37 and M38 archived generation paths")
    cfg = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "configs/benchmark_generation_3d.yaml"
    )
    case = OmegaConf.load(
        Path(__file__).resolve().parents[1] / "cases/HIT_LES_FORCED.yaml"
    )
    if str(cfg.spectra.periodic_endpoint_mode) != "drop_max" or case.case_id != "HIT_LES_FORCED":
        raise ValueError("expected HIT_LES_FORCED 33^3 -> 32^3 periodic convention")
    u_ref = float(case.references.U_ref.value)
    if not np.isfinite(u_ref) or u_ref <= 0:
        raise ValueError("invalid physical velocity reference")

    result = {
        "protocol": PROTOCOL,
        "status": "complete",
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "case": "HIT_LES_FORCED",
        "U_ref_m_per_s": u_ref,
        "trim_cells_per_face": args.core_trim,
        "periodic_endpoint_mode": "drop_max",
        "definitions": {
            "duplicate_endpoint": "u_source[x=0] versus u_source[x=n] at identical coordinates (n+1)^3 stored mesh; should match",
            "one_cell_wrap": "u_unique[x=0] minus u_unique[x=n-1] separated by one periodic cell; should NOT generally vanish",
            "interior_one_cell": "all ordinary differences u[i+1]-u[i] for i=0..n-2",
            "face_adjacent_one_cell": "first and last ordinary one-cell differences on each axis",
            "boundary_depth": "min distance in cell indices to any face of unique 32^3 grid; disjoint shells 0..trim-1 and core>=trim",
            "divergence_energy_fraction": "sum(div(u)^2) over shell divided by full-domain sum(div(u)^2)",
            "divergence_enrichment": "shell divergence-energy fraction divided by shell voxel fraction",
        },
        "caveats": (
            "HIT_LES_FORCED reference is LES, not DNS. Centered derivatives and FFT use "
            "periodicity. Duplicated endpoint mismatches and periodic wrap increments "
            "are distinct. The wrap is one cell apart and need not vanish. "
            "High boundary localization does not by itself prove a missing graph edge. "
            "Generated/reference populations are unpaired."
        ),
        "generation_paths": {k: str(v) for k, v in generations.items()},
        "populations": {},
    }
    reference_ids0 = None
    reference_hash0 = None
    signature0 = None
    for label in ("M37", "M38", "M40"):
        if label not in generations:
            continue
        (generated, reference, generated_ids, reference_ids, channels,
         grid, mesh) = _load(generations[label], cfg)
        signature = _mesh_signature(grid, mesh)
        checksum = _fingerprint(reference)
        if reference_ids0 is None:
            reference_ids0 = reference_ids
            reference_hash0 = checksum
            signature0 = signature
            result["reference_ids"] = list(reference_ids)
            result["reference_field_sha256"] = checksum
            result["mesh"] = signature
            result["populations"]["reference"] = _population(
                reference, reference_ids, channels, grid, u_ref, args.core_trim,
            )
        elif reference_ids != reference_ids0 or checksum != reference_hash0 or signature != signature0:
            raise ValueError(f"{label} does not share identical held-out reference/mesh with M37")
        result["populations"][label] = _population(
            generated, generated_ids, channels, grid, u_ref, args.core_trim,
        )
        out = result["populations"][label]
        s = out["summary"]
        print(
            f"[M47.9B] {label}: valid={out['valid_count']}, "
            f"divcurl_full={s['divergence']['full_divergence_to_vorticity_rms']['mean']:.6g}, "
            f"divcurl_core={s['divergence']['core_divergence_to_vorticity_rms']['mean']:.6g}",
            flush=True,
        )
        del generated, reference

    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "audit.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    _csv_rows(result["populations"], args.output_dir, args.core_trim)
    if not args.no_plots:
        _plot_summaries(result["populations"], args.output_dir / "plots", args.core_trim)
    print(f"[M47.9B] COMPLETE: {args.output_dir}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", action="append", required=True,
                        help="M37|M38|M40=/absolute/path/to/completed/generation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--core-trim", type=int, default=4)
    parser.add_argument("--no-plots", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
