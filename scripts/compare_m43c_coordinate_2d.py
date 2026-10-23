"""Compare completed M43C absolute-vs-relative-only 2-D HIT experiments.

Reads only NEW experiment directories. Refuses comparisons with unequal
splits, standardizers or scientific configs; never reads/writes M43-B pilots.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _scalar_scaler_same(a: Path, b: Path) -> bool:
    x = torch.load(a, map_location="cpu", weights_only=True)
    y = torch.load(b, map_location="cpu", weights_only=True)
    if set(x) != set(y):
        return False
    for k in x:
        if isinstance(x[k], dict):
            if set(x[k]) != set(y[k]):
                return False
            for field, value in x[k].items():
                other = y[k][field]
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, other):
                        return False
                elif value != other:
                    return False
        elif x[k] != y[k]:
            return False
    return True


def _paths(root: Path, epochs: int, variant: str) -> dict[str, Path]:
    run = root / f"m43c_{variant}_s42_e{epochs}"
    generation = run / "generations" / "flow_heun_steps50_seed5678"
    benchmark = root / f"benchmark_{variant}" / generation.name
    return {"run": run, "generation": generation, "benchmark": benchmark}


def _paired_generation_arrays(paths: dict[str, dict[str, Path]]) -> dict[str, object]:
    """Verify both variants sampled the same test keys and reference states.

    This is an UNPAIRED generative-distribution comparison; deterministic
    sampling keys give the two models identical initial noise, but neither
    generated sample is paired to a particular real test snapshot.
    """
    data = {
        name: torch.load(
            paths[name]["generation"] / "generated_test.pt",
            map_location="cpu", weights_only=True,
        )
        for name in ("absolute", "relative")
    }
    old, new = data["absolute"], data["relative"]
    for key in (
        "generated_ids", "reference_ids", "node_counts",
        "channel_names", "sampling_steps", "sampling_method",
        "sampling_solver", "sampling_seed", "model_evaluations",
    ):
        if old[key] != new[key]:
            raise RuntimeError(
                f"generated test population or sampler differs in {key!r}"
            )
    if (
        int(old["sampling_steps"]) != 50
        or old["sampling_method"] != "flow_ode"
        or old["sampling_solver"] != "heun"
        or int(old["sampling_seed"]) != 5678
        or int(old["model_evaluations"]) != 100
    ):
        raise RuntimeError("generated population does not use requested Heun-50 protocol")
    if old.get("generated_reference_pairing") is not False:
        raise RuntimeError("generation is unexpectedly flagged as paired DNS targets")
    for name in ("absolute", "relative"):
        artifact = data[name]
        if not torch.isfinite(artifact["generated_nondimensional"]).all():
            raise RuntimeError(f"{name}: generated field contains NaN/Inf")
    if not torch.equal(
        old["target_nondimensional"], new["target_nondimensional"],
    ):
        raise RuntimeError("held-out test reference fields differ between variants")
    counts = tuple(int(n) for n in old["node_counts"])
    if not counts or len(set(counts)) != 1:
        raise ValueError("comparison expects a shared fixed-size 2-D mesh")
    count = len(counts)
    nodes = counts[0]
    channels = tuple(old["channel_names"])
    arrays = {
        name: data[name]["generated_nondimensional"].reshape(
            count, nodes, len(channels),
        ).numpy()
        for name in ("absolute", "relative")
    }
    reference = old["target_nondimensional"].reshape(
        count, nodes, len(channels),
    ).numpy()
    return {
        "generated": arrays, "reference": reference,
        "channel_names": channels, "num_samples": count,
        "nodes_per_sample": nodes,
    }


def _save_comparison_pdfs(
    output: Path, paired: dict[str, object],
    wasserstein: dict[str, list[dict[str, str]]],
) -> None:
    channels = paired["channel_names"]
    reference = paired["reference"]
    generated = paired["generated"]
    channel_indices = {channel: i for i, channel in enumerate(channels)}
    output_dir = output / "marginal_pdfs"
    output_dir.mkdir(exist_ok=False)
    for channel, index in channel_indices.items():
        fields = {
            "Test DNS slices": reference[:, :, index].reshape(-1),
            "Absolute coordinates": generated["absolute"][:, :, index].reshape(-1),
            "Relative geometry only": generated["relative"][:, :, index].reshape(-1),
        }
        minimum = min(float(np.min(x)) for x in fields.values())
        maximum = max(float(np.max(x)) for x in fields.values())
        if not maximum > minimum:
            maximum = minimum + 1.0
        bins = np.linspace(minimum, maximum, 161)
        centers = 0.5*(bins[:-1]+bins[1:])
        fig, ax = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
        for label, values in fields.items():
            pdf, _ = np.histogram(values, bins=bins, density=True)
            ax.plot(centers, pdf, label=label, linewidth=2.6)
        ax.set_title(f"Unpaired marginal comparison — {channel}", fontsize=16)
        ax.set_xlabel(f"Nondimensional {channel}", fontsize=14)
        ax.set_ylabel("Probability density", fontsize=14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=11)
        fig.savefig(output_dir / f"{channel}.pdf")
        plt.close(fig)

    variants = ("absolute", "relative")
    rows = {name: wasserstein[name] for name in variants}
    for name in variants:
        if [row["channel"] for row in rows[name]] != list(channels):
            raise RuntimeError(f"{name}: standardized Wasserstein channel order differs")
    idx = np.arange(len(channels))
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    width = .35
    for shift, name in ((-.5, "absolute"), (.5, "relative")):
        vals = [float(row["wasserstein_1_standardized"]) for row in rows[name]]
        ax.bar(idx+shift*width, vals, width=width,
               label="Absolute coordinates" if name == "absolute" else "Relative-only")
    ax.set_xticks(idx, labels=channels)
    ax.set_ylabel("Wasserstein-1 in train-standardized space", fontsize=13)
    ax.tick_params(labelsize=12)
    ax.grid(axis="y", alpha=.25)
    ax.legend(fontsize=11)
    fig.savefig(output / "wasserstein_comparison.pdf")
    plt.close(fig)


def compare(root: Path, epochs: int) -> dict:
    names = ("absolute", "relative")
    paths = {name: _paths(root, epochs, name) for name in names}
    for name in names:
        p = paths[name]
        for f in (
            p["run"] / "resolved_config.yaml",
            p["run"] / "summary.json",
            p["run"] / "history.csv",
            p["run"] / "standardizers.pt",
            p["run"] / "dataset_split_manifest.json",
            p["generation"] / "summary.json",
            p["generation"] / "generation_wall_seconds.txt",
            p["generation"] / "generated_test.pt",
            p["benchmark"] / "energy_spectrum_summary.csv",
            p["benchmark"] / "standardized_wasserstein.csv",
        ):
            if not f.is_file():
                raise FileNotFoundError(f"{name}: missing {f}")

    paired = _paired_generation_arrays(paths)
    a, b = paths["absolute"], paths["relative"]
    # Manifest includes runtime provenance (hardware, Python, checkout
    # state), which is NOT part of the scientific train/test split. Only
    # compare immutable source sample/group IDs, split membership and input
    # paths here; other scientific config and scaler checks follow.
    manifests = {
        name: _json(paths[name]["run"] / "dataset_split_manifest.json")
        for name in names
    }
    scientific_manifest_fields = (
        "group_metadata_key", "sample_ids", "group_ids",
        "train_ids", "validation_ids", "test_ids",
        "train_groups", "validation_groups", "test_groups",
        "sample_files", "shared_mesh_file", "case_definition_file",
    )
    for field in scientific_manifest_fields:
        if manifests["absolute"].get(field) != manifests["relative"].get(field):
            raise RuntimeError(
                f"scientific split or dataset field {field!r} differs"
            )
    if not _scalar_scaler_same(
        a["run"] / "standardizers.pt", b["run"] / "standardizers.pt"
    ):
        raise RuntimeError("train-only standardizers differ")
    model_cfgs = {}
    for name in names:
        cfg = OmegaConf.to_container(
            OmegaConf.load(paths[name]["run"] / "resolved_config.yaml"),
            resolve=True,
        )
        assert isinstance(cfg, dict)
        actual = cfg["model"]["use_absolute_position_encoding"]
        if actual is not (name == "absolute"):
            raise RuntimeError(f"{name}: incorrect coordinate flag in resolved config")
        cfg["model"]["use_absolute_position_encoding"] = True
        cfg["run_name"] = "__ablation__"
        model_cfgs[name] = cfg
    if model_cfgs["absolute"] != model_cfgs["relative"]:
        raise RuntimeError("other scientific configuration differs: refuse comparison")

    for name in names:
        gen = _json(paths[name]["generation"] / "summary.json")
        if (
            gen.get("source_run_name") != paths[name]["run"].name
            or gen.get("source_checkpoint") != "best.pt"
            or gen.get("reference_population") != "test"
            or gen.get("generated_reference_pairing") is not False
        ):
            raise RuntimeError(f"{name}: incorrect generation provenance")

    spectra = {n: _csv(paths[n]["benchmark"] / "energy_spectrum_summary.csv")
               for n in names}
    histories = {n: _csv(paths[n]["run"] / "history.csv") for n in names}
    k = np.array([float(row["k"]) for row in spectra["absolute"]])
    if not np.array_equal(
        k, np.array([float(row["k"]) for row in spectra["relative"]])
    ):
        raise RuntimeError("spectral wavenumber grids differ")
    ref_a = np.array([float(row["reference_mean"])
                      for row in spectra["absolute"]])
    ref_b = np.array([float(row["reference_mean"])
                      for row in spectra["relative"]])
    if not np.allclose(ref_a, ref_b, rtol=1.e-12, atol=1.e-15):
        raise RuntimeError("test-reference population energy spectra differ")

    out = root / f"coordinate_comparison_e{epochs}"
    if out.exists():
        raise FileExistsError(f"comparison output exists: {out}")
    out.mkdir(parents=True, exist_ok=False)

    wasserstein = {
        name: _csv(paths[name]["benchmark"] / "standardized_wasserstein.csv")
        for name in names
    }
    _save_comparison_pdfs(out, paired, wasserstein)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for name in names:
        hist = histories[name]
        ax.plot(
            [int(row["epoch"])+1 for row in hist],
            [float(row["validation_flow_velocity_mse"]) for row in hist],
            linewidth=2.5, label=f"{name}: validation MSE",
        )
    ax.set_xlabel("Completed epoch", fontsize=15)
    ax.set_ylabel("Flow Matching validation velocity MSE", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2)
    ax.legend(fontsize=12)
    fig.savefig(out / "validation_mse_comparison.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    ax.loglog(k, ref_a, label="Reference DNS slice population",
              linewidth=3, color="black")
    for name in names:
        spec = spectra[name]
        e = np.array([float(row["generated_mean"]) for row in spec])
        q10 = np.array([float(row["generated_q10"]) for row in spec])
        q90 = np.array([float(row["generated_q90"]) for row in spec])
        line = ax.loglog(k, e, linewidth=2.5, label=name)[0]
        ax.fill_between(k, q10, q90, alpha=0.16, color=line.get_color())
    ax.set_xlabel("Physical wavenumber k [rad / nondimensional length]", fontsize=15)
    ax.set_ylabel("Population velocity energy spectrum E(k)", fontsize=15)
    ax.tick_params(labelsize=12)
    ax.grid(alpha=0.2, which="both")
    ax.legend(fontsize=11)
    fig.savefig(out / "energy_spectrum_comparison.pdf")
    plt.close(fig)

    report = {
        "status": "paired_2d_coordinate_ablation_complete",
        "strict_checks": {
            "exact_test_train_split_match": True,
            "runtime_hardware_provenance_comparison": (
                "recorded for each arm; excluded from scientific split equality"
            ),
            "train_only_standardizers_identical": True,
            "all_scientific_config_except_position_flag_identical": True,
            "reference_energy_spectra_identical": True,
            "identical_sampling_keys_and_sampler": True,
            "identical_nondimensional_test_reference_fields": True,
        },
        "epochs": epochs,
        "num_samples": paired["num_samples"],
        "nodes_per_sample": paired["nodes_per_sample"],
        "channels": list(paired["channel_names"]),
        "comparison_pdfs": {
            "energy": "energy_spectrum_comparison.pdf",
            "marginals": "marginal_pdfs/",
            "wasserstein": "wasserstein_comparison.pdf",
            "validation": "validation_mse_comparison.pdf",
        },
        "variants": {},
        "interpretation": (
            "These are 2-D unconditional Flow Matching runs, not 3-D or "
            "LES-conditioned super-resolution. Faster timing does not "
            "establish better physical generalization to other geometries. "
            "The unused coordinate-MLP weights are intentionally retained "
            "for matched initialization."
        ),
    }
    for name in names:
        p = paths[name]
        summary = _json(p["run"] / "summary.json")
        generation = _json(p["generation"] / "summary.json")
        wall = float(
            (p["generation"] / "generation_wall_seconds.txt").read_text().strip()
        )
        generated = np.array([
            float(row["generated_mean"]) for row in spectra[name]
        ])
        positive = (k > 0) & (ref_a > 1.e-20) & (generated > 1.e-20)
        log_error = (
            float(np.sqrt(np.mean(
                np.square(np.log(generated[positive]/ref_a[positive]))
            )))
            if np.any(positive) else None
        )
        validation = [
            float(row["validation_flow_velocity_mse"]) for row in histories[name]
        ]
        report["variants"][name] = {
            "run_dir": str(p["run"]),
            "best_validation_flow_velocity_mse": float(min(validation)),
            "final_validation_flow_velocity_mse": float(validation[-1]),
            "test_velocity_mse_at_best": summary.get(
                "test_flow_velocity_mse_at_best_validation"
            ),
            "fit_wall_seconds": float(summary["fit_wall_seconds"]),
            "mean_epoch_compute_seconds": float(
                summary["mean_epoch_compute_seconds"]
            ),
            "generation_full_wall_seconds_including_io": wall,
            "generated_sample_count": generation["generation"][
                "num_generated_samples"
            ],
            "generation_model_evaluations": generation["model_evaluations"],
            "spectral_log_rms_error_positive_bins": log_error,
            "standardized_wasserstein_rows": wasserstein[name],
            "mean_standardized_wasserstein": float(np.mean([
                float(row["wasserstein_1_standardized"]) for row in wasserstein[name]
            ])),
            "benchmark_dir": str(p["benchmark"]),
        }
    baseline = report["variants"]["absolute"]
    variant = report["variants"]["relative"]
    report["relative_only_over_absolute"] = {
        key: variant[key]/baseline[key] if baseline[key] else None
        for key in (
            "fit_wall_seconds", "mean_epoch_compute_seconds",
            "generation_full_wall_seconds_including_io",
            "best_validation_flow_velocity_mse",
            "spectral_log_rms_error_positive_bins",
            "mean_standardized_wasserstein",
        )
    }
    (out / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    compare(args.root.expanduser().resolve(), args.epochs)


if __name__ == "__main__":
    main()
