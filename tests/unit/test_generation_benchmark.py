import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from graph_attention.evaluation import (
    empirical_wasserstein_1,
    infer_cartesian_grid_2d,
    run_generation_benchmark,
)
from graph_attention.evaluation.generation_benchmark import _fixed_mesh_samples
from graph_attention.evaluation.nearest_reference import nearest_reference_diagnostics
from graph_attention.evaluation.plotting import (
    _display_name,
    _ordinal_percentile,
    save_model_marginal_comparison,
)
from graph_attention.evaluation.spectra import (
    sample_radial_spectra,
    sample_velocity_energy_spectra,
    scatter_to_grid,
)


def test_empirical_wasserstein_1_equal_size_samples() -> None:
    left = np.array([0.0, 2.0, 4.0])
    right = np.array([1.0, 3.0, 5.0])

    assert empirical_wasserstein_1(left, right) == pytest.approx(1.0)


def test_cartesian_grid_inference_handles_permuted_node_order() -> None:
    x, y = np.meshgrid(np.arange(3), np.arange(4), indexing="ij")
    canonical = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)
    permutation = np.array([7, 0, 11, 3, 5, 9, 1, 10, 6, 2, 8, 4])
    coords = canonical[permutation]
    canonical_linear_index = np.ravel_multi_index(
        (coords[:, 0].astype(int), coords[:, 1].astype(int)),
        dims=(3, 4),
    )

    grid = infer_cartesian_grid_2d(coords, shape_hint=(3, 4))
    field = scatter_to_grid(canonical_linear_index, grid)

    np.testing.assert_array_equal(field, np.arange(12).reshape(3, 4))


def test_radial_spectra_drop_empty_low_k_shells() -> None:
    nx = ny = 33
    spacing = 1.7712747649056837e-05
    x, y = np.meshgrid(
        np.arange(nx) * spacing,
        np.arange(ny) * spacing,
        indexing="ij",
    )
    coords = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)
    grid = infer_cartesian_grid_2d(coords, shape_hint=(nx, ny))

    field = np.sin(2.0 * np.pi * np.arange(nx)[:, None] / nx)
    sample = np.broadcast_to(field, (nx, ny)).reshape(-1)
    samples = sample[None, :, None]

    centers, power = sample_radial_spectra(
        samples,
        grid,
        num_k_bins=24,
        subtract_mean=True,
    )

    fundamental = 2.0 * np.pi / (nx * spacing)
    assert centers[0] > 1.0e4
    assert centers[0] - 0.5 * (grid.k_nyquist_min / 24.0) <= fundamental
    assert fundamental < centers[0] + 0.5 * (grid.k_nyquist_min / 24.0)
    assert power.shape[-1] == centers.size


def test_velocity_energy_spectrum_recovers_resolved_tke_for_axis_mode() -> None:
    nx = ny = 8
    x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    coords = np.column_stack((x.reshape(-1), y.reshape(-1))).astype(np.float64)
    grid = infer_cartesian_grid_2d(coords, shape_hint=(nx, ny))

    u = np.sin(2.0 * np.pi * x / nx).reshape(-1)
    zeros = np.zeros_like(u)
    rho = np.ones_like(u)
    sample = np.stack((rho, rho * u, rho * zeros, rho * zeros, 10.0 * rho), axis=1)
    samples = sample[None, ...]
    channel_names = ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value")

    _, energy = sample_velocity_energy_spectra(
        samples,
        grid,
        channel_names,
        num_k_bins=8,
        subtract_mean=True,
    )

    expected_tke = 0.5 * np.mean(u**2)
    delta_k = grid.k_nyquist_min / 8.0
    assert float(np.sum(energy[0]) * delta_k) == pytest.approx(expected_tke)


def test_nearest_reference_uses_unpaired_descriptor_distance() -> None:
    reference = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    generated = np.array(
        [
            [0.1, 0.1],
            [1.9, 0.1],
        ]
    )

    diagnostics = nearest_reference_diagnostics(
        generated,
        reference,
        ("noise_a", "noise_b"),
        ("test_0", "test_1", "test_2"),
        ("feature_a", "feature_b"),
        normalization_eps=1.0e-12,
    )

    np.testing.assert_array_equal(diagnostics.generated_to_reference_indices, [0, 1])
    assert diagnostics.summary["generated_ids_are_sampling_keys_not_target_pairings"] is True
    assert diagnostics.summary["test_to_test_leave_one_out"]["mean"] > 0.0
    assert diagnostics.rows[0]["source_id_role"] == "sampling_key"
    assert diagnostics.rows[0]["nearest_id_role"] == "test_sample_id"


def test_generation_benchmark_accepts_legacy_shared_sample_ids() -> None:
    artifact = {
        "sample_ids": ("sample_0", "sample_1"),
        "node_counts": (2, 2),
        "channel_names": ("a",),
        "generated_nondimensional": torch.zeros((4, 1)),
        "target_nondimensional": torch.ones((4, 1)),
    }

    _, _, generated_ids, reference_ids, _ = _fixed_mesh_samples(artifact)

    assert generated_ids == ("sample_0", "sample_1")
    assert reference_ids == ("sample_0", "sample_1")


def test_generation_benchmark_writes_isolated_run_directory(tmp_path: Path) -> None:
    run_dir = tmp_path / "m12_example"
    run_dir.mkdir()
    mesh_file = tmp_path / "slice_mesh.pt"

    x, y = torch.meshgrid(torch.arange(4), torch.arange(4), indexing="ij")
    coords = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1).to(torch.float32)
    torch.save(
        {
            "coords": coords,
            "metadata": {"grid_shape_2d": [4, 4]},
        },
        mesh_file,
    )

    num_samples = 3
    num_nodes = 16
    channel_names = ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value")
    reference_samples = []
    for sample_index in range(num_samples):
        xcoord = coords[:, 0]
        ycoord = coords[:, 1]
        rho = 1.0 + 0.01 * xcoord + 0.001 * sample_index
        rhou = 0.1 * (xcoord - 1.5) + 0.01 * sample_index
        rhov = 0.1 * (ycoord - 1.5) - 0.01 * sample_index
        rhow = 0.05 * (xcoord + ycoord - 3.0)
        rhoe = 10.0 * rho + 0.02 * xcoord
        reference_samples.append(torch.stack((rho, rhou, rhov, rhow, rhoe), dim=1))
    reference = torch.stack(reference_samples, dim=0)
    generated = reference.roll(shifts=1, dims=0).clone()
    generated[..., 1] += 0.02
    generated[..., 4] *= 0.995

    torch.save(
        {
            "generated_ids": tuple(f"gen_{index:06d}" for index in range(num_samples)),
            "reference_ids": tuple(f"sample_{index}" for index in range(num_samples)),
            "node_counts": (num_nodes,) * num_samples,
            "channel_names": channel_names,
            "generated_standardized": generated.reshape(-1, len(channel_names)),
            "generated_nondimensional": generated.reshape(-1, len(channel_names)),
            "target_nondimensional": reference.reshape(-1, len(channel_names)),
            "sampling_steps": 10,
            "sampling_eta": 0.0,
            "sampler": "ddim",
            "sampling_seed": 1,
        },
        run_dir / "generated_test.pt",
    )
    OmegaConf.save(
        OmegaConf.create({"data": {"mesh_file": str(mesh_file)}}),
        run_dir / "resolved_config.yaml",
    )
    (run_dir / "summary.json").write_text(
        json.dumps(
            {
                "model": "AlternatingDilatedGeometricSparseGraphTransformer",
                "model_parameters": 123,
                "best_epoch": 7,
                "grid_shape_2d": [4, 4],
            }
        )
    )

    output_root = tmp_path / "benchmark"
    cfg = OmegaConf.create(
        {
            "run_dir": str(run_dir),
            "output_root": str(output_root),
            "generated_tensor": "generated_test.pt",
            "overwrite": False,
            "sample_space": "nondimensional",
            "eps": 1.0e-30,
            "quantiles": [0.05, 0.5, 0.95],
            "spectra": {
                "enabled": True,
                "num_k_bins": 4,
                "subtract_mean": True,
                "cartesian_tolerance": 1.0e-8,
                "bands": {
                    "low": [0.0, 0.25],
                    "mid": [0.25, 0.5],
                    "high": [0.5, 1.0],
                },
            },
            "energy_spectrum": {
                "enabled": True,
                "lower_quantile": 0.10,
                "upper_quantile": 0.90,
            },
            "nearest_reference": {
                "enabled": True,
                "include_channel_correlations": True,
                "normalization_eps": 1.0e-12,
            },
            "physics": {"enabled": True},
            "plots": {"enabled": False, "fields": False},
        }
    )

    summary = run_generation_benchmark(cfg)
    output_dir = output_root / run_dir.name

    assert summary["run_name"] == run_dir.name
    assert summary["reference_population"] == "test"
    assert summary["comparison_mode"] == "unpaired_population"
    assert summary["generated_reference_pairing"] is False
    assert summary["generated_ids_semantics"] == "explicit_generation_keys"
    assert summary["num_generated_samples"] == num_samples
    assert summary["num_reference_samples"] == num_samples
    assert output_dir.is_dir()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "benchmark_config.yaml").is_file()
    assert (output_dir / "channel_metrics.csv").stat().st_size > 0
    assert (output_dir / "sample_statistics.csv").stat().st_size > 0
    assert (output_dir / "correlation_matrix.csv").stat().st_size > 0
    assert (output_dir / "spectra.csv").stat().st_size > 0
    assert (output_dir / "spectral_bands.csv").stat().st_size > 0
    assert (output_dir / "nearest_reference.csv").stat().st_size > 0
    assert (output_dir / "physical_metrics.csv").stat().st_size > 0
    assert (output_dir / "energy_spectrum_per_sample.csv").stat().st_size > 0
    assert (output_dir / "energy_spectrum_summary.csv").stat().st_size > 0
    assert (output_dir / "energy_spectral_bands.csv").stat().st_size > 0
    assert summary["channel_summary"]["rhou.x"]["wasserstein_1"] > 0.0
    assert summary["nearest_reference"]["generated_to_test"]["mean"] >= 0.0
    assert summary["physical_summary"]["u_mean"]["generated_over_reference"] is None
    assert summary["physical_summary"]["u_mean"]["normalized_difference"] is not None
    assert summary["energy_spectrum"]["enabled"] is True
    assert summary["energy_spectrum"]["population_interval"] == [0.1, 0.9]
    assert summary["energy_spectrum"]["central_aggregations"] == ["mean", "median"]
    assert summary["energy_spectrum"]["mean_rms_log_error"] is not None
    assert summary["energy_spectrum"]["median_rms_log_error"] is not None

    with pytest.raises(FileExistsError, match="overwrite=true"):
        run_generation_benchmark(cfg)


def test_plotting_uses_explicit_cfd_variable_names() -> None:
    assert _display_name("rho.value") == "Density rho"
    assert _display_name("rhou.x") == "X-momentum rhou"
    assert _display_name("rhov.y") == "Y-momentum rhov"
    assert _display_name("rhow.z") == "Z-momentum rhow"
    assert _display_name("rhoE.value") == "Total energy rhoE"
    assert _ordinal_percentile(0.10) == "10th"
    assert _ordinal_percentile(0.90) == "90th"


def test_model_pdf_comparison_writes_one_plot_per_variable(tmp_path: Path) -> None:
    channel_names = ("rho.value", "rhou.x", "rhov.y", "rhow.z", "rhoE.value")
    reference = np.arange(2 * 4 * 5, dtype=np.float64).reshape(2, 4, 5)
    generated = {
        "EDM": reference + 0.1,
        "DDPM": reference - 0.2,
        "Flow Matching (t-scale=1000)": reference + 0.3,
    }
    output_dir = tmp_path / "pdf"

    save_model_marginal_comparison(
        reference,
        generated,
        channel_names,
        output_dir,
        bins=8,
        dpi=72,
    )

    assert sorted(path.name for path in output_dir.glob("*.png")) == [
        "pdf_Density_rho.png",
        "pdf_Total_energy_rhoE.png",
        "pdf_X-momentum_rhou.png",
        "pdf_Y-momentum_rhov.png",
        "pdf_Z-momentum_rhow.png",
    ]
