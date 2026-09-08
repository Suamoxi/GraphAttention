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
from graph_attention.evaluation.spectra import scatter_to_grid


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
    target_samples = []
    for sample_index in range(num_samples):
        xcoord = coords[:, 0]
        ycoord = coords[:, 1]
        rho = 1.0 + 0.01 * xcoord + 0.001 * sample_index
        rhou = 0.1 * (xcoord - 1.5) + 0.01 * sample_index
        rhov = 0.1 * (ycoord - 1.5) - 0.01 * sample_index
        rhow = 0.05 * (xcoord + ycoord - 3.0)
        rhoe = 10.0 * rho + 0.02 * xcoord
        target_samples.append(torch.stack((rho, rhou, rhov, rhow, rhoe), dim=1))
    target = torch.stack(target_samples, dim=0)
    generated = target.clone()
    generated[..., 1] += 0.02
    generated[..., 4] *= 0.995

    torch.save(
        {
            "sample_ids": tuple(f"sample_{index}" for index in range(num_samples)),
            "node_counts": (num_nodes,) * num_samples,
            "channel_names": channel_names,
            "generated_standardized": generated.reshape(-1, len(channel_names)),
            "generated_nondimensional": generated.reshape(-1, len(channel_names)),
            "target_nondimensional": target.reshape(-1, len(channel_names)),
            "sample_steps": 10,
            "solver": "heun",
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
            "physics": {"enabled": True},
            "plots": {"enabled": False},
        }
    )

    summary = run_generation_benchmark(cfg)
    output_dir = output_root / run_dir.name

    assert summary["run_name"] == run_dir.name
    assert summary["reference_population"] == "paired_test_target"
    assert summary["num_samples"] == num_samples
    assert output_dir.is_dir()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "benchmark_config.yaml").is_file()
    assert (output_dir / "channel_metrics.csv").stat().st_size > 0
    assert (output_dir / "sample_statistics.csv").stat().st_size > 0
    assert (output_dir / "correlation_matrix.csv").stat().st_size > 0
    assert (output_dir / "spectra.csv").stat().st_size > 0
    assert (output_dir / "spectral_bands.csv").stat().st_size > 0
    assert (output_dir / "physical_metrics.csv").stat().st_size > 0
    assert summary["channel_summary"]["rhou.x"]["wasserstein_1"] > 0.0

    with pytest.raises(FileExistsError, match="overwrite=true"):
        run_generation_benchmark(cfg)
