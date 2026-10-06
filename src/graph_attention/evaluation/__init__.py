"""Evaluation and scientific post-processing utilities."""

from .generation_benchmark import empirical_wasserstein_1, run_generation_benchmark
from .generation_benchmark_3d import run_generation_benchmark_3d
from .spectra import CartesianGrid2D, infer_cartesian_grid_2d
from .spectra_3d import CartesianGrid3D, infer_cartesian_grid_3d

__all__ = [
    "CartesianGrid2D",
    "CartesianGrid3D",
    "empirical_wasserstein_1",
    "infer_cartesian_grid_2d",
    "infer_cartesian_grid_3d",
    "run_generation_benchmark",
    "run_generation_benchmark_3d",
]
