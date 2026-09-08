"""Evaluation and scientific post-processing utilities."""

from .generation_benchmark import empirical_wasserstein_1, run_generation_benchmark
from .spectra import CartesianGrid2D, infer_cartesian_grid_2d

__all__ = [
    "CartesianGrid2D",
    "empirical_wasserstein_1",
    "infer_cartesian_grid_2d",
    "run_generation_benchmark",
]
