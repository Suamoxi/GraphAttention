"""True-volume CFD ray casting, with all three Cartesian axes represented."""

from __future__ import annotations

import numpy as np
import pytest

from graph_attention.evaluation.spectra_3d import infer_cartesian_grid_3d
from graph_attention.evaluation.volume_rendering_3d import (
    raycast_scalar_volume,
    save_volume_population_comparison,
)


def test_3d_raycast_responds_to_off_slice_interior_voxel() -> None:
    # A volume plot must show changes away from the three central planes.
    empty = np.zeros((8, 8, 8), dtype=np.float64)
    off_plane = empty.copy()
    off_plane[2, 3, 5] = 1.0
    baseline = raycast_scalar_volume(
        empty, vmin=-1.0, vmax=1.0, resolution=64, steps=88
    )
    modified = raycast_scalar_volume(
        off_plane, vmin=-1.0, vmax=1.0, resolution=64, steps=88
    )
    assert baseline.shape == (64, 64, 3)
    assert np.isfinite(modified).all()
    assert np.max(np.abs(modified - baseline)) > 1.0e-4


def test_3d_raycast_rejects_invalid_transfer_parameters() -> None:
    data = np.ones((5, 5, 5))
    with pytest.raises(ValueError, match="vmin/vmax"):
        raycast_scalar_volume(data, vmin=1.0, vmax=1.0, resolution=16, steps=20)
    with pytest.raises(ValueError, match="optical_depth"):
        raycast_scalar_volume(
            data, vmin=0.0, vmax=1.0, resolution=16, steps=20,
            optical_depth=-0.2,
        )


def test_3d_volume_plot_saves_matched_population_figure(tmp_path) -> None:
    side = 6
    x, y, z = np.meshgrid(
        np.arange(side, dtype=np.float64),
        np.arange(side, dtype=np.float64),
        np.arange(side, dtype=np.float64),
        indexing="ij",
    )
    coords = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    # The input order is shuffled: rendering must respect mesh coordinates.
    permutation = np.random.default_rng(11).permutation(coords.shape[0])
    grid = infer_cartesian_grid_3d(
        coords[permutation],
        periodic_endpoint_mode="none",
    )
    a = np.sin(x / side * np.pi).ravel()[permutation]
    b = np.cos(y / side * np.pi).ravel()[permutation]
    ref = a.reshape(1, -1, 1)
    generated = b.reshape(1, -1, 1)
    save_volume_population_comparison(
        {"DNS": ref, "Generated": generated},
        ("rhou.x",),
        grid,
        tmp_path,
        num_examples=1,
        dpi=70,
        resolution=32,
        steps=30,
    )
    assert (tmp_path / "example_000" / "rhou_x.png").stat().st_size > 1000
