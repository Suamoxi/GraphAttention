"""Read-only Cartesian HIT periodic-face and divergence-localization metrics.

Distinguishes equal-coordinate duplicated endpoints in a (n+1)^3 archive
from the physical one-cell periodic wrap of its n^3 unique-grid field.
"""
from __future__ import annotations

import numpy as np

AXES = ("x", "y", "z")


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def _ratio(numerator: float, denominator: float) -> float | None:
    return float(numerator / denominator) if denominator > 0.0 else None


def face_jump_statistics(full: np.ndarray, *, velocity_vector: bool = False) -> dict:
    """Compare matched duplicate faces and one-cell differences on all axes.

    RMS is taken over face/grid points and components. A periodic field must
    have identical min/max *stored* endpoint values, but the unique-grid
    wrap (n-1)->0 normally has a nonzero one-cell velocity increment.
    """
    f = np.asarray(full, dtype=np.float64)
    if f.ndim != 4 or any(n < 5 for n in f.shape[:3]) or f.shape[3] < 1:
        raise ValueError("full field must have shape [nx+1,ny+1,nz+1,C] with nx>=4")
    if not np.isfinite(f).all():
        raise ValueError("nonfinite full field")
    if velocity_vector and f.shape[-1] != 3:
        raise ValueError("velocity vector must have three components")
    unique = f[:-1, :-1, :-1, :]
    output = {}
    for axis, label in enumerate(AXES):
        n = unique.shape[axis]
        low = tuple(0 if a == axis else slice(0, unique.shape[a]) for a in range(3))
        high = tuple(n if a == axis else slice(0, unique.shape[a]) for a in range(3))
        equal_coordinate_delta = f[high] - f[low]
        wrap_delta = np.take(unique, 0, axis=axis) - np.take(unique, n - 1, axis=axis)
        interior_delta = np.diff(unique, axis=axis)
        near_low_delta = np.take(unique, 1, axis=axis) - np.take(unique, 0, axis=axis)
        near_high_delta = np.take(unique, n - 1, axis=axis) - np.take(unique, n - 2, axis=axis)
        endpoint_rms = _rms(equal_coordinate_delta)
        wrap_rms = _rms(wrap_delta)
        interior_rms = _rms(interior_delta)
        near_face_rms = float(np.sqrt(
            0.5 * (_rms(near_low_delta) ** 2 + _rms(near_high_delta) ** 2)
        ))
        row = {
            "stored_duplicate_endpoint_rms": endpoint_rms,
            "periodic_one_cell_wrap_rms": wrap_rms,
            "ordinary_one_cell_interior_rms": interior_rms,
            "face_adjacent_one_cell_rms": near_face_rms,
            "duplicate_over_interior": _ratio(endpoint_rms, interior_rms),
            "duplicate_over_face_adjacent": _ratio(endpoint_rms, near_face_rms),
            "wrap_over_interior": _ratio(wrap_rms, interior_rms),
            "wrap_over_face_adjacent": _ratio(wrap_rms, near_face_rms),
        }
        if velocity_vector:
            norm_end = _rms(equal_coordinate_delta[..., axis])
            norm_wrap = _rms(wrap_delta[..., axis])
            norm_int = _rms(interior_delta[..., axis])
            row["normal_component"] = {
                "stored_duplicate_endpoint_rms": norm_end,
                "periodic_one_cell_wrap_rms": norm_wrap,
                "ordinary_one_cell_interior_rms": norm_int,
                "duplicate_over_interior": _ratio(norm_end, norm_int),
                "wrap_over_interior": _ratio(norm_wrap, norm_int),
            }
        output[label] = row
    return output


def divergence_localization(
    velocity: np.ndarray,
    spacing_m: tuple[float, float, float],
    *, core_trim: int = 4,
) -> dict:
    """Localize div(u)^2 and |curl(u)|^2 on disjoint boundary-depth shells.

    Derivatives use the identical periodic centered stencil as M47.6.
    Depth d=min(i,nx-1-i,j,ny-1-j,k,nz-1-k); outer shells are
    d=0..core_trim-1 and the final core has d>=core_trim. Every voxel
    belongs to exactly one shell.
    """
    u = np.asarray(velocity, dtype=np.float64)
    if u.ndim != 4 or u.shape[-1] != 3 or min(u.shape[:3]) < 5:
        raise ValueError("velocity must have shape [nx,ny,nz,3], nx>=5")
    if not np.isfinite(u).all():
        raise ValueError("nonfinite velocity")
    if len(spacing_m) != 3 or not all(np.isfinite(h) and h > 0 for h in spacing_m):
        raise ValueError("positive metric spacing required")
    if not isinstance(core_trim, int) or core_trim < 2 or 2 * core_trim >= min(u.shape[:3]):
        raise ValueError("core_trim must leave a nonempty stencil-safe core")

    grad = [
        [(np.roll(u[..., c], -1, axis=a) - np.roll(u[..., c], 1, axis=a))
         / (2 * spacing_m[a]) for c in range(3)]
        for a in range(3)
    ]
    div = grad[0][0] + grad[1][1] + grad[2][2]
    curl_x = grad[1][2] - grad[2][1]
    curl_y = grad[2][0] - grad[0][2]
    curl_z = grad[0][1] - grad[1][0]
    div2 = div**2
    vort2 = curl_x**2 + curl_y**2 + curl_z**2
    total_div = float(np.sum(div2))
    total_vort = float(np.sum(vort2))
    volume = int(np.prod(u.shape[:3]))
    distance = None
    for axis, n in enumerate(u.shape[:3]):
        coord = np.minimum(np.arange(n), n - 1 - np.arange(n))
        shape = [1, 1, 1]
        shape[axis] = n
        coordinate_distance = coord.reshape(shape)
        distance = coordinate_distance if distance is None else np.minimum(distance, coordinate_distance)
    shells = {}
    for d in range(core_trim + 1):
        label = str(d) if d < core_trim else f"{core_trim}_plus"
        mask = distance == d if d < core_trim else distance >= core_trim
        count = int(np.count_nonzero(mask))
        if not count:
            raise ValueError(f"empty boundary-depth shell {label}")
        sum_div = float(np.sum(div2[mask]))
        sum_vort = float(np.sum(vort2[mask]))
        voxel_fraction = count / volume
        divergence_fraction = _ratio(sum_div, total_div)
        shells[label] = {
            "point_count": count,
            "voxel_fraction": voxel_fraction,
            "mean_divergence_squared_s_minus_2": sum_div / count,
            "mean_vorticity_squared_s_minus_2": sum_vort / count,
            "divergence_energy_fraction": divergence_fraction,
            "vorticity_energy_fraction": _ratio(sum_vort, total_vort),
            "divergence_enrichment_over_uniform": (
                float(divergence_fraction / voxel_fraction)
                if divergence_fraction is not None else None
            ),
            "divergence_to_vorticity_rms": (
                float(np.sqrt(sum_div / sum_vort)) if sum_vort > 0 else None
            ),
        }
    if sum(s["point_count"] for s in shells.values()) != volume:
        raise ValueError("boundary-depth shell coverage error")

    profiles = {}
    for axis, label in enumerate(AXES):
        avg_axes = tuple(a for a in range(3) if a != axis)
        d2 = np.mean(div2, axis=avg_axes)
        w2 = np.mean(vort2, axis=avg_axes)
        profiles[label] = {
            "mean_divergence_squared_s_minus_2": d2.tolist(),
            "mean_vorticity_squared_s_minus_2": w2.tolist(),
            "divergence_enrichment_over_uniform": (
                d2 / np.mean(div2)
            ).tolist() if total_div > 0 else None,
            "vorticity_enrichment_over_uniform": (
                w2 / np.mean(vort2)
            ).tolist() if total_vort > 0 else None,
        }
    core = shells[f"{core_trim}_plus"]
    return {
        "core_trim_cells_per_face": core_trim,
        "full_divergence_rms_s_minus_1": _rms(div),
        "full_vorticity_rms_s_minus_1": float(np.sqrt(np.mean(vort2))),
        "full_divergence_to_vorticity_rms": (
            float(np.sqrt(total_div / total_vort)) if total_vort > 0 else None
        ),
        "core_divergence_to_vorticity_rms": core["divergence_to_vorticity_rms"],
        "outer_shell_divergence_energy_fraction": (
            float(sum(shells[str(d)]["divergence_energy_fraction"] for d in range(core_trim)))
            if total_div > 0 else None
        ),
        "shells": shells,
        "axis_profiles": profiles,
    }
