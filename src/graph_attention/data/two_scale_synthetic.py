"""Operator-consistent *synthetic* HIT pairs for M45 architecture comparisons.

Neither Q/B family is claimed to represent filtered LES. These local, periodic
operators isolate geometry, scale coupling and stochastic-correlations tests
while the physical Q/B contract is developed separately in dnstoles.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from graph_attention.geometry.two_scale_hit import HITTwoScaleGeometry

CHANNELS = ("rho", "rhou", "rhov", "rhow", "rhoE")
OPERATORS = ("block_average", "anchor_trilinear")
CASES = ("local", "shared_mode")


@dataclass(frozen=True)
class SyntheticPair:
    """One reproducible coarse/fine snapshot in dimensionless variables."""

    coarse: torch.Tensor
    fine: torch.Tensor
    baseline: torch.Tensor
    residual: torch.Tensor
    mode: torch.Tensor
    sample_id: int


class SyntheticQB:
    """Exact simple Q/B right-inverse families on nested periodic grids.

    `block_average`: Q averages r**3 fine DOFs per coarse cell and B repeats
    the cell value. The nodal lattice is treated as volume-associated sites.
    `anchor_trilinear`: Q reads fine nodes at coarse anchors and B uses the
    same periodic trilinear geometric weights as M45 cross-scale supports.
    Neither choice is the physical dnstoles DNS/LES restriction.
    """

    def __init__(self, mesh: HITTwoScaleGeometry, kind: str) -> None:
        if kind not in OPERATORS:
            raise ValueError(f"operator must be one of {OPERATORS}")
        self.mesh = mesh
        self.kind = kind
        self.nc = mesh.coarse_side
        self.nf = mesh.fine_side
        self.r = mesh.refinement

    def restrict(self, fine: torch.Tensor) -> torch.Tensor:
        self._shape(fine, self.nf**3, "fine")
        channels = fine.shape[1]
        grid = fine.reshape(self.nc, self.r, self.nc, self.r, self.nc, self.r, channels)
        if self.kind == "block_average":
            return grid.mean(dim=(1, 3, 5)).reshape(-1, channels)
        return fine.reshape(self.nf, self.nf, self.nf, channels)[::self.r, ::self.r, ::self.r].reshape(-1, channels)

    def reconstruct(self, coarse: torch.Tensor) -> torch.Tensor:
        self._shape(coarse, self.nc**3, "coarse")
        channels = coarse.shape[1]
        if self.kind == "block_average":
            return coarse.reshape(self.nc, self.nc, self.nc, channels).repeat_interleave(
                self.r, dim=0
            ).repeat_interleave(self.r, dim=1).repeat_interleave(self.r, dim=2).reshape(-1, channels)
        source, target = self.mesh.coarse_to_fine
        if source.device != coarse.device or self.mesh.cross_weights.device != coarse.device:
            raise ValueError("operator geometry and coarse field must share device")
        return coarse.new_zeros((self.nf**3, channels)).index_add_(
            0, target, coarse[source] * self.mesh.cross_weights[:, None]
        )

    def nullspace(self, fine: torch.Tensor) -> torch.Tensor:
        """Projection I-BQ. Exact because QB=I; no global solve."""
        return fine - self.reconstruct(self.restrict(fine))

    @staticmethod
    def _shape(state: torch.Tensor, nodes: int, name: str) -> None:
        if state.ndim != 2 or state.shape[0] != nodes or not state.is_floating_point():
            raise ValueError(f"{name} must be floating-point [N,{name}_channels]")


def shared_mode(qb: SyntheticQB) -> torch.Tensor:
    """Fixed, normalized, high-frequency spatial mode in ker(Q), five channels.

    Its amplitude is randomized *per snapshot*. The coarse state cannot
    identify that amplitude. This tests long-range conditional correlations.
    """
    coords = qb.mesh.fine_coords
    phase = 2 * math.pi * qb.nc * coords / qb.mesh.periodic_length
    raw = (torch.cos(phase[:, 0]) + 0.3 * torch.cos(phase[:, 1])
           + 0.2 * torch.cos(phase[:, 2]))[:, None]
    projected = qb.nullspace(raw)
    rms = projected.square().mean().sqrt()
    if not rms > 1.0e-7:
        raise ValueError("shared mode was annihilated by Q/B; choose another mode")
    return projected / rms


def sample_conditional_residual(
    qb: SyntheticQB, coarse: torch.Tensor, *, case: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent fine residual for a *fixed* coarse state and PRNG stream.

    Return (R, spatial_template). Reuses the identical stochastic construction
    as the training pairs; unlike make_synthetic_pair, coarse is not resampled.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    qb._shape(coarse, qb.nc**3, "coarse")
    baseline = qb.reconstruct(coarse)
    local_noise = torch.randn((qb.nf**3, 5), generator=generator, dtype=coarse.dtype)
    strength = 0.75 + 0.25 * torch.tanh(20 * (baseline[:, :1] - 1.0))
    channel_scale = coarse.new_tensor((0.018, 0.023, 0.023, 0.023, 0.050))
    candidate = local_noise * strength * channel_scale
    template = shared_mode(qb)
    if case == "shared_mode":
        latent = torch.randn((), generator=generator, dtype=coarse.dtype)
        candidate = candidate + latent * template * channel_scale * 1.7
    residual = qb.nullspace(candidate)
    return residual, template


def make_synthetic_pair(
    qb: SyntheticQB, *, case: str, sample_id: int, seed: int
) -> SyntheticPair:
    """Generate a new snapshot without fine truth leaking into its coarse input.

    Coarse state is independent of the global unresolved random amplitude.
    Fine truth = B(coarse) + (I-BQ) * unconstrained candidate fluctuations.
    """
    if case not in CASES:
        raise ValueError(f"case must be one of {CASES}")
    if sample_id < 0 or seed < 0:
        raise ValueError("sample_id and seed must be nonnegative")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    coords = qb.mesh.coarse_coords
    phi = 2 * math.pi * torch.rand(3, generator=generator)
    xx = coords * (2 * math.pi / qb.mesh.periodic_length)
    wave = torch.sin(xx[:, 0] + xx[:, 1] + phi[0]) + 0.4 * torch.cos(
        xx[:, 1] - xx[:, 2] + phi[1]
    )
    coarse = torch.stack((
        1.0 + 0.025 * wave,
        0.040 * torch.sin(xx[:, 0] + phi[1]) + 0.010 * wave,
        0.035 * torch.cos(xx[:, 1] + phi[2]),
        0.035 * torch.sin(xx[:, 2] + phi[0]),
        2.5 + 0.080 * wave + 0.010 * torch.sin(xx[:, 2] + phi[2]),
    ), dim=1)
    baseline = qb.reconstruct(coarse)

    residual, template = sample_conditional_residual(qb, coarse, case=case, generator=generator)
    fine = baseline + residual
    torch.testing.assert_close(qb.restrict(fine), coarse, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(qb.restrict(residual), torch.zeros_like(coarse), atol=1e-6, rtol=1e-6)
    return SyntheticPair(coarse, fine, baseline, residual, template, sample_id)


@dataclass(frozen=True)
class SyntheticScales:
    mean_coarse: torch.Tensor
    std_coarse: torch.Tensor
    rms_residual: torch.Tensor


def fit_synthetic_scales(pairs: list[SyntheticPair]) -> SyntheticScales:
    """Fit once on training snapshots; never on held-out validation samples."""
    if not pairs:
        raise ValueError("training pairs must not be empty")
    coarse = torch.cat([p.coarse for p in pairs], dim=0)
    residual = torch.cat([p.residual for p in pairs], dim=0)
    mean = coarse.mean(dim=0)
    std = coarse.std(dim=0, unbiased=False)
    rms = residual.square().mean(dim=0).sqrt()
    if torch.any(std < 1.0e-8) or torch.any(rms < 1.0e-8):
        raise ValueError("training data has a degenerate channel variance")
    return SyntheticScales(mean, std, rms)
