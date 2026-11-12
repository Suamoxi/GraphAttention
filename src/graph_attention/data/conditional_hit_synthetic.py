"""M45.8.1 synthetic HIT-like coarse states and locally conditioned Gaussian residuals.

This is a *statistical* benchmark, NOT a Navier--Stokes solution. The coarse
velocity is periodic and spectrally solenoidal; the fine candidate noise is
also spectrally solenoidal. Local strain-dependent modulation and exact
block-average nullspace projection generally break fine-grid solenoidality.
That defect must be measured, not assumed absent.

All stochasticity is explicit in the supplied seeds. R_f|U_c is Gaussian:
a fixed linear map of an independent fine-grid standard Gaussian.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from graph_attention.data.two_scale_synthetic import SyntheticQB


@dataclass(frozen=True)
class LocalHITConfig:
    coarse_rms: float = 0.2
    residual_fraction: float = 0.25
    context_hops: int = 2
    correlation_cutoff: float = 0.48
    anisotropy: float = 0.25

    def validate(self) -> None:
        if not self.coarse_rms > 0 or not self.residual_fraction > 0:
            raise ValueError("coarse_rms and residual_fraction must be positive")
        if self.context_hops not in (0, 1, 2):
            raise ValueError("context_hops must be 0, 1 or 2")
        if not 0.15 <= self.correlation_cutoff <= 1.0:
            raise ValueError("correlation_cutoff must be in [0.15,1.0]")
        if not 0.0 <= self.anisotropy <= 0.5:
            raise ValueError("anisotropy must be in [0,0.5]")


@dataclass(frozen=True)
class LocalHITPair:
    coarse: torch.Tensor
    fine: torch.Tensor
    baseline: torch.Tensor
    residual: torch.Tensor
    sample_id: int
    coarse_strain: torch.Tensor
    coarse_sigma: torch.Tensor


def _rng(seed: int) -> torch.Generator:
    if seed < 0:
        raise ValueError("random seed must be nonnegative")
    return torch.Generator(device="cpu").manual_seed(int(seed))


def _wavevectors(n: int) -> tuple[torch.Tensor, torch.Tensor]:
    k = torch.fft.fftfreq(n, d=1.0 / n)
    xyz = torch.stack(torch.meshgrid(k, k, k, indexing="ij"), dim=-1)
    k2 = xyz.square().sum(dim=-1)
    return xyz, k2


def _solenoidal_spectrum(noise: torch.Tensor, envelope: torch.Tensor,
                         wavevectors: torch.Tensor, k2: torch.Tensor) -> torch.Tensor:
    """Fourier Helmholtz projection, masking self-conjugate Nyquist planes."""
    n = noise.shape[0]
    hat = torch.fft.fftn(noise, dim=(0, 1, 2))
    hat = hat - wavevectors * (
        (wavevectors * hat).sum(-1, keepdim=True) / k2.clamp_min(1.0)[..., None]
    )
    nyquist = (wavevectors.abs() == n // 2).any(dim=-1) if n % 2 == 0 else torch.zeros_like(k2, dtype=torch.bool)
    envelope = envelope.masked_fill((k2 == 0) | nyquist, 0)
    return torch.fft.ifftn(hat * envelope[..., None], dim=(0, 1, 2)).real


def generate_coarse_velocity(qb: SyntheticQB, config: LocalHITConfig, *, seed: int) -> torch.Tensor:
    """Independent coherent 3D coarse velocity with an HIT-inspired spectrum."""
    config.validate()
    if qb.kind != "block_average":
        raise ValueError("M45.8 requires block_average Q")
    n = qb.nc
    generator = _rng(seed)
    noise = torch.randn((n, n, n, 3), generator=generator)
    k, k2 = _wavevectors(n)
    km = k2.sqrt()
    peak = max(1.0, 0.2 * n)
    # Per-mode amplitude: E(k) approximately k^4 at small k and k^-5/3
    # at larger k, with a finite-resolution dissipative envelope.
    envelope = km / (1.0 + (km / peak).square()).pow(17.0 / 12.0)
    envelope = envelope * torch.exp(-0.5 * (km / (0.38 * n)).pow(4))
    field = _solenoidal_spectrum(noise, envelope, k, k2)
    scale = field.square().mean().sqrt().clamp_min(1e-10)
    energy_modulation = 0.8 + 0.4 * torch.rand((), generator=generator)
    field = field * (config.coarse_rms * energy_modulation / scale)
    return field.reshape(-1, 3).contiguous()


def _interpolate(qb: SyntheticQB, coarse: torch.Tensor) -> torch.Tensor:
    """Existing periodic coarse-to-fine interpolation support, any channels."""
    source, target = qb.mesh.coarse_to_fine
    weights = qb.mesh.cross_weights
    return coarse.new_zeros((qb.nf**3, coarse.shape[-1])).index_add_(
        0, target, coarse[source] * weights[:, None]
    )


def consistent_baseline(qb: SyntheticQB, coarse: torch.Tensor) -> torch.Tensor:
    """Locally interpolated B with an exact block-average QB=I correction."""
    if qb.kind != "block_average":
        raise ValueError("M45.8 requires block_average Q")
    interpolated = _interpolate(qb, coarse)
    return interpolated + qb.reconstruct(coarse - qb.restrict(interpolated))


def _neighbor_smooth(x: torch.Tensor, hops: int) -> torch.Tensor:
    """One iteration uses only first-hop coarse neighbors, repeated to radius 2."""
    for _ in range(hops):
        neighbors = sum(
            torch.roll(x, shifts=shift, dims=axis)
            for axis in range(3) for shift in (-1, 1)
        )
        x = (2 * x + neighbors) / 8
    return x


def conditional_features(qb: SyntheticQB, coarse: torch.Tensor,
                         config: LocalHITConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Coarse strain indicator, local residual sigma and anisotropy tensor.

    Features depend on the parent coarse cell and at most context_hops
    neighbors beyond the finite-difference gradient stencil.
    """
    config.validate()
    n = qb.nc
    u = coarse.reshape(n, n, n, 3)
    spacing = qb.mesh.periodic_length / n
    gradient = torch.stack([
        (u.roll(-1, dims=axis) - u.roll(1, dims=axis)) / (2 * spacing)
        for axis in range(3)
    ], dim=-1)
    strain_tensor = 0.5 * (gradient + gradient.transpose(-1, -2))
    spin_tensor = 0.5 * (gradient - gradient.transpose(-1, -2))
    strain = (2.0 * strain_tensor.square().sum(dim=(-1, -2))).sqrt()
    rotation = (2.0 * spin_tensor.square().sum(dim=(-1, -2))).sqrt()
    # Dimensionless local invariants: no statistics from other coarse fields.
    s = _neighbor_smooth(strain * spacing / config.coarse_rms, config.context_hops)
    w = _neighbor_smooth(rotation * spacing / config.coarse_rms, config.context_hops)
    intensity = (0.75 * s + 0.25 * w).clamp(0, 5)
    sigma = config.coarse_rms * config.residual_fraction * (
        0.35 + 1.2 * torch.sigmoid(2.0 * (intensity - 0.75))
    )
    # Symmetric, traceless local orientation; bounded anisotropy.
    traceless = strain_tensor - torch.eye(3) * strain_tensor.diagonal(
        dim1=-2, dim2=-1
    ).sum(-1)[..., None, None] / 3.0
    norm = traceless.square().sum(dim=(-1, -2)).sqrt().clamp_min(1e-6)
    orientation = _neighbor_smooth(traceless / norm[..., None, None], config.context_hops)
    return strain.reshape(-1), sigma.reshape(-1), orientation.reshape(-1, 9)


def sample_local_residual(
    qb: SyntheticQB, coarse: torch.Tensor, config: LocalHITConfig, *, seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Draw R_f|U_c; Gaussian with finite-scale correlations and local amplitude.

    Conditional Gaussianity is exact: no sample-dependent RMS scaling or
    nonlinear transformation is applied to the fine random input.
    """
    if coarse.shape != (qb.nc**3, 3):
        raise ValueError("expected three-component coarse velocity")
    strain, coarse_sigma, coarse_orientation = conditional_features(qb, coarse, config)
    n = qb.nf
    xi = torch.randn((n, n, n, 3), generator=_rng(seed), dtype=coarse.dtype)
    k, k2 = _wavevectors(n)
    km = k2.sqrt()
    coarse_nyquist = qb.nc / 2
    envelope = (km / coarse_nyquist).square() * torch.exp(
        -0.5 * (km / (config.correlation_cutoff * n)).pow(4)
    )
    # Deterministic linear-filter normalization; no fieldwise random normalization.
    nyquist = (k.abs() == n // 2).any(dim=-1) if n % 2 == 0 else torch.zeros_like(k2, dtype=torch.bool)
    valid = envelope.masked_fill((k2 == 0) | nyquist, 0)
    expected_component_rms = ((2.0 / 3.0) * valid.square().mean()).sqrt().clamp_min(1e-8)
    correlated = _solenoidal_spectrum(xi, envelope, k, k2) / expected_component_rms
    sigma_f = _interpolate(qb, coarse_sigma[:, None]).reshape(n, n, n, 1)
    direction_f = _interpolate(qb, coarse_orientation).reshape(n, n, n, 3, 3)
    transform = torch.eye(3) + config.anisotropy * direction_f
    candidate = torch.einsum("...ij,...j->...i", transform, correlated) * sigma_f
    residual = qb.nullspace(candidate.reshape(-1, 3))
    return residual.contiguous(), strain, coarse_sigma


def make_local_pair(
    qb: SyntheticQB, config: LocalHITConfig, *, sample_id: int,
    coarse_seed: int, residual_seed: int,
    coarse: torch.Tensor | None = None,
) -> LocalHITPair:
    if sample_id < 0:
        raise ValueError("sample_id must be nonnegative")
    if coarse is None:
        coarse = generate_coarse_velocity(qb, config, seed=coarse_seed)
    residual, strain, sigma = sample_local_residual(
        qb, coarse, config, seed=residual_seed
    )
    baseline = consistent_baseline(qb, coarse)
    fine = baseline + residual
    return LocalHITPair(
        coarse=coarse, fine=fine, baseline=baseline, residual=residual,
        sample_id=sample_id, coarse_strain=strain, coarse_sigma=sigma,
    )
