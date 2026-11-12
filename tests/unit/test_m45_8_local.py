"""M45.8.1 tests: exact Q/B, local conditional structure and three-channel DiNAT."""

from __future__ import annotations

from argparse import Namespace
import json

import pytest
import torch

from graph_attention.data.conditional_hit_synthetic import (
    LocalHITConfig, conditional_features, consistent_baseline,
    generate_coarse_velocity, make_local_pair, sample_local_residual,
)
from graph_attention.data.two_scale_synthetic import SyntheticQB
from graph_attention.geometry.two_scale_hit import build_hit_two_scale_geometry
from scripts.m45_8_local_evaluate import (
    conditional_bin_energy, divergence_rms, radial_spectrum,
    spatial_correlations, summarize_conditional_bins,
)
from scripts.m45_8_local_train import data_seed, residual_seed, run


@pytest.mark.parametrize("radius", [0, 1, 2])
def test_local_gaussian_residual_is_reproducible_and_coarse_consistent(radius):
    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    cfg = LocalHITConfig(context_hops=radius)
    coarse = generate_coarse_velocity(qb, cfg, seed=111)
    a = make_local_pair(qb, cfg, sample_id=5, coarse=coarse,
                        coarse_seed=111, residual_seed=333)
    b = make_local_pair(qb, cfg, sample_id=5, coarse=coarse,
                        coarse_seed=111, residual_seed=333)
    c = make_local_pair(qb, cfg, sample_id=5, coarse=coarse,
                        coarse_seed=111, residual_seed=334)
    torch.testing.assert_close(a.residual, b.residual, atol=0, rtol=0)
    assert not torch.equal(a.residual, c.residual)
    torch.testing.assert_close(qb.restrict(a.baseline), coarse, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(qb.restrict(a.fine), coarse, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(
        qb.restrict(a.residual), torch.zeros_like(coarse),
        atol=1e-6, rtol=1e-6,
    )
    assert a.residual.shape == (8**3, 3)
    assert torch.isfinite(a.residual).all()
    assert a.coarse_sigma.min() > 0
    assert a.coarse_strain.max() > 0


def test_coarse_spectral_velocity_is_smooth_and_divergence_free():
    mesh = build_hit_two_scale_geometry(coarse_side=8, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    cfg = LocalHITConfig()
    coarse = generate_coarse_velocity(qb, cfg, seed=321)
    coarse2 = generate_coarse_velocity(qb, cfg, seed=322)
    assert coarse.shape == (8**3, 3)
    assert not torch.equal(coarse, coarse2)
    assert abs(float(coarse.mean())) < 1e-5
    assert float(coarse.square().mean().sqrt()) > 0.1
    # On a periodic Cartesian coarse lattice, the spectral field is solenoidal.
    assert divergence_rms(coarse, 8, mesh.periodic_length) < 1e-4


def test_local_coarse_features_are_not_constant():
    mesh = build_hit_two_scale_geometry(coarse_side=8, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    cfg = LocalHITConfig(context_hops=2)
    coarse = generate_coarse_velocity(qb, cfg, seed=555)
    strain, sigma, orientation = conditional_features(qb, coarse, cfg)
    assert strain.std() > 0
    assert sigma.std() > 0
    assert orientation.shape == (8**3, 9)
    assert float(sigma.max() - sigma.min()) > 0.001
    # No cross-field statistics or fine truth are used in conditional features.
    s2, sig2, ori2 = conditional_features(qb, coarse, cfg)
    torch.testing.assert_close(sigma, sig2, atol=0, rtol=0)
    torch.testing.assert_close(strain, s2, atol=0, rtol=0)
    torch.testing.assert_close(orientation, ori2, atol=0, rtol=0)


def test_conditional_residual_energy_tracks_coarse_environment():
    mesh = build_hit_two_scale_geometry(coarse_side=6, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    cfg = LocalHITConfig(context_hops=1)
    coarse = generate_coarse_velocity(qb, cfg, seed=556)
    pair = make_local_pair(qb, cfg, sample_id=0, coarse=coarse,
                           coarse_seed=556, residual_seed=901)
    draws = torch.stack([
        sample_local_residual(qb, coarse, cfg, seed=7000 + i)[0]
        for i in range(48)
    ])
    boundaries = torch.quantile(pair.coarse_strain, torch.linspace(0, 1, 6))
    bins = conditional_bin_energy(qb, pair, draws, boundaries)
    assert sum(row["nodes"] for row in bins) == qb.nf**3
    # This is a deliberate amplitude-conditioned generator, not stationary noise.
    assert bins[-1]["kinetic_energy_per_node"] > bins[0]["kinetic_energy_per_node"]
    rows = [dict(distribution="reference", **x) for x in bins]
    rows.extend(dict(distribution="generated", **x) for x in bins)
    summary = summarize_conditional_bins(rows)
    assert len(summary) == 10
    assert summary[0]["mean_kinetic_energy"] == pytest.approx(
        summary[5]["mean_kinetic_energy"]
    )


def test_physical_wavenumber_energy_spectrum_obeys_parseval():
    n = 8
    x = torch.randn((n**3, 3), generator=torch.Generator().manual_seed(100))
    spectrum = radial_spectrum(x, n, 1.0)
    energy = 0.5 * x.square().sum(-1).mean()
    torch.testing.assert_close(spectrum.sum() * (2 * torch.pi), energy, rtol=1e-5, atol=1e-5)
    with pytest.raises(ValueError, match="expected"):
        radial_spectrum(torch.randn(n**3, 5), n, 1.0)


def test_correlations_use_centered_conditional_ensemble():
    mesh = build_hit_two_scale_geometry(coarse_side=4, refinement=2)
    qb = SyntheticQB(mesh, "block_average")
    coarse = generate_coarse_velocity(qb, LocalHITConfig(), seed=998)
    res = torch.stack([
        sample_local_residual(qb, coarse, LocalHITConfig(), seed=9000+i)[0]
        for i in range(4)
    ])
    corr = spatial_correlations(res, qb.nf)
    assert len(corr) == 3 * (qb.nf // 2 + 1)
    assert all(x["correlation"] == pytest.approx(1.0) for x in corr if x["lag"] == 0)
    assert all(abs(x["correlation"]) <= 1.0001 for x in corr)
    with pytest.raises(ValueError, match=">= 2"):
        spatial_correlations(res[:1], qb.nf)


@pytest.mark.parametrize("mode", ["static", "dynamic"])
def test_short_training_and_evaluation_for_three_channels(mode, tmp_path):
    args = Namespace(
        output_dir=tmp_path / mode, mode=mode, residual_policy="resample",
        device="cpu", coarse_side=3, refinement=2, train_fields=6,
        val_fields=2, normalization_fields=4, steps=2,
        depth=1, width=8, heads=2, learning_rate=1e-3,
        dataset_seed=100, residual_seed=200, init_seed=300,
        schedule_seed=400, validation_seed=500,
        coarse_rms=0.2, residual_fraction=0.25, context_hops=1,
        correlation_cutoff=0.48, anisotropy=0.25,
    )
    result = run(args)
    assert result["settings"]["channels"] == 3
    assert result["settings"]["mode"] == mode
    assert (args.output_dir / "local_hit" / "block_average" / mode / "checkpoint.pt").is_file()
    from scripts.m45_8_local_evaluate import evaluate
    eval_dir = tmp_path / (mode + "_eval")
    report = evaluate(
        checkpoint_root=args.output_dir, output_dir=eval_dir,
        val_fields=1, draws=2, heun_steps=2, device_name="cpu",
    )
    assert report["settings"]["mode"] == mode
    assert report["results"]["generated"]["max_coarse_defect"] < 1e-5
    assert (eval_dir / "spectra.csv").is_file()
    assert (eval_dir / "conditional_energy.csv").is_file()
    assert (eval_dir / "spatial_correlations.csv").is_file()
    with pytest.raises(FileExistsError):
        run(args)


def test_invalid_settings_and_independent_seeds():
    with pytest.raises(ValueError, match="context_hops"):
        LocalHITConfig(context_hops=3).validate()
    with pytest.raises(ValueError, match="nonnegative"):
        data_seed(-1, 0)
    assert data_seed(100, 0) != data_seed(100, 1)
    assert residual_seed(200, 0) != residual_seed(200, 1)
