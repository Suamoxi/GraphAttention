# M47.2 — independent, matched generation evaluation of three M47.1 models

## Objective

Test whether the 36–41% improvement in **Flow Matching transport-velocity** MSE in
M47.1 produces physically meaningful improvements in *independently generated*
3-D compressible HIT fields. This is not supervised reconstruction: no generated
state is paired to a particular held-out DNS field.

**Frozen training models:** `none`, `mach`, `mach_re`, each from
`m47_conditioned/<mode>/train_s42/best.pt` (seed 42, two epochs, 240 training,
30 validation, 42 held-out test snapshots, six regimes). Retain shared
adaLN-Zero. Do not alter training or infer convergence from two epochs.

**Sampling:** one full periodic 65³ or 129³ graph per case, initialized from
an independent standard Gaussian with a mode-invariant hashed sample key.
Integrate the same deterministic Flow Matching ODE with **50 Heun steps**
(100 network evaluations). Use 8 independent initial Gaussian samples for each
of 6 regime cases, per model. Reference population uses the 7 held-out DNS
snapshots per case (42 total). Generated and reference populations have
**unequal sample counts and no one-to-one pairing**.

The archive test snapshot used to construct graph edges supplies only the
mesh and reference condition; the sampler discards its state as an initial
condition. All three models receive the exact same latent Gaussian tensors for
the same case and replicate ID by construction.

## Physical and numerical diagnostics

All observations are de-standardized using the model's frozen **training-only**
M6 scaler; conserved fields are then in the original M3.3 *convectively
nondimensionalized* state.

* One-point distributions: exact empirical 1D Wasserstein-1 on fixed bounded
  spatial subsamples of `{rho,rhou,rhov,rhow,rhoE}`. Report both raw
  nondimensional W1 and W1/reference-test-standard-deviation. Every snapshot
  contributes the same number of uniformly sampled *unique* mesh nodes.
* Positivity: fraction of unique grid points with `rho<=0`; fraction violating
  `rhoE-(rhou²+rhov²+rhow²)/(2rho)>0` or positive density. No clipping,
  projection, or rejection of bad samples from the one-point comparisons.
* Periodic seam: repeated maximum-coordinate planes must agree with the
  corresponding minimum-coordinate planes; report seam RMS/reference sample
  field std, per conserved field. **FFT `drop_max` is not a substitute for
  checking physical periodicity.**
* Velocity: only where a **whole generated field has strictly positive
  density**, compute `u=(rhou,rhov,rhow)/rho`, an isotropic velocity-energy
  spectrum in dimensional `k` (rad/m), normalized by `U_ref²`; calculate
  `L1(E_gen-E_ref)/integral(E_ref)` using matched shell bins.
* Helmholtz split: Fourier longitudinal projection `P_k=kkᵀ/|k|²` and
  dilatational energy fraction `sum|P_k u_hat|²/sum|u_hat|²` on unique
  64³/128³ grids. Test and reference must both be computed in the same frame.
* Axis-averaged periodic velocity two-point correlations (max lag 8 cells)
  from existing `spatial_statistics_3d` implementation, physical separation m.
  Unavailable spectra/correlations remain **null**, not silently filled.

The code uses `spectra_3d.infer_cartesian_grid_3d(...,
periodic_endpoint_mode='drop_max')`. Physical k comes from dimensional AVBP
coordinates, **not** `k/k_Nyquist`. The plot/CSV interpretation must include
this periodic-mesh caveat. If the state is not globally admissible, skip its
velocity-based statistics and report counts; keep its conserved-field
statistics, normalized Wasserstein values and endpoint jumps.

## Execute on Calypso

Pull the branch and run the deliberately small three-arm GPU smoke check:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
unset M47_CONDITION
sbatch --array=0-2 scripts/slurm/m47_generation_test.slurm
```

Smoke runs 2 Heun steps and 1 sample on Re54/Ma0.2, exercising checkpoint
reconstruction, strict source signatures, dimensional grid and diagnostics.
Inspect all three logs, including positivity and finite-result checks.

After all smoke jobs complete successfully:

```bash
sbatch --array=0-2 \
  --export=ALL,M47_GEN_MODE=full,M47_GEN_CONFIRM=YES \
  scripts/slurm/m47_generation_test.slurm
```

Set `M47_GEN_STORE_FIELDS=YES` if full conservative-volume snapshots are
required for later inspection; without it, the per-case JSON still contains
all recorded physical metrics and the sampling keys reproduce the volumes.

After the full Slurm job **array ID** is returned, submit a CPU-only dependent
comparison (replace the placeholder with the real array ID):

```bash
sbatch --dependency=afterok:<FULL_ARRAY_JOB_ID> \
  scripts/slurm/m47_generation_compare.slurm
```

A full experiment writes:

```
/scratch/coop/theret/GraphAttention_runs/m47_generated/
  none/ensemble_s42_h50_n8_seed20261116/{summary.json,case_*.json}
  mach/ensemble_s42_h50_n8_seed20261116/{summary.json,case_*.json}
  mach_re/ensemble_s42_h50_n8_seed20261116/{summary.json,case_*.json}
  comparison_ensemble_s42_h50_n8_seed20261116.json
  comparison_ensemble_s42_h50_n8_seed20261116.csv
```

The aggregation requires identical Gaussian keys, reference samples, FFT grids,
sampling solver/steps, numerical settings and generator code revision. It
refuses partial/mismatched runs. Each case has equal weight in cross-regime
reporting; **do not pool nodes across 64³ and 128³** as though they were
equivalent independent samples.

## Interpretation constraints

* The best M47.1 checkpoints are from **epoch 1 of 2**; all losses still fell
  substantially between epochs. This is an early-training generative pilot,
  not a converged result.
* There are 8 generated fields per case and only 7 held-out reference fields,
  potentially temporally correlated. These are descriptive diagnostics, not
  calibrated confidence intervals.
* Re54 and Re94 are confounded with mesh resolution; conditioning may learn a
  resolution proxy. Global Mach/Reynolds interpolation and extrapolation need
  a different held-out-regime experiment.
* Graph attention does not yet include AVBP SOLUTBOUND periodic graph edges.
  Generated field seam jumps are an explicit diagnostic and potential dominant
  failure mode.
* Positive density and internal energy is a necessary perfect-gas admissibility
  proxy, not a complete independent AVBP thermodynamic-state validation.
* No architecture can be declared physically superior solely from Flow Matching
  velocity MSE. Test-set results are evaluation only, not tuning targets.

## Suggested decision rule

Proceed to longer convergence runs or more seeds only after verifying (1)
physically acceptable density/internal energy across a substantial fraction
of generated fields, (2) improved normalized conservative-field W1 and
spectral discrepancies over the unconditioned baseline, (3) no worsening
of seam mismatch and (4) acceptable sensitivity to sample/Heun budgets.
If no generated field is density-positive everywhere, numerical spectra are
*not available*, and that negative result should be reported rather than
masked by an arbitrary density floor.
