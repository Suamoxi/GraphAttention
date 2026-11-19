# M47.6 - Spatial-physics and Flow Matching trajectory diagnostics

**Status:** Implemented in main. Local NumPy synthetic tests passed. Full
repository tests and GPU sampling still require Calypso execution.

M47.5 showed that the conditioned models learn regime-dependent total-energy
means, but all models have dilatational FFT fractions near one-third,
large periodic seams, deficient low-k energy and excessive high-k energy.

M47.6 asks whether these errors stem primarily from nonperiodic boundaries
or persist in the domain interior, whether the spatial structure develops
during Flow Matching sampling, and whether 50 Heun steps are sufficient.

## Scope

Frozen M47.4 epoch10 last.pt models for none, mach and mach_re.
Original M47.1 five validation DNS snapshots from each of Re54/Ma0.05,
Re54/Ma0.2 and Re54/Ma0.3. All these regimes were in training.

Generated Gaussian keys: M47.2/<case>/replicate_<index:04d>, seed 20261116.
M47.1 training scales, graph edges, checkpoint weights and normalization
are unchanged. Nothing is clipped, periodified, trained or physically
projected.

### 1. Interior divergence and vorticity

Physical velocity is U_ref * (rhou,rhov,rhow)/rho. Compute central
differences at the original physical mesh spacing and the following metrics:

- RMS divergence and RMS vorticity, in s^-1;
- RMS(div u) / RMS(curl u);
- [RMS(div u)]^2 / ([RMS(div u)]^2 + [RMS(curl u)]^2).

Evaluate with 0, 2, 4 and 8 cells excluded per face. Trim 0 uses periodic
stencils and includes boundary effects. Trims >=2 use strictly interior
stencils. The gradient ratio is NOT the spectral dilatational energy fraction.

### 2. Longitudinal / transverse spectra

Compute 3-D Fourier spectra from physical velocity on the unique 64^3 grid
after dropping duplicate stored maximum-coordinate planes. Output
wavenumber in rad/m and specific kinetic-energy spectrum in m^3/s^2.
Check E_total = E_solenoidal + E_dilatational.

Two variants:

- Raw periodic FFT - physically interpretable only if the field is actually
  periodic (the generated seams currently suggest this is violated).
- Hann-windowed FFT projection - boundary-sensitivity check ONLY. It is
  not a physical Helmholtz decomposition because tapering creates divergence.

The generated fields themselves are never modified.

### 3. Flow Matching trajectory

On one fixed Gaussian latent in Re54/Ma0.2, instrument the existing
50-step Heun algorithm at t=0,0.2,0.4,0.6,0.8,1. Intermediate states
are noisy standardized model variables: report standardized momentum
spectra and interior lag-one correlation, not physical velocity.

Verify final-state numerical parity with the original canonical sampler
using atol=rtol=1e-6. A failure aborts the experiment.

### 4. Heun step convergence

Use two matched Gaussian keys on Re54/Ma0.2 to compare 50- and
100-step Heun results. Report per-channel RMS difference normalized by
the DNS reference standard deviation, relative velocity-spectrum L1
difference, full FFT dilatational fraction, and interior div/curl.
This is a step-size sensitivity experiment, not a formal convergence proof.

## Calypso commands

Run from a clean checkout, with host paths under /scratch (not the
host-resolved /gpfs-calypso/scratch prefix inside Singularity):

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase origin main
git status --short
unset M47_CONDITION

# First gate: unconditioned model
sbatch --array=0 scripts/slurm/m47_6_spatial_physics.slurm

# After successful completion of task 0
sbatch --array=1-2 scripts/slurm/m47_6_spatial_physics.slurm
~~~

Array indices: 0=none, 1=mach, 2=mach_re.
Default setup: four generations per regime, two paired 50/100 comparisons,
seed 20261116, 32 wavenumber bins.

Optional environment overrides:
- M47_6_OUT_ROOT for a new immutable output root;
- M47_6_SAMPLES (minimum 2);
- M47_6_COMPARE_COUNT (between 1 and SAMPLES);
- M47_6_SEED, M47_4_OUT_ROOT, M47_CONDITION_REFS,
  M47_ARCHIVE_ROOT, M47_REPO, M47_IMAGE, M47_PYTHON.

The job runs these four CPU test suites in Singularity before GPU sampling:

- tests/unit/test_m47_6_spatial_metrics.py
- tests/unit/test_m47_5_regime_diagnostics.py
- tests/unit/test_m47_4.py
- tests/unit/test_m47_generation.py

Local NumPy-only tests passed for exactly solenoidal and dilatational
analytic fields, a boundary-only defect, and nonphysical intermediate
standardized states. They do not establish full GPU correctness.

Existing output folders cannot be overwritten. Use a new M47_6_OUT_ROOT
to rerun a failed or completed experiment.

## Mandatory artifacts for analysis

For each none, mach and mach_re run, upload all six JSON files:

~~~text
/scratch/coop/theret/GraphAttention_runs/m47_6_spatial_physics/
  <mode>/epoch10_h50_h100_n4_seed20261116/
    summary.json
    case_Re54_Ma0.05_divcurl.json
    case_Re54_Ma0.2_divcurl.json
    case_Re54_Ma0.3_divcurl.json
    flow_trajectory_Re54_Ma0.2.json
    heun_step_convergence.json
~~~

Also upload the Slurm standard output and error logs, both per array task:

~~~text
/scratch/coop/theret/GraphAttention_runs/m47_6_spatial_<jobid>_<task>.out
/scratch/coop/theret/GraphAttention_runs/m47_6_spatial_<jobid>_<task>.err
~~~

For initial analysis, checkpoints, raw DNS HDF5 and full generated
conservative fields are NOT necessary: the JSON files include per-sample
curves and scalar statistics. If a job fails, share .out and .err regardless
of whether JSON artifacts exist.

## Decision logic

If the interior divergence/vorticity ratio matches DNS while FFT dilatational
fraction does not, boundary effects become a stronger explanation.
If interior divergence/vorticity also strongly mismatches, the failure
persists in the bulk. A large 50/100 difference warrants sampler refinement;
a small difference suggests prioritizing architecture/representation.
No unseen-regime generalization can be claimed from this in-distribution audit.
