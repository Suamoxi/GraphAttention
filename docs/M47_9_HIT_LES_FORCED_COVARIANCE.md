# M47.9 — Retrospective Helmholtz audit of HIT_LES_FORCED generations

## Objective

Determine whether the earlier **M37 DiNAT-DiT** and **M38 Full DiT**
unconditional full-field Flow Matching generations suffer from the same
missing longitudinal/transverse coupling identified in M47.7/M47.8B.
Optionally include **M40 local-self DiNAT** if a completed best-checkpoint
Heun-50 generation is present. Reuse stored 190 generated and 190 held-out
reference samples; no new training, noise generation, or Heun sampling.

M37/M38 source data are **HIT_LES_FORCED**, not the newer
`HIT_DNS_COMP/Re54_Ma0.2` case. Do not treat the LES reference as DNS.
Stored mesh has 33^3 endpoint-inclusive nodes and 32^3 unique periodic nodes.
Use the existing M47.7 spectral tensor operator, reference U_ref from
`cases/HIT_LES_FORCED.yaml`, and the same spherical k<=Nyquist cutoff.
No Fourier projection or periodification modifies the field.

## Outputs

For each physical-velocity snapshot u=U_ref*(rho u)_nd/rho_nd, record

- Longitudinal Fourier energy fraction fL and integrated cross-term
  cancellation `-C/D`, each using the exact M47.7 implementation.
- Interior divergence-to-vorticity RMS ratio (trim=0 and trim=4) as a
  complementary real-space metric, to check periodic seam sensitivity.
- Total fluctuation kinetic energy, so low fL cannot be 'fixed' simply
  by erasing fluctuations.
- Nonpositive-density / nonfinite samples excluded *and explicitly listed*;
  nonpositive-internal-energy fractions listed for all positive-density fields.
- Wavenumber-resolved energy, longitudinal energy, transverse energy, and
  longitudinal/cross covariance; mean and standard deviation over fields.

Original saved generation/reference arrays and held-out IDs must agree
**exactly** between architectures. The audit aborts on mismatched
reference population, mesh, node order or input channel convention.

Existing old benchmark R_LL(r), R_NN(r) and longitudinal flatness **are not**
the same as fL. The Helmholtz audit explicitly resolves the missing
wavevector-weighted cross-component covariance.

## Run once on Grace

The CPU-only job executes all Python, including pytest, in the standard
Singularity environment under Slurm (not on the login node):

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short
sbatch scripts/slurm/m47_9_hit_les_forced_helmholtz.slurm

# Optional only after the M37/M38 comparison completes successfully:
sbatch --export=ALL,M47_9_INCLUDE_M40=1 \
  scripts/slurm/m47_9_hit_les_forced_helmholtz.slurm
```

Sources:

- `/scratch/coop/theret/GraphAttention_runs/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16/generations/`
- `/scratch/coop/theret/GraphAttention_runs/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18/generations/`
- optional analogous M40 run directory

Each source must contain
`generated_test.pt`, `summary.json`, and `resolved_config.yaml`.
The launcher selects the latest best-checkpoint Heun-50 generation and
supports the original legacy `flow_heun_steps50_seed5678` directory if
its summary confirms a best.pt checkpoint.

Output:

`/scratch/coop/theret/GraphAttention_runs/m47_9_hit_les_forced_helmholtz/job_<SLURM_JOB_ID>/audit.json`

Logs:

`/scratch/coop/theret/GraphAttention_runs/m47_9_hit_LT_<SLURM_JOB_ID>.out`
and `.err`.

No explicit Slurm memory request; the job uses the established Grace CPU-only
allocation configuration.

## Interpretation

- **M37 ~1/3 fL, M38 near LES reference:** supports inadequate global
  communication/receptive field or relative-geometry inductive bias in M37;
  still not causal proof because M37/M38 train with different batch sizes.
- **Both near ~1/3 fL:** supports a failure beyond DiNAT's local
  receptive field (e.g. shared training objective or optimization limits).
- **Both match the LES reference:** M47.7's failure may depend on changed
  dataset statistics, resolution, training budget, or implementation.
- **LES reference itself near 1/3 fL:** old cases are not a useful
  transverse-covariance stress test without examining their true reference
  physics.

Compare not only integrated fL but also shellwise fL, cross cancellation,
interior div/curl, and total TKE. Spectral FFT measurements have the usual
periodic seam sensitivity; interior ratios are independent checks.
