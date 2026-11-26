# M47.9B — Periodic-boundary diagnostics for HIT_LES_FORCED

## Question

The earlier M47.9 retrospective Helmholtz audit measured approximately
8.30% generated longitudinal velocity energy for M37 DiNAT but only
0.0171% in its held-out `HIT_LES_FORCED` LES reference. For M37, the
full-domain divergence/vorticity RMS ratio (~0.291) greatly exceeded the
four-cell-trimmed interior ratio (~0.0557). M38 had a much weaker distinction.

M47.9B determines **which type of boundary defect** is present using
existing M37/M38 best-checkpoint Heun-50 `generated_test.pt` archives.
The source grid has 33^3 nodes with repeated periodic maximum planes;
all physical differences are measured on the distinct 32^3 grid.

No model inference, retraining, repair of periodic data, or replacement
of saved fields takes place. M40 can be included only when its completed
archived generation path is given explicitly.

## Three independent diagnostics

1. **Duplicated periodic endpoint consistency (exactly coincident physical
   coordinates).** Compare the original stored `x=0` and `x=32` planes,
   likewise for y and z. These should be equal for a correctly periodic
   field. Compute RMS of each conservative channel and of physical velocity
   (m/s). Normalize by both the typical interior one-cell increment and the
   face-adjacent ordinary one-cell increment. Exclude max planes on the two
   *other* axes to avoid counting duplicate points twice.

2. **Unique-grid one-cell wrap, not an endpoint-equality test.** Compare
   `x=31` with `x=0` on the unique grid. This is a legitimate periodic
   one-cell separation and normally has a **nonzero** velocity difference.
   Compare RMS against the n-1 ordinary one-cell differences, and against
   the two face-adjacent differences (0->1 and 30->31). Report normal
   velocity component separately. The expected wrap/interior ratio is
   statistical, not guaranteed to equal one in every individual turbulent
   realization.

3. **Divergence localization.** Use the same centered periodic derivative
   stencil as `scripts/m47_6_spatial_metrics.py` on physical velocity
   `u=U_ref (rhou,rhov,rhow)/rho`. Define the minimum unique-grid boundary
   depth d by the distance to any of six faces; mutually exclusive shells
   are d=0,1,2,3 and core d>=4. For each shell report its volume fraction,
   share of `sum((div u)^2)`, divergence-enrichment relative to uniform
   distribution, and local divergence-to-vorticity RMS ratio. Produce
   the index-by-index div(u)^2 profile along all three coordinate axes.
   On each population's first valid field, assert numerical agreement
   with the existing M47.6 full/core divergence-to-vorticity estimator.

These measurements distinguish whether M37 is primarily affected by a
failure to maintain identical stored endpoints, an anomalous one-cell
periodic wrap in the **unique field**, or a broad near-boundary divergence
excess. A large measured defect is **not proof** that the training graph
lacks periodic edges; graph connectivity is a separate audit.

## Frozen data and checks

- Same M37 and M38 archives verified by M47.9; both must have identical
  held-out reference sample IDs, complete conservative reference arrays,
  mesh source path, node ordering and grid resolution.
- Positive density is required at **all stored nodes**, including duplicate
  planes. Invalid fields are explicitly listed, never silently clipped.
- The physical velocity reference is from `cases/HIT_LES_FORCED.yaml`.
- Each field is analyzed independently. Population summaries report means,
  sample std deviations and 10th/90th percentiles, not paired comparisons
  of independent generated and held-out fields.
- No universal threshold is imposed on wrap-to-interior ratios; reference
  distributions provide the calibration.

## Run on Grace

```bash
cd /scratch/coop/theret/GraphAttention
git pull --ff-only origin main
git status --short
sbatch scripts/slurm/m47_9b_periodic_boundary.slurm
```

The launcher defaults to the *exact archived generation paths* identified
in the completed M47.9 audit. Override via `M47_9B_M37_GEN` and
`M47_9B_M38_GEN` only if needed. Optionally set `M47_9B_M40_GEN` to an
absolute completed M40 generation directory; that run must share the
M37/M38 held-out reference and mesh exactly.

Job outputs:

```
/scratch/coop/theret/GraphAttention_runs/m47_9b_periodic_boundary/
  job_<SLURM_JOB_ID>/
    audit.json
    per_field.csv
    per_axis.csv
    per_shell.csv
    plots/face_mismatch.png
    plots/divergence_by_depth.png
    plots/divergence_slice_profiles.png
```

Standard stdout/stderr paths are
`/scratch/coop/theret/GraphAttention_runs/m47_9b_boundary_<JOB_ID>.{out,err}`.
These metrics run on the Grace node under the project's Singularity
Python; the Slurm allocation requests **no GPU** and no explicit memory.

## Interpretation of specific outcomes

- **Large stored duplicate mismatch, but normal unique-grid wrap:** the
  model disagrees at physically identical duplicated mesh nodes; FFT
  drops the maximum plane, so this mismatch alone does **not** explain
  the M47.9 Fourier longitudinal fraction.
- **Large unique-grid wrap, normal interior gradients:** likely a defect
  that can affect periodic FFT and boundary-layer divergence statistics.
  Quantify the source of the discrepancy before ascribing it to graph edges.
- **Large divergence enrichment on the outermost shells:** consistent
  with boundary-localized error; use axis profiles to distinguish one or
  several boundary orientations and compare with the LES reference.
- **Elevated divergence throughout the interior:** boundary repairs alone
  may not recover solenoidal covariance; a complementary polarization-aware
  modeling or training intervention may be needed.
