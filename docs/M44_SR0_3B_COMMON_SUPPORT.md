# M44 SR-0.3b — Common-support checkpoint comparison

## Motivation

The 50-epoch constrained point-sampled-DNS pilot (`point_sample_constraint=true`)
satisfies coarse-node equality with max error 0 on all five conservative
fields, compared with nonzero anchor errors in the original unrestricted
pilot. Both 50-epoch checkpoints selected best epoch 48.

The native training MSE values cannot be directly ranked:

- SR-0.1: standard linear Gaussian Flow Matching, all 729 fine nodes.
- SR-0.3: Gaussian and velocity masked at 125 observed coarse nodes, and
  loss normalized over 604 **unobserved** fine nodes.

We must compare on the same scoring support before making any claim
about relative free-node velocity fitting.

## Read-only common-support evaluation

The existing checkpoint diagnostic
`scripts/diagnose_m44_sr0.py --path-repeats N` now emits
`repeated_flow_matching.<split>.common_unobserved_node_support` for
every split when `N>=2`.

For both checkpoints it:

1. Validates that each sample is an explicitly tagged point-sampled DNS proxy.
2. Verifies baseline and fine target equal the coarse state exactly at all
   125 injected nodes, then constructs the same unobserved-node mask.
3. Uses the original deterministic sample-ID seed and additional independent
   Flow Matching paths to evaluate each model **on those 604 nodes only**.
4. Preserves the native training path for each model; it does not retrain
   or force the unrestricted model's observed-node Gaussian to zero.
5. Reports common-support matched, wrong-coarse and zero-reference losses,
   with a per-field and per-source-snapshot breakdown.

The *unobserved-node random time and Gaussian* are identical for a
corresponding sample ID and path seed in the two models. The **full
model input differs at observed nodes**, because the unrestricted model
was trained with Gaussian noise there and the constrained model was not.
The common support fixes the metric denominator and the free-node
random inputs, but does not turn the experiment into an identical-input
architecture comparison.

## Run both saved 50-epoch best checkpoints (sequentially)

Old unrestricted checkpoint:

```bash
sbatch --export=ALL,M44_SR02_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_483921,M44_SR02_PATH_REPEATS=16,M44_SR02_DRAWS=16,M44_SR02_MAX_PATCHES=4 scripts/slurm/m44_sr02_diagnostics.slurm
```

Constrained checkpoint: substitute the new SR-0.3 Slurm training
`job_<ID>` directory for `M44_SR02_SOURCE`. Run with the same
`PATH_REPEATS`, `DRAWS`, and `MAX_PATCHES`.

Each read-only report goes to its own
`m44_sr02_diagnostics/job_<diagnostic_JOB_ID>` directory.

## Interpretation

The original one-path `flow_matching_counterfactual` and native
`repeated_flow_matching` remain available and backward compatible.
The new `common_unobserved_node_support` is the comparison field.

Also inspect `generation_ensemble` on the same held-out sample IDs:

- true versus generated residual RMS by conservative field;
- sample-to-sample standard deviation across multiple generated draws;
- raw coarse-node discrepancy, with and without postprojection;
- nonpositive density count;
- the variation across independent source DNS snapshots.

Even with common support, lower velocity MSE **does not establish** better
conditional distribution quality, spectral reconstruction or SGS
physics. A 9x9x9 nonperiodic patch does not establish a domain-wide
kinetic-energy spectrum; use larger patches/whole DNS for physical
wavenumber analysis once the proxy experiment is stable.

No physical LES restriction is implemented in SR-0.3.
