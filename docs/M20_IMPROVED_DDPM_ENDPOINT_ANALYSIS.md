# M20 — Improved-DDPM endpoint sampling analysis

## Purpose

M20 isolates the effect of the terminal cosine-schedule transition on the trained
M17 Improved-DDPM model.  The M18 diagnostic on the M13 epsilon-only model found
that omitting the single `1000 -> 999` reverse transition changed generation from
catastrophically over-scaled to approximately reference-scaled.  M20 asks whether
learned reverse variance and the Improved-DDPM training objective remove that
endpoint failure.

## Controlled comparison

The same trained M17 checkpoint is evaluated in two modes:

1. **Canonical Improved DDPM**
   - Gaussian initialization at `t=1000`;
   - learned-range reverse variance;
   - all 1000 adjacent ancestral transitions;
   - `eta=1`;
   - sampling seed 5678.

2. **Gaussian restart at `t=999`**
   - Gaussian initialization directly at `t=999`;
   - learned-range reverse variance unchanged;
   - all 999 adjacent transitions from `999 -> ... -> 1 -> 0`;
   - `eta=1`;
   - sampling seed 5678.

The second mode is a project sampling hypothesis.  For `t_start < T`, the initial
state is deliberately taken as `N(0,I)` rather than being constructed from an
unknown clean sample.  It is therefore not labelled an exact draw from the trained
forward marginal.  At `t=999` the forward marginal is nevertheless expected to be
very close to Gaussian for the present cosine schedule, and M18 measured near-
identity behavior between exact-forward and Gaussian restarts for M13.

## Sampling contract

`ImprovedDiffusionDenoisingTask.sample_standardized` now accepts
`start_timestep`.  The implementation deliberately supports only adjacent
learned-variance ancestral chains:

- `steps == start_timestep`;
- `eta == 1`;
- no DDIM-style timestep skipping;
- no change to the learned reverse mean or variance equations.

`start_timestep=None` preserves the original M17 behavior and starts at `T`.

Sampler labels are explicit:

- `improved_ddpm_ancestral_learned_variance` for the canonical full chain;
- `improved_ddpm_ancestral_learned_variance_gaussian_restart` for `start<T`.

Generation artifacts persist the effective start timestep and whether the run is
a Gaussian restart.  A restart is not marked as an exact ancestral-DDPM prior
initialization.

## Evaluation

Both populations use the same held-out 228-sample reference population and the
standard generation-distribution benchmark.  The comparison should prioritize:

- dimensionless marginal W1 error;
- dimensionless mean bias and standard-deviation-ratio error;
- spectral low/mid/high-band errors;
- local spatial statistics;
- cross-channel correlations;
- nearest-reference descriptor coverage;
- physical admissibility metrics.

A relative benchmark is then computed with canonical `t=1000` as baseline and
`t=999` as candidate.

## Interpretation

The comparison separates two hypotheses:

- **Improved-DDPM formulation fixes the endpoint:** canonical `t=1000` is already
  stable and competitive with `t=999`.
- **Cosine endpoint remains pathological:** canonical `t=1000` is substantially
  worse while `t=999` is stable, showing that learned variance and loss-aware
  training do not remove the terminal-transition issue.

No performance or quality claim is made until the generated populations are
benchmarked on the target cluster.

## Implementation

- `src/graph_attention/tasks/improved_diffusion.py`
- `scripts/generate_slice_improved_diffusion.py`
- `tests/unit/test_improved_diffusion_task.py`
- `scripts/slurm/m20_improved_ddpm_start1000_vs999_generate_benchmark.slurm`

## Validation status

Implemented in repository code.  Target-cluster `ruff`, `pytest`, generation, and
benchmark validation remain required before the M20 sampling change is considered
target-validated.
