# M44 SR-0.3 — Intrinsic point-sampled coarse-node consistency

## Evidence and scope

The preceding 50-epoch M44 SR-0.1 checkpoint (epoch 48) had 10 input
channels, 4 DiNAT-DiT layers, 64 hidden channels, and no additive absolute
coordinate embedding. It was trained on **point-sampled** HIT DNS patches
(9^3 fine nodes, 5^3 coarse nodes; refinement s=2).

The 16-path M44 SR-0.2b checkpoint diagnostic showed that matched coarse
conditioning improved Flow Matching velocity MSE relative to mismatched
conditions on **15/15 training, 3/3 validation, and 4/4 test snapshots**.
The mean wrong-minus-matched gaps were +0.1354, +0.1043, and +0.2098,
respectively. Repeating time/noise draws reduces sampling variability but
does not increase the independent number of snapshots.

Generated fine fields nevertheless violated point-sampling identity at
the 125 retained coarse nodes. A diagnostic *post hoc* projection fixed
those 125 values, but did not constrain the generator itself.

**SR-0.3 is an ablation of point-observation consistency, not a conservative
LES filter and not a physical SGS closure.**

## Mathematical construction

Let Q_sample select the N_c=125 observed coarse vertices from the
N_f=729 fine vertices, and let the baseline be L = T U_c with Q_sample T = I.
For the true DNS residual r=(U_DNS-T U_c)/sigma_r, Q_sample r = 0.

Define the diagonal free-node mask M=I-Q_sample^T Q_sample, acting
identically on each of the five conservative channels.

Training uses a restricted linear Gaussian FM path:

    epsilon ~ N(0,I)
    z0 = M epsilon
    z_data = M r (= r for exact point-sampled data)
    z_t = (1-t) z0 + t z_data
    v_target = z_data - z0

The network's output is projected to the same subspace:

    v = M v_theta(z_t, t | T standardize(U_c))

Loss is averaged ONLY over unobserved fine vertices (604 per 9^3 patch).
Zeroing anchor-node losses without changing the normalization would
artificially reduce MSE by a factor 604/729, and is explicitly avoided.

Sampling starts with z0=M epsilon and integrates

    dz/dt = M v_theta(z,t|C),

using exactly the existing M11 Heun integrator with a new optional
Boolean `state_mask` argument. Both Heun slopes are masked before state
updates, so Q_sample z(t)=0 at EVERY ODE stage. The reconstructed field
U_hat=T U_c+sigma_r z(1) satisfies Q_sample U_hat=U_c to the precision of
the CSR interpolation.

## Implementation contract

- `FlowMatchingTask.sample_standardized(state_mask=None)` defaults to
  the unchanged M11 path, random draws, numerical updates and outputs.
  When a Boolean mask is passed, it zeros the **initial noise** and
  both numerical velocity slopes; no post-generation correction.
- `SR0FlowMatching(point_sample_constraint=False)` retains the original
  SR-0 behavior. When enabled, it requires explicit
  `sr0_pilot=point_sampled_dns_not_les` sample metadata, exact
  coarse/interpolation/target anchor agreement, and a nonempty set of
  free vertices for every sample.
- The manifest contains `point_sample_constraint=true` ONLY in the new
  experiment, preserving historical manifest hashes and exact resume
  for existing unconstrained runs.
- `--point-sample-constraint` is an optional training CLI flag.
  `M44_SR01_POINT_SAMPLE_CONSTRAINT=1` enables the flag in the Slurm
  launcher; it should be used with **RESUME=0** and a new output directory.
- The checkpoint diagnostic reconstructs the constraint from the manifest
  and uses per-field MSE on free fine vertices for parity with training.
  The 16-path paired-coarse evaluation and ensemble generation remain
  available for matched and wrong conditioning comparisons.

## Calypso run: NEW 50-epoch model, not a resume of SR-0.1

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase

sbatch --export=ALL,M44_SR01_POINT_SAMPLE_CONSTRAINT=1,M44_SR01_EPOCHS=50 \
  scripts/slurm/m44_sr0_multisnapshot.slurm
```

Default output: `/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_<JOBID>/`.
This path differs automatically from the original `job_483921`.
Check `summary.json` for `point_sample_constraint=true` and
`generated_point_sample_max_abs_by_field` near zero.

After training, use the read-only M44 SR-0.2b diagnostic with
`M44_SR02_SOURCE=<NEW_RUN_DIR>` and `M44_SR02_PATH_REPEATS=16`.

## Scientific comparison caveats

The constrained and unconstrained models **do not** have the same support:
the new FM target/source lives on 604 free vertices, whereas the previous
unconstrained objective included noise at all 729 fine vertices.
Consequently, their raw training/validation FM MSE values are not directly
comparable. Compare full generation statistics, fine residual RMS, true
per-field spectra on consistent physical grids, and conditional dependence
on the same observed coarse state.

No independent LES solution, volume-integral conservative restriction,
positivity guarantee, pressure solver, or turbulence-spectrum validation
is implemented by this point-sample nullspace.
