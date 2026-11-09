# M45.7.2 — conditional synthetic residual resampling

**Scientific scope:** synthetic Gaussian block-average `Q/B`, not LES/DNS.
**Code:** extends the original `scripts/m45_7_oracle_supervision.py`.
**Architectures:** existing static/dynamic `TwoScaleDiNAT`, unchanged.
**Question:** does representing the conditional distribution more fully in the
training *states* improve learned stochastic covariance beyond the fixed-pair
M45.7.1 pilot?

## 1. Design

Choose one of 24 *fixed* synthetic coarse training fields \(U_c^{(i)}\).
The matched experiments change only the policy for drawing the target fine
residual \(R_f\):

- `fixed`: reuse the single originally generated paired residual \(R_f^{(i)}\).
- `resample`: draw an independent residual \(R_f^{(i,s)}\) at each update
  from the **existing** `sample_conditional_residual(qb, U_c, case, generator)`
  function. Never regenerate or modify \(U_c^{(i)}\).

Both policies satisfy \(QR_f=0\) and \(U_f=B U_c+R_f\), so the input/output
pairing is preserved exactly. The resampled target is **not** invented data
for a physical DNS run; the Gaussian conditional distribution is explicitly
known in this synthetic test.

The training path remains
\(X_t=(1-t)\epsilon+t R_f/\sigma_R\) with source
\(\epsilon\in\ker(Q)\), and target is either standard
\(R_f/\sigma_R-\epsilon\) or the analytical Gaussian conditional-mean
velocity. Coarse state normalization, baseline, and residual RMS scales
are *fitted once on the original 24 fixed training pairs and frozen* for
all eight configurations. This is deliberate: changing normalization while
changing residual diversity would conflate two effects.

At every update, all jobs use the same selected coarse sample index,
FM time, and source Gaussian noise (independent, deterministic PRNG streams).
The `resample` stream depends only on train seed and update index, not
supervision type or architecture. Oracle covariance is computed for the
selected *coarse field*, using the exact Gaussian conditional distribution,
and cached by coarse index. The validation dataset and seed are unchanged.

## 2. Full factorial at each update budget

| Array task | Model | Supervision | Residual policy |
|---|---|---|---|
| 0 | static | FM | fixed |
| 1 | static | FM | resample |
| 2 | static | oracle | fixed |
| 3 | static | oracle | resample |
| 4 | dynamic | FM | fixed |
| 5 | dynamic | FM | resample |
| 6 | dynamic | oracle | fixed |
| 7 | dynamic | oracle | resample |

Defaults: `shared_mode`, block-average Q/B, coarse side 4, refinement 2,
24 coarse fields, 8 held-out pairs, train seed 2028, validation seed
7002026, depth 4, width 16, 4 attention heads, 2,000 updates, 128
generated samples, 25 Heun steps.

The `fixed` runs at 2,000 steps are intentional independent controls and
can be checked against M45.7.1. A separate 10,000-update matrix holds *all*
other settings fixed to determine whether the effect is training-budget
limited. Do not interpret 2,000 vs 10,000 as equal compute; report elapsed
training time and number of updates.

## 3. Slurm-only execution

**All Python scripts, compilation, tests, training and diagnostics must execute
inside Slurm**, in the standard CUDA Singularity container. The launcher
requests GPU:1 as Calypso requires explicit GRES.

On the login node, submit commands only:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase

# Recommended: verify all stages in a short job array first.
sbatch --array=0-7%2 \
  --export=ALL,M45_7_2_STEPS=8,M45_7_2_GENERATED_SAMPLES=8,M45_7_2_VELOCITY_DRAWS=1 \
  scripts/slurm/m45_7_2_resampling.slurm

# Full matched 2,000-update comparison:
sbatch --array=0-7%2 scripts/slurm/m45_7_2_resampling.slurm

# Optional second-stage 10,000-update comparison:
sbatch --array=0-7%2 \
  --export=ALL,M45_7_2_STEPS=10000 \
  scripts/slurm/m45_7_2_resampling.slurm
```

The short smoke test does not provide meaningful generative statistics.
The 10,000-update sweep is optional until the first comparison is reviewed.

Inspect job status using `sacct -j <ARRAY_JOB_ID> --format=JobID,State,ExitCode,Elapsed`.
No results are overwritten. Output roots encode the job ID and training budget.

## 4. Output contract

Root directory:

```text
/scratch/coop/theret/GraphAttention_runs/m45_7_2_resampling/
  job_<ARRAY_JOB_ID>_<TASK>_<MODE>_<SUPERVISION>_<POLICY>_<STEPS>steps/
    summary.json
    environment.json
    shared_mode/block_average/<MODE>/
      checkpoint.pt
      metrics.json
      train_history.csv
    diagnostics/
      summary.json
      time_resolved.csv
      time_summary.csv
      generation_metrics.csv
      mode_amplitudes.csv
      spatial_correlations.csv
```

Log files:

```text
/scratch/coop/theret/GraphAttention/m45_7_resample_<ARRAY_JOB_ID>_<TASK>.out
/scratch/coop/theret/GraphAttention/m45_7_resample_<ARRAY_JOB_ID>_<TASK>.err
```

Share the eight `summary.json` files and `diagnostics/` CSVs. Retain the
checkpoint files on the cluster for later validation. The summary records
the residual policy, fixed reference normalization convention and
resampling seed rule. Check `max_Q_residual` in generation metrics.

## 5. Analysis priorities

1. **Matched fixed vs resampled** for the same architecture, supervision,
   seed and update count: held-out oracle-excess MSE as a function of FM time.
2. Compare **centered** mode variance, not just the uncentered second
   moment. The synthetic reference has zero expected modal mean.
3. Compare `rhou` spatial covariances, normalized correlations at physical
   grid lags, channel RMS and nullspace restriction defect.
4. Determine whether fixed vs resampled affects the FM and oracle objectives
   differently, and whether the change persists after 10,000 updates.

This identifies a training-distribution/optimization limitation but does
not establish that any particular architecture is universal. If full
conditional resampling and longer oracle training still cannot reproduce
the modal covariance, then investigate direct fine-resolution global
communication and expressivity tests *without lossy hierarchical
compression*. Keep M42 physical DNS/LES dataset construction out of scope.
