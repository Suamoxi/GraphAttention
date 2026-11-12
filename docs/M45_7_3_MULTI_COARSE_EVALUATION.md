# M45.7.3 — Eight-conditioning-field evaluation of six 10k static-oracle checkpoints

**Scope:** synthetic `shared_mode`, block-average `Q/B`, not physical DNS/LES.
**Status:** evaluation implementation only; launch and check Slurm jobs for runtime results.

## Question

M45.7.2 found that one 10,000-update, fixed-pair static-oracle checkpoint
(seed 2026) could recover approximately 46.5% of the reference centered
global-mode variance at converged Heun step counts. Other seeds recovered
much less. Evaluate **all six** saved 10k static-oracle checkpoints on
**all eight** held-out coarse fields at **200 Heun steps** to distinguish
training-seed sensitivity from conditioning-field sensitivity.

Neither the architecture nor its weights, training policy, fitted
normalization, or numerical solver is modified. The existing M45
checkpoint loader, projected Heun generator and correlation statistics
are reused.

## Comparison design

- Train seeds: 2026, 2027, 2028.
- Conditional residual training policies: `fixed`, `resample`.
- Existing 10k checkpoints: Slurm arrays 503042, 503043, 503044,
  tasks 2 (fixed) and 3 (resampled), respectively.
- Hold-out: `val_seed=7002026`, indices `0..7`, rebuilt using the
  exact original validation data generator, not freshly chosen coarse
  samples. **No retraining** and **no changes to checkpoints**.
- For each fixed validation coarse field, draw 128 independent initial
  source noises in `ker(Q)`, integrate the trained model with 200-step
  Heun, and generate 128 independent reference residuals from the
  synthetic conditional distribution `p(R_f | U_c)`.
- Source random-number streams are matched across all six checkpoints
  at the *same validation field and draw index*, and separated across
  validation fields. The physical sample scales still reflect each
  checkpoint's own training normalization.
- The reference streams are also shared across all six jobs and
  separately drawn at each coarse field.
- Validation index 0 uses the original M45.3 source and reference
  random seeds, so its 200-step results can be cross-checked against
  the earlier seed-2026 fixed checkpoint run.

### Primary metric

For each validation field \(U_c^{(i)}\), first center its generated
shared-mode amplitudes and compute

\[
  R_i =
  \frac{\widehat{\mathrm{Var}}\left(a_{\rm generated}\mid U_c^{(i)}\right)}
       {\mathrm{Var}_{\rm analytic}\left(a\mid U_c^{(i)}\right)}.
\]

The analytical denominator is the exact conditional zero-mean mode
variance for **that coarse field**. This avoids adding finite-sample
noise from the reference denominator. Report `100*R_i` as a percent.

The empirical reference variance is retained as a secondary check.
**Do not pool modal amplitudes across different coarse states before
centering.** Such pooling would introduce between-condition mean
effects and bias covariance recovery.

Additional metrics: signed mode mean, uncentered second moment,
channel-by-channel physical RMS (`rho`, `rhou`, `rhov`, `rhow`,
`rhoE`), within-field spatial covariance/correlation on all three
axes, and maximum `Q R_f` defect.

### Hierarchy of uncertainty

The 128 generated draws characterize sampling error *within a fixed
checkpoint and coarse field*. The eight coarse fields quantify
conditioning-field variation. The three independent training seeds
are the proper replication units for claims about training policies.
Do not interpret the 24 seed/coarse pairs as 24 independent trained
models; with three seeds, cross-seed uncertainty remains substantial.

## Slurm setup and execution

All Python compilation, pytest, generation and aggregation run **only
inside Slurm jobs in the pinned Singularity environment**. Login-node
commands below perform Git and Slurm submission only.

Paths in `scripts/slurm/m45_7_3_condition_sweep.slurm` refer to the
six already completed checkpoint directories under
`/scratch/coop/theret/GraphAttention_runs/m45_7_2_resampling/`.
Every task verifies checkpoint existence and metadata before generating.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
bash -n scripts/slurm/m45_7_3_condition_sweep.slurm
bash -n scripts/slurm/m45_7_3_aggregate.slurm

# Smoke test: all six checkpoint paths, 1 field, 4 draws, 2 Heun steps.
smoke_id=$(sbatch --parsable --array=0-5%2 \
  --export=ALL,M45_7_3_MAX_FIELDS=1,M45_7_3_GENERATED_SAMPLES=4,M45_7_3_HEUN_STEPS=2 \
  scripts/slurm/m45_7_3_condition_sweep.slurm)

# Run the CPU aggregation only after all 6 tasks succeed.
sbatch --dependency=afterok:"$smoke_id" \
  --export=ALL,M45_7_3_PARENT_JOB="$smoke_id",M45_7_3_MAX_FIELDS=1,M45_7_3_GENERATED_SAMPLES=4,M45_7_3_HEUN_STEPS=2 \
  scripts/slurm/m45_7_3_aggregate.slurm
```

If all six smoke-test tasks and their aggregation succeed, execute
the full comparison:

```bash
cd /scratch/coop/theret/GraphAttention
array_id=$(sbatch --parsable --array=0-5%2 \
  scripts/slurm/m45_7_3_condition_sweep.slurm)

sbatch --dependency=afterok:"$array_id" \
  --export=ALL,M45_7_3_PARENT_JOB="$array_id" \
  scripts/slurm/m45_7_3_aggregate.slurm

sacct -j "$array_id" --format=JobID,State,ExitCode,Elapsed
```

A smoke test is not a scientific result. The full run uses the default
200 Heun steps, 128 draws and all eight held-out coarse fields.

## Results contract

```text
/scratch/coop/theret/GraphAttention_runs/m45_7_3_multi_coarse/
  job_<EVALUATION_ARRAY_ID>/
    task_0_2026_fixed/
      summary.json
      per_field.csv
      spatial_correlations.csv
      mode_amplitudes.csv
    task_1_2026_resample/
      ...
    task_2_2027_fixed/
      ...
    task_3_2027_resample/
      ...
    task_4_2028_fixed/
      ...
    task_5_2028_resample/
      ...
    aggregate/
      summary.json
      paired_per_field.csv
      per_seed.csv
      per_validation_field.csv
```

Each `per_field.csv` has one row per validation coarse field
(`validation_index=0..7`). All diagnostic samples and full spatial
correlations remain accessible in their respective CSV files.
Aggregate `paired_per_field.csv` contains the 24 matched
seed-by-validation-index fixed-versus-resampled comparisons.

The aggregation refuses incomplete matrices, unexpected seed/policy
configurations, mismatched evaluation budgets or disagreement in
reference statistics for the same coarse field. It reports differences
in percentage points, averages **within** each seed over the eight
fields, and only then averages over the three independent seeds.
Inspect `per_validation_field.csv` for conditions where one model
fails despite good seed-averaged performance.

Slurm logs:

```text
/scratch/coop/theret/GraphAttention/m45_7_3_coarse_<ARRAY_ID>_<TASK_ID>.out
/scratch/coop/theret/GraphAttention/m45_7_3_coarse_<ARRAY_ID>_<TASK_ID>.err
/scratch/coop/theret/GraphAttention/m45_7_3_agg_<AGG_JOB_ID>.out
/scratch/coop/theret/GraphAttention/m45_7_3_agg_<AGG_JOB_ID>.err
```

For review, share the `aggregate/` CSV and JSON files together with
at least the six task `per_field.csv` files. Check per-task logs if
Slurm does not create all six outputs.

This experiment isolates conditioning robustness for the existing
learned velocity fields; it cannot by itself identify whether remaining
errors are due to training optimization, missing conditional support or
architecture.
