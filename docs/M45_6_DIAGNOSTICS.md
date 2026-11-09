# M45.6 — reproducibility, analytical sampling, and matched depth qualification

These are **synthetic** block-average Q/B experiments, not LES/DNS validation. The
M45.4 and M45.5 model implementations and prior checkpoints are unchanged.

## 1. Reproducibility and exact partition equivalence

`scripts/m45_6_reproducibility.py` reuses M45 pilot dataset creation, normalization,
model initialization, path sampling, optimizer, gradient clipping, and patch
planner. Two separate Python processes generate matched `trace` runs using the
same seed. Each update records a SHA-256 of the CPU FM interpolation path,
parameter bytes after the AdamW update, unprojected gradient norm, and loss.
The `compare` step checks that **both initial parameter states and input paths
match** before locating the first differing parameter fingerprint. A matching
input path plus diverging parameters isolates numerical execution differences.

`parity` creates two initially identical models in one process, then trains
one on the dependency-complete patch (`patch_owned`) and the other on the
complete fine graph with exactly the same owned-node loss (`full_owned`).
Each update records prediction, loss, raw-gradient, and post-optimizer maximum
absolute differences. It is run at depths 2 and 4. Since patch context is
512/512 fine nodes on this qualification mesh, we expect close agreement.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_6_reproducibility.slurm
sbatch --export=ALL,M45_6_TASK=parity,M45_6_STEPS=32 \
  scripts/slurm/m45_6_reproducibility.slurm
```

By default `M45_6_DETERMINISTIC=off`, matching prior runs. Set `warn` to request
PyTorch deterministic algorithm warnings, or `strict` to force deterministic
algorithms and fail on unsupported operations. The Slurm job exports
`CUBLAS_WORKSPACE_CONFIG=:4096:8` before Python/CUDA initialization.
Report the execution mode; do **not** infer that a reproducible result from
one execution proves the sparse kernels deterministic on all devices.

## 2. Exact Gaussian transport positive control

`scripts/m45_6_oracle_transport.py` constructs the **symmetric covariance
square-root transport** from the projected standard Gaussian source to the
known Gaussian conditional target. It uses the spatial-block diagonal local
covariance and the shared-mode rank-one update. Its only dense eigendecomposition
is a matrix with `(coarse_side**3) * 5` entries per dimension, not the full fine
field covariance.

Let `D` contain the blockwise local covariances and `g` the global shared mode.
In each coarse block and channel the target covariance is `lambda_b * P + gg^T`
with block-average nullspace projector `P`. Decompose each block into a unit
vector aligned with `g_b` and orthogonal directions. The aligned coordinates
are coupled by the small matrix `diag(lambda) + h h^T`, where
`h_b = ||g_b||`. Apply its symmetric square root to these coordinates; apply
`sqrt(lambda_b)` independently to the orthogonal components. This is the exact
positive square root of the target covariance on `ker(Q)`.

On the same initial Gaussian sources, compare direct transport against Heun
integration of the **analytically optimal Flow Matching velocity**, with
25, 50, 100, 200, 400, and 800 steps. The paired relative RMSE measures
integration error directly; mode statistics and coarse consistency are recorded
as separate checks. Use more cheap direct samples than costly Heun trajectories
to reduce uncertainty in the reference second moment.

```bash
sbatch scripts/slurm/m45_6_oracle_transport.slurm
# Optional faster qualification:
sbatch --export=ALL,M45_6_DRAWS=4,M45_6_DIRECT_DRAWS=64 \
  scripts/slurm/m45_6_oracle_transport.slurm
```

The default uses 32 paired Heun sources and 512 exact direct draws; do not
confuse sampling noise with Heun error. Results are `oracle_transport.csv`
and `summary.json`.

## 3. Matched 2-layer versus 4-layer full-field supervision

`scripts/slurm/m45_6_depth_full_all.slurm` launches a 12-run matrix:
- depth: 2, 4 (every fine/coarse stage);
- training seeds: 2026, 2027, 2028;
- training fields: 24, 256;
- common full-field supervised loss, 2,000 updates, fixed validation split,
  width 16, heads 4, `block_average`, r=2, and shared-mode distribution.

Adjacent array indices have the **same seed, dataset realization, and fitted
normalization**, differing only in architecture depth (and parameter count).
Both static and dynamic variants are trained in each job. Checkpoint evaluation
uses the same 128 generation seeds and 25 Heun steps, plus time-resolved oracle
velocity errors.

```bash
sbatch --array=0-11%2 scripts/slurm/m45_6_depth_full_all.slurm
```

After collecting results in the common root, submit the aggregation itself
as a **CPU-only Slurm job**, using the same Singularity/Python environment as
the training jobs. Do not run Python or pytest directly on the login node:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m45_6_depth_aggregate.slurm
```

Drive contains an explicit rerun: job_493421 has the same
(seed=2028, train_samples=256, depth=2) as job_493366. The Slurm
launcher defaults to excluding the *later* job_493421, and the output
summary records this choice in `excluded_jobs`. Override with
`--export=ALL,M45_6_AGG_EXCLUDE_JOB=none` for strict behavior,
which will reject duplicate configurations. No deduplication is silent.

Expected outputs:

```text
/scratch/coop/theret/GraphAttention_runs/m45_6_depth_comparison/job_<SLURM_JOB_ID>/
  summary.json
  per_job.csv
  matched_depth.csv
  aggregates.csv
```

Logs: `/scratch/coop/theret/GraphAttention/m45_6_aggregate_<SLURM_JOB_ID>.out`
and `.err`. The job requires all 24 per-model rows and 12 matched depth
comparisons; any missing run or unexpected duplicate causes a failure.

The comparator rejects duplicate `(seed, train_samples, depth, mode)` keys and
inconsistent training/validation configurations instead of silently merging
them. Files: `per_job.csv`, `matched_depth.csv`, `aggregates.csv`, `summary.json`.

## Result locations

- `GraphAttention_runs/m45_6_repro/job_<ID>/`
- `GraphAttention_runs/m45_6_oracle/job_<ID>/`
- `GraphAttention_runs/m45_6_depth_full_all/job_<ID>/`

Every Slurm job checks that the Git checkout is clean, syntax-compiles all new
scripts and tests, runs M45.6 unit tests, and refuses existing output paths.

**Interpretation limitations.** The shared-mode synthetic distribution is a
controlled Gaussian rank-one problem; matching it would not validate physical
CFD statistics. Matching fixed-seed runs says nothing about the statistical
accuracy of a learned FM vector field; conversely, convergence of the oracle
ODE solver does not validate the trained model.
