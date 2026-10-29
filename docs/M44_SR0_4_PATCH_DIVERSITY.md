# M44 SR-0.4 — Fixed-update-budget spatial training-patch diversity

## Objective and baseline

SR-0.3 already guarantees point-sampling consistency and reduces the
common 604-free-node Flow Matching velocity MSE by about 4–5% relative
to SR-0.1. However, on four later DNS test snapshots the generated
residual RMS is consistently too small, especially for rhou (about
0.545 times the DNS target). SR-0.4 tests whether more **spatial**
training examples improve validation generalization, without changing
the DiNAT-DiT architecture, native FM path, update budget, scales,
held-out snapshot files or validation/test patch origins.

The paired observation is still **point-sampled DNS, NOT LES filtering**.
The dnstoles repository is untouched.

## Controlled ablation

- Keep the 24 selected source snapshots and same chronological split:
  15 train, 3 validation, 4 test, 2 embargo.
- Keep the existing 9³ fine-node / 5³ coarse-node patch, 4-layer,
  64-wide, 4-head, 10-channel DiNAT-DiT, 50-step Heun generation.
- Train pool: 16 deterministic interior nonoverlapping-cell patch
  origins for **each training source snapshot**. The first two
  fixed SR-0.3 origins are included, and remain the only validation
  and test origins.
- Per epoch, choose just **two origins per train snapshot** from a
  snapshot-seeded shuffled cyclic permutation. The same snapshot sees
  all 16 pool origins once in 8 epochs; the cycle repeats thereafter.
  Epoch index, snapshot index and experiment seed fully determine
  selection without consuming the optimizer's FM RNG stream.
- Optimizer updates: 15 train snapshots × 2 patches = 30 per epoch,
  50 epochs = 1500, exactly the original SR-0.3 budget.
  Distinct available train patches: 15 × 16 = 240, versus 30 originally.
  CPU data qualification and resident-pair memory increase even though
  optimizer updates remain fixed.
- Freeze the exact original train-only coarse mean/std and residual RMS
  from constrained SR-0.3 job `484619`, rather than refitting them to
  the new 240-patch pool. The training source manifest, reference data,
  geometry, snapshot identity, model spec, seed and optimizer protocol
  must match. Hashes of the original manifest and scales are pinned
  in the new manifest.
- Intrinsic `point_sample_constraint=True` remains active throughout.
  The saved manifest records the complete pool, epoch-sampling policy
  and scaling provenance. Exact restart reconstructs epoch selections.

## Run inside grace Slurm + Singularity (no login-node Python)

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch scripts/slurm/m44_sr04_patch_diversity.slurm
```

Default output:
`/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_<JOB_ID>/`.

The Slurm launcher uses the established
`/softs/local_arm/singularity/images/pyg25.03.sif` image and
`/scratch/coop/theret/avbpML_env/bin/python3`, executing **both**
pytest and training inside Singularity on the `grace` partition.
It checks a clean Git tree, frozen source availability, and output
non-overwrite. The preflight tests include an end-to-end synthetic
HDF5 train/resume/diagnose experiment.

Optional environment variables:
`M44_SR04_FROZEN_SCALES_RUN`, `M44_SR04_POOL_SIZE`,
`M44_SR04_EPOCHS`, `M44_SR04_OUTPUT`, and `M44_SR04_RESUME`.
For resume, set `M44_SR04_RESUME=1` and point
`M44_SR04_OUTPUT` to the existing SR-0.4 job directory.

The 50-epoch run must create a new directory; **do not resume**
the SR-0.3 optimizer, because the new data schedule is a different
training experiment. The checkpoint continues using best-validation
model selection, not test-set model selection.

## Read-only comparison diagnostics

After a successful SR-0.4 run, evaluate its best checkpoint:

```bash
cd /scratch/coop/theret/GraphAttention
sbatch --export=ALL,M44_SR02_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_<JOB_ID>,M44_SR02_PATH_REPEATS=16,M44_SR02_DRAWS=16,M44_SR02_MAX_PATCHES=4 \
  scripts/slurm/m44_sr02_diagnostics.slurm
```

The diagnostic runner recognizes an SR-0.4 manifest and evaluates
the **original two fixed train patch origins only**, not the full
pool, for a fair paired comparison against job 484619.
Validation and test pairs are unchanged and must reproduce the saved
best-epoch validation loss exactly.

In addition to the standard diagnostic report, each generated ensemble
now reports:

- RMS of the stochastic ensemble-mean residual;
- ensemble-mean residual bias relative to one DNS realization;
- RMS of the pointwise ensemble standard deviation;
- reference spatial residual mean and standard deviation;
- generated versus reference RMS residual increments along unique
  minimum-distance edges on the fine Cartesian graph, at one
  nondimensional separation `separation_over_L_ref`.

These metrics are descriptive. One-scale increments are **not**
a turbulent kinetic-energy spectrum, and no conditional PDF or SGS
validity is established.

## Paired SR-0.3 versus SR-0.4 report

Use the **new** SR-0.4 paired comparator because both architectures have
`point_sample_constraint=true`. Do not use the older SR-0.1 versus
SR-0.3 comparator: it expects one unconstrained checkpoint.

The comparator requires a fresh SR-0.3 checkpoint diagnostic produced
by this revision of `scripts.diagnose_m44_sr0`, because the previously
saved SR-0.3 report lacks `residual_spatial_moments`.
Submit its read-only evaluation on grace:

```bash
cd /scratch/coop/theret/GraphAttention
sbatch --export=ALL,M44_SR02_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_484619,M44_SR02_PATH_REPEATS=16,M44_SR02_DRAWS=16,M44_SR02_MAX_PATCHES=4 \
  scripts/slurm/m44_sr02_diagnostics.slurm
```

Once the SR-0.4 checkpoint diagnostic above has completed, submit
the paired comparison, supplying the **SR-0.4 training run**:

```bash
sbatch --export=ALL,M44_SR04_DIVERSE_RUN=/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_<SR04_TRAIN_JOB_ID> \
  scripts/slurm/m44_sr04_compare.slurm
```

For each training source, the comparison launcher discovers the associated
`m44_sr02_diagnostics/job_*/diagnostic_summary.json` using an exact
`source_run_dir` match. It refuses ambiguity; use
`M44_SR04_BASE_REPORT` and/or `M44_SR04_DIVERSE_REPORT`
to select the intended reports explicitly when multiple are found.
The launcher tests and executes all Python inside Singularity under grace
and writes:

`/scratch/coop/theret/GraphAttention_runs/m44_sr04_paired_comparison/job_<SLURM_JOB_ID>/comparison_summary.{json,md}`

The new comparator checks frozen-scale provenance against the original
SR-0.3 manifest, matched sample IDs, evaluation support, noise/time
path count, DNS target moments, generation draws, and fine-graph increment
separations. It fails closed if any physical reference or comparison
support does not match.

## Acceptance and interpretation

Compare SR-0.3 and SR-0.4 on identical validation snapshots, free-node
FM paths, point-sampling consistency, per-field generation RMS and
spatial increments. Prefer validation metrics for decisions; inspect
test only for final reporting. Repeating over multiple training
seeds will ultimately be necessary to separate patch-diversity effects
from optimization randomness.

Adding 240 spatial patches does not create 240 independent
turbulent realizations: the training data still come from 15
temporally correlated DNS snapshots.

## TensorBoard logging for M44 training

The standalone `scripts.train_m44_sr0` loop does not use Lightning's
automatic TensorBoard logger. It now accepts the opt-in `--tensorboard`
flag. The SR-0.4 Slurm training launcher enables this **by default**
(`M44_SR04_TENSORBOARD=1`, set to 0 only to disable), and legacy
SR-0.1 Slurm launcher uses `M44_SR01_TENSORBOARD=1` by default.

TensorBoard events are written under
`m44_sr04/job_<TRAIN_JOB_ID>/tensorboard/` with epoch-level tags:

- `loss/train_fm_velocity_mse`
- `loss/validation_fm_velocity_mse`
- `loss/best_validation_fm_velocity_mse`
- `optimizer/learning_rate`, `optimizer/updates_per_epoch`
- `time/epoch_seconds`

The writer does not modify losses, model weights, optimizer RNG, the
original `history.json` or checkpoint format. On an exact checkpoint
resume it uses TensorBoard's `purge_step` convention. If logging is
enabled upon resuming a historical run that had no TensorBoard events,
past epoch scalars are imported from the saved checkpoint history.

**For a run already launched without TensorBoard**, editing the
repository does not inject a writer into its running Python process.
Instead create event files from the existing `history.json` with a
separate grace Slurm job (no retraining):

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase
sbatch --export=ALL,M44_TB_SOURCE=/scratch/coop/theret/GraphAttention_runs/m44_sr04/job_<TRAIN_JOB_ID> \
  scripts/slurm/m44_tensorboard_backfill.slurm
```

This imports the history currently on disk into
`job_<TRAIN_JOB_ID>/tensorboard/`. If the training run continues to
write `history.json` but has no native event writer, the previous
import can be refreshed after additional epochs using
`M44_TB_REFRESH=1` (TensorBoard purge and full-history replay).
Do **not** use history refresh simultaneously with native event logging.
The source checkpoints are never modified. All conversion Python and
unit tests execute through Singularity on grace.

