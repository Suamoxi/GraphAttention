# M44 SR-0.5 — Matched longer training + complete two-origin generation diagnostics

## Why this experiment

M44 SR-0.3 (2 fixed origins/snapshot) and SR-0.4 (16-origin training pool,
2 origins/snapshot/epoch) both trained for 50 epochs, with 30 optimizer
updates per epoch. SR-0.4 improved the matched validation FM MSE by about
2.7%, but lost about 1.8% on held-out test snapshots, reduced the
wrong-coarse-condition gap, and did not consistently improve generated
DNS-residual amplitudes.

**Question:** Is 50 epochs / 1500 gradient updates insufficient for the
16-origin pool? Do not alter the architecture, frozen scales, patch
geometry, training/evaluation splits, initial weights, optimizer, or
conditioning objective.

## Controlled protocol

- Continue the original constrained SR-0.3 training job **484619**
  and diversified SR-0.4 training job **485096** from their **last.pt**
  model, optimizer and Gaussian Flow Matching generator RNG states.
- **Never resume in the original directories**. The new
  `scripts.prepare_m44_sr05_continuation` verifies the complete
  source manifest, scaler, optimizer/RNG state and 50-epoch history,
  then copies the source files into a *new*, fail-if-exists directory.
  A separate `continuation_provenance.json` records SHA-256 hashes,
  the source run, original and target epochs, and optimizer budget;
  the checkpoint manifest is copied byte-for-byte unchanged.
- Train each continuation to **150 total epochs**: 4500 optimizer updates
  per model (an additional 100 epochs/3000 updates). Both retain
  best-validation checkpoint selection and their original full
  training-data protocols.
- The original baseline sees the same 30 fixed training patches;
  diversified training continues to rotate 2 of its 16 origins per
  source snapshot through an exact 8-epoch cycle.
- Train-only SR-0.3 normalization remains frozen in both protocols;
  validation/test still comprise the same 3/4 chronological snapshots
  and the same 2 spatial origins per snapshot. This is **not**
  physically filtered LES: coarse observations are point sampled DNS.
- TensorBoard is enabled on both continuations by default; the copied
  50-epoch history is imported into a new run-specific TensorBoard log.
  No original event files, run histories or checkpoints are modified.
- **No optimizer state or RNG is reset** on continuation. The actual
  training step index continues from epoch 50. The original 50-epoch
  run remains a separately reproducible reference.

## Submit two independent grace Slurm + Singularity continuations

From the cluster login node, only use Git and Slurm:

```bash
cd /scratch/coop/theret/GraphAttention
git switch main
git pull --rebase

sbatch --export=ALL,M44_SR05_MODE=baseline \
  scripts/slurm/m44_sr05_continue.slurm

sbatch --export=ALL,M44_SR05_MODE=diverse \
  scripts/slurm/m44_sr05_continue.slurm
```

Outputs are created separately as:
`/scratch/coop/theret/GraphAttention_runs/m44_sr05/baseline_job_<SLURM_JOB_ID>/`
and
`/scratch/coop/theret/GraphAttention_runs/m44_sr05/diverse_job_<SLURM_JOB_ID>/`.

Each launcher executes pytest, source-checkpoint cloning and resumed
training through the established `pyg25.03.sif` Singularity image on
the **grace** partition. Never execute any Python on a login node.

To resume a genuine interrupted new continuation (no restart of the
original job):

```bash
sbatch --export=ALL,M44_SR05_MODE=baseline,M44_SR05_RESUME=1,M44_SR05_OUTPUT=/scratch/coop/theret/GraphAttention_runs/m44_sr05/baseline_job_<ORIGINAL_CONTINUATION_JOB_ID> \
  scripts/slurm/m44_sr05_continue.slurm
```

Use `M44_SR05_MODE=diverse` with the diversified directory to resume
that run instead. The provenance and copied source SHA hashes are
rechecked before loading optimizer state.

The launcher allows `M44_SR05_TOTAL_EPOCHS` only if both jobs use the
same target epoch budget; the default and recommended controlled value
is 150. The comparison refuses mismatched epoch/update totals.

## Read-only diagnostics: both validation/test patch origins

After **both** new training runs complete, submit the diagnostic
Slurm job for **each** continuation:

```bash
BASE=/scratch/coop/theret/GraphAttention_runs/m44_sr05/baseline_job_<BASELINE_CONTINUATION_JOB_ID>
DIVERSE=/scratch/coop/theret/GraphAttention_runs/m44_sr05/diverse_job_<DIVERSE_CONTINUATION_JOB_ID>

sbatch --export=ALL,M44_SR02_SOURCE="$BASE",M44_SR02_PATH_REPEATS=16,M44_SR02_DRAWS=16,M44_SR02_MAX_PATCHES=4,M44_SR02_ALL_GENERATION_ORIGINS=1 \
  scripts/slurm/m44_sr02_diagnostics.slurm

sbatch --export=ALL,M44_SR02_SOURCE="$DIVERSE",M44_SR02_PATH_REPEATS=16,M44_SR02_DRAWS=16,M44_SR02_MAX_PATCHES=4,M44_SR02_ALL_GENERATION_ORIGINS=1 \
  scripts/slurm/m44_sr02_diagnostics.slurm
```

The `--all-generation-origins` option changes ensemble coverage to
6 validation patches (3 snapshots x 2 origins) and 8 test patches
(4 snapshots x 2 origins), 16 draws per patch and 50-step Heun
generation. The original default, one origin per source snapshot,
remains unchanged for historical diagnostics. The maximum-patches flag
counts **source snapshots**, not patch origins, when both are included.

The new
`generation_ensemble_by_snapshot` report groups the two origins
within each source snapshot and calculates separately for each of the
five CFD conservative variables:

```text
E_draw,node[r^2] = E_node[(E_draw r)^2] + E_node[Var_draw(r)]
     full moment    squared ensemble mean    stochastic ensemble variance
```

Only **source DNS snapshots** are statistical units; neither spatial
patch origins nor 16 noise draws increase the number of independent
flow realizations. The decomposition reports RMS/DNS, mean-RMS/DNS,
spread-RMS/DNS, mean and variance contributions, and nearest-neighbor
residual increments. These do **not** validate a conditional PDF or a
turbulent energy spectrum.

## Strict paired comparison

After both diagnostic jobs succeed:

```bash
sbatch --export=ALL,M44_SR05_BASELINE_RUN="$BASE",M44_SR05_DIVERSE_RUN="$DIVERSE" \
  scripts/slurm/m44_sr05_compare.slurm
```

The launcher auto-discovers a single checkpoint diagnostic for each
run using **top-level** `source_run_dir`, not the nested frozen-scaler
source. If multiple diagnostics exist, specify
`M44_SR05_BASELINE_REPORT` and `M44_SR05_DIVERSE_REPORT`
as absolute paths.

It runs paired reference/epoch/provenance checks and unit tests
**inside Singularity on grace**, and writes read-only outputs under
`GraphAttention_runs/m44_sr05_paired/job_<COMPARE_JOB_ID>/`:

- `comparison_summary.json`
- `comparison_summary.md`

Compare validation first. The chronological test set is held out for
final reporting and must not be used to choose an epoch budget, patch
pool or architecture. One seed and 3 validation snapshots remain
insufficient for uncertainty claims; repeat seeds is a later step.
