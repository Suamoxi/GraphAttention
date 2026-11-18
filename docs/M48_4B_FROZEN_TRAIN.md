# M48.4b — frozen 44-teacher normalization and matched FM training

## Scope and evidence

This is the next controlled real-data training stage after:

- M48.3 CEDAR / partition qualification, job 510080.
- M48.4a two-source, two-architecture optimizer qualification, job 510111.
- M48 normalization audit of 44 teacher snapshots across 22 physical
  cases, job 510184. The audited original JSON is identified by SHA-256
  defcf3ea5ea19644ab2f9b0bd507575ade316f7dc5582fce5fb10eb1e6daf1f4.

**Only the teacher role is available to optimization.** Neither the
validation role nor the locked-in-regime test role is opened by this
training runner. No MS_SR source HDF5, manifest, or original CEDAR code is
modified.

Implementation state: GitHub source and synthetic regression tests are
prepared. The new training runner is **not Calypso-qualified** until
the two mode-specific runs have passed on real data.

## Fixed numerical preprocessing

The repository-tracked file
\`configs/m48/teacher_44_w1p25_l1_q4_q8.json\` freezes exactly:

- published MS_SR manifest SHA-256
  \`993f791f88d801e44e5feb0cee927c4c48eaa459bd6171e9e364b002bb01fdca\`;
- the case/snapshot/source/operator identity selection hash
  \`32119fababe014f19d3d1fcfa0e06e4e6527f1e8a66d24cf2f84650e00155ca8\`;
- 44 teacher snapshots, 22 cases, filter \`w1p25_l1\`, coarse q8/fine q4;
- the all-teacher, equal-snapshot-weighted coarse mean and standard
  deviation, and fine CEDAR residual RMS;
- named conservative components \`rho,rhou,rhov,rhow,rhoE\` in that order;
- the original dimensional MS_SR field convention and the audit file hash.

| Component | Frozen coarse mean | Frozen coarse std | Frozen residual RMS |
|---|---:|---:|---:|
| rho | 1.1566685830 | 0.2905924987 | 0.0328722402 |
| rhou | -1.1810174875 | 138.4811342 | 20.9126828 |
| rhov | -3.8266518399 | 138.2737115 | 21.0880950 |
| rhow | -2.1115302470 | 139.9969689 | 21.2055310 |
| rhoE | 276040.0204872 | 104628.8427294 | 11345.6261511 |

Full precision is stored in JSON; this table is rounded.

For each sample and component, let \`C\` be the original CEDAR coarse
restriction and \`P\` its original lifting operator. The task is

    B = P Uc, R = Uf - B, C R = 0
    coarse_input = (Uc - mu_c) / std_c
    lifted_condition = (B - mu_c) / std_c
    X1 = R / rms_R
    X0 = (I - P C) eps        # full fine field; eps ~ N(0, I)
    Xt = (1 - t) X0 + t X1
    v_target = X1 - X0

No residual mean subtraction or case-specific residual scaling. Fixed
per-channel scales commute with the per-channel linear CEDAR restriction,
preserving its residual nullspace. No CEDAR filtering or projection is
performed on artificial patch boundaries.

The 44-source statistics are used for **all** model modes and future
validation/inference. The old M48.4a two-source scalers are not reused.

## Training protocol

\`scripts/m48_4b_frozen_train.py\` reuses the already-qualified
\`scripts.m48_ms_sr_train._prepare_pair\` and \`_one_step\`. There is no
new physical loader, no new CEDAR inverse, and no new FM objective.

- Each epoch traverses all 44 teacher sources in a seeded shuffle.
- Every source contributes the same number of patch optimizer steps, so
  the larger ReT100 mesh does not get extra training weight by node
  count. A patch loss is the mean MSE on the owned fine nodes.
- Random periodic owned-core origins, FM times, and globally projected
  noise are derived deterministically from the global step and seed.
- **Static and dynamic run in separate processes**, with the same source
  schedule, seeds, owned patches, optimizer, hidden width and layer-depth
  settings. Each run gets its own optimizer/checkpoint.
- Only one source is cached at a time, so repeated patches of that
  source avoid redundant HDF5 loads. No full-campaign RAM/GPU cache.
- Each step logs loss, preclipping gradient norm, fine/coarse read-node
  counts, owned node count, CEDAR projection defect, synchronized
  forward/backward time, and GPU peak allocated bytes.
- Checkpoints contain the model and AdamW state, RNG state, global
  optimizer step, training history, and configuration/source
  fingerprint. Resume rejects changed source selection, normalization,
  mode, architecture, seed, schedule or source-code commit.
- Checkpoints and summary are written atomically at intervals and at
  normal exit. If a job is interrupted, uncheckpointed steps are
  intentionally replayed from the last saved step.

The default *proposed* training schedule is 8 epochs, 4 owned patches
per source, an owned 4x4x4 fine-node cube, width 32 and depth 1. This
corresponds to 44 * 4 * 8 = **1,408 optimizer steps per architecture**.
The default launcher executes only **4 steps per architecture** as
the initial real-data feasibility gate.

**Parameter budgets are not matched** at equal width: dynamic has
additional cross-scale feedback modules. This is a matched-configuration
comparison, not a capacity-controlled ablation. The two runs also have
different dependency halos. Observe speed and memory before launching
the full training.

## How to run on Calypso

From the original GraphAttention checkout:

    cd /scratch/coop/theret/GraphAttention
    git pull --rebase
    bash -n scripts/slurm/m48_4b_frozen_train.slurm

Submit one Grace GPU job for each mode:

    sbatch --export=ALL,M48_MODE=static scripts/slurm/m48_4b_frozen_train.slurm
    sbatch --export=ALL,M48_MODE=dynamic scripts/slurm/m48_4b_frozen_train.slurm

Each job runs the old and new tests before the bounded four real
optimizer steps. Outputs go to:

    /scratch/coop/theret/GraphAttention_runs/m48_4b_frozen_train/
        job_<JOBID>_static/ or job_<JOBID>_dynamic/
            config.json
            normalization.json
            checkpoint_latest.pt
            summary.json

After reviewing memory, step time, gradient norms and CEDAR errors,
the explicit long-run form is:

    sbatch --time=12:00:00 --export=ALL,M48_MODE=static,M48_MAX_STEPS=0,M48_CONFIRM_LONG_TRAIN=YES scripts/slurm/m48_4b_frozen_train.slurm
    sbatch --time=12:00:00 --export=ALL,M48_MODE=dynamic,M48_MAX_STEPS=0,M48_CONFIRM_LONG_TRAIN=YES scripts/slurm/m48_4b_frozen_train.slurm

The long jobs are an **option**, not yet an approved throughput estimate.
Use \`M48_MAX_STEPS=N\` for bounded chunks and
\`M48_RESUME_DIR=/path/to/existing/job_<JOBID>_<MODE>\` to resume a
checkpoint with identical mode, source, architecture and commit. Set
\`M48_CONFIRM_LONG_TRAIN=YES\` for more than four steps per invocation.

## Failure behavior and limitations

- Wrong manifest/source fingerprint, changed field order, missing source,
  altered frozen scales, or nonteacher data: fail before training.
- Nonfinite loss, zero/nonfinite gradients, or failed full-field CEDAR
  qualification: fail without silently skipping a source.
- Patch read-set cap violations or CUDA OOM: fail; reduce core/width/depth
  only in **new independent runs**, not midway through a checkpoint.
- CEDAR operators and source Gaussian noise are full-lattice CPU operations
  for every sampled step. Thus the per-step timer includes those
  operations and CPU patch construction; HDF5 source load occurs outside
  the timed step. Results are not pure network kernel timings.
- Validation is intentionally **not** part of this step-2 experiment.
  Full-lattice velocity assembly and globally projected Heun generation,
  validation per physical regime, turbulence spectra, and generalization
  to unseen cases remain separate gates.
- The 22 cases each have only two teacher snapshots. Statistical
  convergence/temporal independence cannot be inferred from this audit.
- Geometry and M48 test coverage include full-graph-versus-patch
  numerical/gradient comparisons for both architectures, but actual
  larger-core/depth equivalence and cluster runtime must still be
  checked before a scientific performance claim.
