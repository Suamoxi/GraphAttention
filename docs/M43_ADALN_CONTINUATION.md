# M43-B: controlled, isolated multi-update continuation gate

**Status: implemented, not executed on Calypso.** This is a numerical
diagnostic, not scientific training resumption. Draft PR #42.

## Motivation and experiment

The M43 single-graph adaLN fast path removed 11 expensive indexing
backward kernels and accelerated a GH200 129^3 microbatch by ~4.13x.
Original-vs-optimized predictions and gradients vary no more than
original-vs-original GPU replays, but we still need to test actual
training trajectories across several full M43-B six-mesh updates.

The new `scripts.validate_m43_adaln_continuation` reuses the
**unchanged scientific `train_epoch` function** and the original
train-only scaler, source scientific signature, and saved eight-epoch
model/AdamW/RNG state. It uses the next original epoch's deterministic
`balanced_case_interleave` ordering, with exactly one physical
snapshot per each of the six cases on every optimizer step:

| Case | Nodes |
| --- | --- |
| Re54/Ma0.05 | 65^3 |
| Re94/Ma0.05 | 129^3 |
| Re54/Ma0.2 | 65^3 |
| Re94/Ma0.2 | 129^3 |
| Re54/Ma0.3 | 65^3 |
| Re94/Ma0.3 | 129^3 |

This is genuine per-physical-sample Flow Matching on every mesh with
one sampled time and noise per case. The loss is the equal-sample
mean across six physical graphs, independent of their node counts.
The original AdamW update and activation checkpointing are unchanged.

There are four sequential independent trajectories on one GH200,
each restored from the SAME original checkpoint, AdamW and training
generator state:

- `indexed_a`: old graph-row indexing, reference
- `expand_a`: optimized single-graph expand
- `indexed_b`: old code replay, estimates baseline CUDA variation
- `expand_b`: new code replay, estimates optimized CUDA variation

Default: three optimizer updates/arm, 18 distinct training snapshots
per arm (three 65^3 and three 129^3 samples on each update).

After EACH update, the observer runs deterministic held-out validation
(default one sample per physical case, six total), preserving the model
training flag and global RNG. It saves in diagnostic CPU memory the
complete updated model parameters, original per-parameter gradients,
and AdamW state, plus a generator-state SHA256 and case-specific losses.
**No test-set samples are read or evaluated.**

The final report compares code implementations at every step and both
same-code replays. It reports combined relative L2 of full gradients,
updated model parameters and AdamW moments; the worst significant
per-parameter errors; train and per-case validation velocity MSE; and
forward/backward timings by mesh resolution.

## Execution on Calypso

Use the separate M43 optimization worktree. Do not switch the source
scientific training checkout:

~~~bash
cd /scratch/coop/theret/GraphAttention_m43_fastpath
git fetch origin
git checkout --detach origin/feature/m43-b-adaln-single-graph-fastpath
~~~

Run small CPU/GPU correctness tests (no continuation):

~~~bash
sbatch --export=ALL,M43_CONT_REPO="$PWD",M43_CONT_STAGE=tests scripts/slurm/m43_adaln_continuation.slurm
~~~

After tests pass, run three optimizer steps on each of four arms:

~~~bash
sbatch --export=ALL,M43_CONT_REPO="$PWD",M43_CONT_STAGE=continue,M43_CONT_STEPS=3,M43_CONT_VALIDATION_PER_CASE=1 scripts/slurm/m43_adaln_continuation.slurm
~~~

The separate output directory is automatically named:

`/scratch/coop/theret/GraphAttention_runs/m43_adaln_cont_JOBID/`

It contains:
- `provenance.json` — scientific original signature, exact sample IDs,
  hardware, source run, and original 8-epoch checkpoint details.
- `continuation_progress.json` — incremental per-arm summaries
  without dumping model or optimizer state.
- `continuation_summary.json` — per-step results, same-mode
  reproducibility, cross-mode gradients/weight/optimizer moments and
  a provisional numerical compatibility report.

Environmental overrides: M43_CONT_TRAIN_RUN, M43_CONT_DNS_ROOT,
M43_CONT_REFERENCE_DIR, M43_CONT_OUTPUT, M43_CONT_STEPS (2–6),
M43_CONT_VALIDATION_PER_CASE (1–3).

## Numerical acceptance and scientific limitations

The runner defines **provisional diagnostic thresholds** at EACH step:

| Check | Maximum |
| --- | --- |
| Combined updated model parameters relative L2 | 5e-5 |
| Combined full parameter gradients relative L2 | 5e-4 |
| Mean six-case held-out validation MSE absolute difference | 3e-3 |

The two same-mode baselines are reported independently. If a provisional
threshold fails, the GH200 job exits nonzero **after preserving results**.
These thresholds are not a replacement for the original strict operator
correctness tests, nor automatic permission to resume scientific M43-B.

The reference implementation's 129^3 GPU sparse operations already
vary across identical replays; small-model analytical parity and
GPU reproducibility are separate questions. No bitwise repeatability
claim is made.

The original source experiment files — run_signature, split,
train-only standardizers, both checkpoints, history JSON/CSV — are
SHA256 checked before and after the run. Source reference YAML checksums
and all selected HDF5/mesh file metadata are checked against the original
scientific manifest. Output directories under the source run or source
data are rejected. Existing diagnostic output is never overwritten.
No checkpoint or train history is saved in the diagnostic output.

This small validation subset detects divergence but is not full-epoch
statistical evidence. The 65^3/Re54 and 129^3/Re94 comparison also
confounds resolution with Reynolds number; the test makes no resolution
convergence or SGS physics claim.

**Next decision after receiving the new outputs:** inspect cross-code
parameter/gradient drift against both same-code replay controls across
all steps and all six cases. Only if acceptable should we design an
explicitly audited migration strategy for the ORIGINAL experiment
checkpoint, with full per-case validation and an intact scientific
provenance chain. Do not silently bypass the current resume guard.

## Singularity Git provenance failure (fixed)

A continuation attempt reached the real diagnostic runner but raised
`CalledProcessError: git rev-parse HEAD (exit 128)` while trying to record
`benchmark_git_revision` from inside the Singularity container. The container
may not resolve the detached worktree Git metadata or trust its ownership;
the captured traceback does not disclose Git's original stderr, so the exact
Git-side reason was not established. This is an execution/provenance failure,
**not an optimizer or adaLN numerical failure**.

The fixed Slurm script performs `git rev-parse --verify HEAD` on the host
**after** checking clean worktree, reviewed ancestry and allowed files, then
passes `--git-revision <full SHA>` explicitly to the Python diagnostic.
The runner validates the 40-character SHA and stores it without attempting
to execute Git inside the container. The scientific input integrity gates
remain unchanged.

The prior failed job can leave an empty diagnostic folder, because its
output directory was created before provenance was recorded. Never reuse
that directory; a new `sbatch` produces a fresh job-ID-specific output.
The container messages about read-only CUDA compatibility files are separate
from the Git exit-128 failure.
