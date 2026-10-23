# M43 — Single-graph adaLN broadcast fast path and paired GH200 gate

**Status:** tested on Grace GH200 in job 479099. Substantial measured
speedup, but full checkpoint equivalence DID NOT pass. Further diagnostics
have been committed and require a new job. The scientific 8-epoch M43-B
checkpoint is unchanged and not resumed.

## GH200 result from Slurm job 479099 (measured, 23 October 2026)

Original source revision: `9789f13b9ac09d04b5d10f33ca90fd0c607803db`.
Actual host: `calypso-grace02` (GH200, 97,871 MiB advertised GPU memory).
All **16 CPU/CUDA reference tests passed**.

Three interleaved real-checkpoint full-size 129^3 trials per broadcast:
- Original index backward: ~20.04 s; expand backward: ~4.035 s.
  **Backward speedup: 4.97x.**
- Whole physical-microbatch step: **4.12x faster.**
- Peak allocated HBM: original **53.206 GiB**, expand **47.064 GiB**
  (saving **6.142 GiB**).
- CUPTI old broadcast: **11** indexing_backward_kernel<float,4>
  launches, **16.016 s** total; optimized: **0 launches**.

**Scientific gate FAILED** (`status=numerical_mismatch`), with:
- Predicted velocity: max absolute difference 1.382e-3, relative L2
  difference 1.565e-4.
- Loss: absolute difference 5.215e-8.
- Worst per-parameter gradient: relative L2 difference **2.933%**.
- Worst post-AdamW parameter: relative L2 difference **3.8e-6**.
- Worst AdamW state: relative L2 difference **1.05e-3**.

These are measured data from logs, not an assertion that the alternate
implementation has been shown scientifically equivalent.

The existing benchmark compared the first A-vs-B run but did NOT quantify
same-implementation output/gradient variability due to CUDA sparse atomic
reductions. The updated comparison now captures two independent replays
per arm and reports: original-vs-original, expand-vs-expand,
original-vs-expand, largest-gradient parameter names and absolute norms.
**It does not weaken any numerical acceptance tolerances or resume guard.**

Do not merge, continue scientific training, or change checkpoints merely
because the speedup is strong. First run the updated repeat diagnostics
with a NEW Slurm job ID and inspect the resulting
`same_mode_reproducibility` and `parity` sections.

## Observed profile

On Re94/Ma0.2 at 129^3, the original M37 checkpointed model on GH200
has 2,146,689 nodes, 10 DiT blocks plus the final modulated projection,
float32, and approximately 1.239s forward + 20.085s backward.
GPU peak allocated was about 53.165 GiB. The profile contains 11 launches
of indexing_backward_kernel totaling 15.910s, or 74.3% of observed CUDA
kernel time.

**Hypothesis, not yet proven:** ten block adaLN broadcasts and the final
one are responsible. The original code graph_modulation[batch_index] uses
a batch_index of 2.15 million zeros, so the gradient is accumulated through
highly contended indexed scatter into a single graph-level row.

## Production change

src/graph_attention/models/dit_common.py now chooses:

- graph count B=1: features.expand(num_nodes, -1) (an expand view).
- graph count B>1: features[batch_index] (EXACT original indexed fallback).

The existing _condition_embedding already validates packed batch indexing
and the expected B conditioning rows, before reaching this helper.

The change alters neither model parameter/state_dict shapes, graph topology,
relative geometric attention, absolute coordinate input, normalization,
Flow Matching, AdamW, nor M43-B physical sample weighting.

Different meshes processed in six successive microbatches are all B=1
and benefit independently from the fast path. Packed heterogeneous
meshes remain B>1 and use the original implementation unchanged.

## Tests and full scientific benchmark

tests/unit/test_m43_adaln_broadcast.py tests:

- Forward and gradient parity for single-graph broadcast.
- Complete 2-D DiNAT-DiT graph training with and without non-reentrant
  block checkpointing.
- Packed heterogeneous 4-node and 7-node unstructured graph topology,
  distinct coordinate extents, and distinct Flow Matching times.
- No cross-graph conditioning contamination, malformed batch rejection.
- Identical in-memory AdamW optimizer updates for one and two graphs.

scripts/benchmark_m43_adaln_broadcast.py uses the original REAL 129^3
sample and train-only standardizer, fully restores last.pt and AdamW,
and generates one fixed true Flow Matching time and noise realization.
Original reference helper is scoped-mocked ONLY within benchmark trials.
Both modes get warmups; all measured trials restore original checkpoint.
The benchmark alternates old/new order and reports synchronized forward,
backward including activation recomputation, optimizer time and GPU memory.

It also compares the entire predicted velocity field, all parameter
gradients, in-memory updated weights, and AdamW optimizer state with
explicit parity tolerances. CUPTI optionally measures the slow kernel
count and total kernel time in both modes.

No source training artifact is written. The test profiles ONE 129^3
physical sample, not the 6-sample accumulated M43 optimizer window.
Data loading and transfer are excluded from timed model steps.

## Calypso execution (separate worktree)

Create a separate detached source worktree:

    cd /scratch/coop/theret/GraphAttention
    git fetch origin
    git worktree add --detach ../GraphAttention_m43_fastpath origin/feature/m43-b-adaln-single-graph-fastpath
    cd ../GraphAttention_m43_fastpath

Test-only gate (no full-size benchmarking):

    sbatch --export=ALL,M43_FAST_REPO="$PWD",M43_FAST_STAGE=tests scripts/slurm/m43_adaln_broadcast.slurm

After tests pass, full GH200 paired benchmark:

    sbatch --export=ALL,M43_FAST_REPO="$PWD",M43_FAST_STAGE=benchmark,M43_FAST_TRIALS=3 scripts/slurm/m43_adaln_broadcast.slurm

The output is the separate scratch directory:

    /scratch/coop/theret/GraphAttention_runs/m43_adaln_fast_JOBID/

Files: provenance.json and benchmark_summary.json, including parity,
timing medians, backward speedup, HBM peak, slow-kernel comparison.

Environment options: M43_FAST_TRAIN_RUN controls which original last.pt
is used (default train_pilot_s42). M43_FAST_PROFILE_KERNELS=NO disables
additional CUPTI profiling for a lower-overhead timing-only gate.

## Scientific acceptance before resuming M43-B

1. CPU and CUDA correctness tests all pass.
2. Full 129^3 predictions and ALL model gradients pass tolerances.
3. AdamW parameter and optimizer state changes remain numerically equivalent.
4. The slow 11-kernel indexed backward concentration decreases and actual
   GH200 total/backward step time improves reliably without unacceptable HBM.
5. Recheck across a second physical sample if results are promising.
6. Review M43 checkpoint signature/revision compatibility explicitly.
   The existing logging-only resume gate correctly REJECTS a change
   to dit_common.py. It must not be bypassed silently.

CPU/GPU reduction ordering changes can create small floating differences:
mathematical equivalence does not imply bitwise identical GPU reductions.
No model speedup or resume compatibility has been asserted yet.
