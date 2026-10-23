# M43 — Single-graph adaLN broadcast fast path and paired GH200 gate

**Status:** implemented on a separate branch, NOT yet run on Grace GH200.
The 8-epoch M43-B scientific checkpoint is unchanged and not resumed.

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
