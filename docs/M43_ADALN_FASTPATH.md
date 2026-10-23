# M43 — Single-graph adaLN broadcast fast path and paired GH200 gate

**Status:** two GH200 benchmarks completed (jobs 479099 and 479169).
The fast path improves full-volume microbatch time approximately 4.13x.
The strict numerical gate still exits nonzero, but same-mode replay controls
show matching numerical variability. The scientific eight-epoch M43-B
checkpoint remains unchanged and has not been resumed.

## GH200 result from Slurm job 479169 (23 October 2026)

The new same-mode replay controls have now been run. **All 16 small
CPU/CUDA tests passed.** Model, optimizer, and sampling inputs were
restored from the same original eight-epoch checkpoint before trials.
The checkpoint SHA-256 was identical before and after the benchmark.

| Comparison | Velocity prediction relative L2 | Worst relative gradient L2 |
| --- | ---: | ---: |
| Indexed vs expand | 1.564750e-4 | 3.640% |
| Indexed vs indexed | 1.565538e-4 | 4.333% |
| Expand vs expand | 1.564398e-4 | 4.020% |

The **cross-implementation error is comparable to the reference
implementation's own replay variability**. The largest relative gradient
errors are tiny geometry-MLP biases: for block 8 the gradient norm is
2.171e-8 and the cross-implementation difference norm is 7.904e-10.
For all 134 trainable parameter-gradient tensors combined, the
cross-implementation relative L2 difference computed from the JSON is
**1.3214e-5**; excluding near-zero geometry gradient tensors, the maximum
single-tensor gradient relative L2 is **2.3673e-5**. Across all
post-AdamW weight tensors, combined relative L2 is **7.929e-9**.

Performance is reproduced across 3 interleaved 129^3 trials per mode:
- Backward: **20.0633 s indexed / 4.0215 s expand = 4.989x** speedup.
- Forward: **1.1936 s indexed / 1.1210 s expand**.
- Full isolated physical microbatch: **21.2588 s / 5.1442 s = 4.133x**.
- Peak allocated GPU memory: **53.206 / 47.064 GiB**, saving **6.143 GiB**.
  Reserved memory remains **58.389 GiB** for both, due to allocator caching.
- CUPTI 11 pathological indexed-backward launches consuming 16.016 s
  become 0 launches, confirming this bottleneck was removed.

**Interpretation:** the specific B=1 gather-to-expand change is
mathematically identical, has passed single- and multigraph correctness
tests, and its full-volume deviations are not detectably larger than
the existing GPU sparse backend's own run-to-run variability. The same
implementation producing non-bitwise-identical results suggests
nondeterministic GPU reductions, but the exact responsible kernel
is not established.

**Acceptance-gate limitation:** the old strict prediction thresholds
(max absolute 1e-5 and relative L2 1e-5) are even tighter than
indexed-versus-indexed replay variability (1.5655e-4 relative L2).
Thus `status=numerical_mismatch` is expected even for self-comparisons.
Do **not** silently loosen the strict gate or treat one noisy comparison
as a mathematical proof. A separate documented numerical reproducibility
criterion is needed for CUDA benchmarks, alongside exact analytical and
small deterministic tests.

**Next scientific action:** verify that the same conclusion holds on
a second physical mesh/case and, before resuming original M43-B,
perform a checkpoint-compatible controlled continuation on an isolated
copy, including the real six-sample accumulation and validation metrics.
Do not bypass the existing scientific revision compatibility guard.

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
