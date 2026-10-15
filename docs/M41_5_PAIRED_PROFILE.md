# M41.5 — repeated 32/64 comparison and synchronized stage diagnostics

**Scope: computational scalability only**. Reuse the already prepared
M41.4 257^3 HDF5-derived one-hop CSR graph and exact-two-hop partition
files. No model retraining, additional graph construction, physical
quality evaluation or full Heun generation.

## Why this experiment

The completed 14-run M41.4 sweep established that:

| Lazy partitions | Median forward (2 trials) | Peak CPU RSS | GPU allocated |
|---:|---:|---:|---:|
| 32 (4x4x2) | 106.2875 s | 20.9924 GiB | 4.4139 GiB |
| 64 (4x4x4) | 110.6150 s | 19.9520 GiB | 2.2552 GiB |

The observed ~4% speed difference is small compared with the
limited two-trial evidence. The next experiment must establish
run-to-run variation with matched hardware allocation and identify
where M41.3's streamed executor spends its time.

## Two strictly separated measurement paths

1. **Ordinary latency:** SIX independent clean-process forwards
   at 32 and SIX at 64 partitions (12 total), on ONE Grace allocation.
   Trial order is AB/BA/AB/BA/AB/BA, so neither layout always
   benefits from the first access or host page cache. Every trial
   uses M37 best.pt, one physical graph, seed 5678 and flow time 0.
   These are UNPROFILED wall times (and per-process RSS/GPU peaks).
2. **Synchronized diagnostics:** ONE additional model forward
   per layout (2 extra). The stage profiler inserts
   torch.cuda.synchronize before/after each timed phase. It logs
   disjoint *wall times* for:
   - Partition NPZ loading and CPU CSR/geometric materialization;
   - CPU hidden read/gather;
   - Hidden data/index host-to-device stage;
   - AdaLN normalization and QKV projections;
   - CSR/relative-displacement host-to-device stage;
   - Geometric sparse attention/softmax/message aggregation;
   - Attention output projection/residual;
   - MLP;
   - Owned GPU-to-CPU readback and CPU scatter;
   - Jacobi hidden-buffer swap;
   - Input projection, final projection, and other fixed stages.
   
   Individual layer wall times and per-layer stage timing aggregates
   are recorded. The **blocking CPU readback is the actual per-partition
   completion dependency**. The Jacobi buffer swap itself is a cheap
   pointer exchange, not a hidden GPU barrier.

Important: timings from synchronized diagnostic forwards include
extra synchronization overhead and **must not be used in the ordinary
32 vs 64 speed comparison**. The attention stage includes
the geometry MLP, sampled sparse dot products, softmax and aggregation;
it is NOT only the GPU SpMM kernel time. Host CPU RSS means Linux
process peak RSS, not system-wide node RAM; GPU memory is PyTorch's
allocated/reserved peak, not instantaneous total HBM usage.

## Before the GH200 experiment

The Slurm script performs:

- Clean-repository, checkpoint and prepared-layout preflight.
- Python py_compile of new/modified code and tests.
- M41.3 small-graph exact M37 numerical equivalence regressions.
- New tests that check profiling does not alter the model forward on
  CPU, paired AB/BA ordering, and immutable CSR/coordinate reuse.
- M41.4 irregular-hex and unstructured-tet numerical parity tests.

The new profiler is **opt-in**; normal M41.4
forward_streamed_dinat calls have no CUDA synchronizations added.

## Single Grace job

Run:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_5_257_paired_profile.slurm
~~~

Default Grace resources: 1 GPU, 4 CPUs, 4 hours and **no explicit
memory limit**, same as the M41.4 jobs. The current prepared 32 and
64 graphs are reused unchanged; there is NO prerequisite rebuilding
or repartitioning job.

Output path by default:

    /scratch/coop/theret/GraphAttention_runs/m41_5_257_paired_profile/

Files:

- progress.json: exact checkpoint/seed/layout/repeat signature and
  resumable run tracking; guards against inadvertently mixing runs.
- timings/layout_032/repeat_XX/forward.json (6 independent repeats).
- timings/layout_064/repeat_XX/forward.json (6 independent repeats).
- timings/layout_NNN/repeat_01/forward.npy: first full output
  for elementwise 32-versus-64 consistency validation.
- profiles/layout_032/forward.json: synchronized stage times
  plus CPU RSS and GPU allocated/reserved peaks.
- profiles/layout_064/forward.json: corresponding 64-partition profile.
- summary.json: per-layout median/mean/stddev/range, pairwise
  differences, phase totals and per-layer timings, output parity.
- process.log in each trial directory, even if the subprocess fails.

If interrupted, resubmit the *same* job with unchanged settings.
Only complete trial reports are reused; an incomplete subprocess
directory must be inspected and cleaned manually. Do not submit two
jobs writing to the same destination concurrently. A running job
holds the .active.lock file and removes it when exiting normally.

## Numerical guarantees

The saved full outputs are compared elementwise (in memory-mapped,
bounded chunks), with atol=rtol=2e-4. Both outputs must have shape
[16,974,593,5], one checkpoint digest and one random seed. This
tests the invariance of the PARTITIONED solver with respect to
ownership decomposition, not an unavailable 257^3 full unpartitioned
M37 reference. Repeated outputs can differ slightly in floating
point reduction order. All original sparse graph attention and
DiT weights remain unchanged.

## Next decision after results

- Compare six-run medians and dispersion at 32 vs 64, not single
  minima; evaluate paired differences.
- Determine fraction of diagnostic time in CPU plan loading,
  CPU gathers, host-device staging, attention and MLP.
- If partition assembly dominates, consider bounded metadata
  caching or partition descriptors that avoid repeated CPU CSR
  assembly while retaining low memory.
- If transfers dominate, consider pinned memory/prefetch and
  benchmarking asynchronous copies *without changing mathematical
  attention semantics*.
- If sparse attention dominates, investigate
  edge-wise kernel efficiency and CSR staging, not physics quality.
- Only after establishing these bottlenecks, design a graph-size
  scaling experiment with comparable topology and field conventions.

This document describes a new **unexecuted** benchmark until a Grace
job and results are provided.
