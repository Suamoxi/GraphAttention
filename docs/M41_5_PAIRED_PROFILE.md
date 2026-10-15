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

The benchmark has now completed on Grace. See the measured results below.


## Grace M41.5 results — October 15, 2026

Source: submitted summary(20261015-125017).json (status=complete)
from calypso-grace02. Exactly **12 unprofiled forwards** (six
per layout) plus **two synchronized diagnostic forwards**
completed using the identical 16,974,593-node graph, fixed
checkpoint SHA-256
01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8
and seed=5678.

### Unprofiled matched-host latency and memory

| Metric | 32 lazy (n=6) | 64 lazy (n=6) |
|---|---:|---:|
| Median forward (s) | **108.92652** | 113.94136 |
| Mean forward (s) | **109.23614** | 114.52810 |
| Stddev forward (s) | 1.28484 | 1.75318 |
| Forward min–max (s) | 108.10068–111.62242 | 112.75238–117.67505 |
| Peak process RSS median (GiB) | 20.96698 | **19.81638** |
| GPU peak allocated (GiB) | 5.00052 | **2.54939** |
| GPU peak reserved (GiB) | 18.82227 | **7.67969** |
| Partition plan opening median (s) | 0.15470 | 0.16271 |

All SIX matched pairs favored 32 partitions: 32 was faster by
6.053, 4.210, 6.698, 5.219, 6.077 and 3.494 seconds.
The mean matched-pair difference is **5.292 s** and the
median speed improvement is **4.40%** relative to the
64-partition median. This is stronger confirmation of the
speed advantage than the prior two-run sweep. The 64-plan
retains lower CPU and GPU peak memory.

Note: M41.5 measured a higher **GPU allocated peak**
than M41.4 for BOTH layouts (32: 5.00052 vs 4.41387 GiB;
64: 2.54939 vs 2.25521 GiB). Code review identified a
likely tensor-lifetime regression in the optional profiling
refactor: it kept the staged GPU CSR topology referenced
after attention, through the residual and MLP. The original
executor passed that CSR as a temporary argument and released
it immediately after attention. A follow-up code patch
explicitly deletes the staged topology after its attention
call, restoring the intended lifetime. **GPU memory recovery
is a hypothesis pending a new Grace validation**; do not
reinterpret this result's actual M41.5 peak measurements.

### Detailed synchronized diagnostics (not the unprofiled performance timings)

| Stage | 32 lazy: seconds (% of profiled wall) | 64 lazy: seconds (% of profiled wall) |
|---|---:|---:|
| Full profiled forward | 111.1940 (100%) | 117.2821 (100%) |
| **Partition metadata load/CPU assembly** | **87.2208 (78.57%)** | **91.9612 (78.53%)** |
| Sparse geometric attention context | 6.8895 (6.20%) | 6.5630 (5.60%) |
| Owned GPU-to-CPU readback | 3.7591 (3.39%) | 4.5550 (3.89%) |
| CPU hidden-state gather | 3.1229 (2.81%) | 3.2773 (2.80%) |
| CPU owned-state scatter | 2.6936 (2.43%) | 2.7143 (2.32%) |
| GPU MLP | 1.7352 (1.56%) | 1.7711 (1.51%) |
| GPU QKV and normalization | 0.9128 (0.82%) | 0.9888 (0.84%) |
| H2D hidden and owned indices | 0.4649 (0.42%) | 0.6103 (0.52%) |
| H2D rectangular topology/geometric displacement | 0.3798 (0.34%) | 0.4473 (0.38%) |

The timings include explicit CUDA synchronization and do
NOT isolate filesystem I/O from CPU parsing, CSR creation
or relative geometry reconstruction. Fractions are of
**synchronized, profiled wall time**, NOT measured fractions
of the unprofiled benchmark.

Of the attention-context stage, local one-hop attention
used 2.2365 s (32) / 2.0436 s (64); exact-two-hop attention
used 4.6530 s (32) / 4.5194 s (64). The two-hop operations
are costlier, but this is NOT the principal bottleneck.

**Fundamental source-level observation:** The lazy
DiskPartitionCollection.__getitem__ reads NPZ arrays,
converts all edge index arrays, assembles BOTH one-hop and
exact-two-hop rectangular CSR objects, and recomputes BOTH
relative-displacement tensors on EVERY partition iteration.
There are 12 traversals of the graph per model forward:
initial projection, ten alternating attention layers, and
final projection. Each attention layer actually uses only
ONE of the two attention types; the initial and final
projection layers use NEITHER attention type.
The 32- and 64-layout traversals therefore materialize
384 and 768 partition objects respectively, even though
their profiler reports count 396 and 780 stage calls
(the difference of 12 is iterator-exhaustion overhead).

**This is the next main optimization target**, not GPU
sparse attention. A defensible next M41.6 task is to profile
NPZ decompression/deserialization, CPU index conversion,
CSR construction, and geometric displacement creation
separately, then load only the edge family needed in each
transformer layer. Initial/final projections need node
ownership descriptors but not attention CSR. Keep global
one-hop CSR immutable and maintain exact M37 output parity.
Avoid fully preloading every CSR into RAM, which earlier
M41.4 runs showed causes much larger CPU RSS and latency.

### Numerical check

The saved first 32- and 64-partition outputs were compared
elementwise over 84,872,965 values:
max_abs=3.337860107421875e-6,
RMS=2.9438352099780447e-7,
relative_L2=2.911468036662084e-7,
zero violations at atol=rtol=2e-4. No same-size
unpartitioned M37 reference exists, and the other
ten unprofiled outputs were not saved for full comparison.

**Status:** the M41.5 benchmark and diagnostic profile
are COMPLETE; the follow-up staged-topology lifetime
adjustment is COMMITTED but has not yet been retested
on the GH200.
