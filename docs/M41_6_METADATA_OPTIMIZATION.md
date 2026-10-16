# M41.6 — Selective and mmap static graph metadata scalability

## Scope and motivation

**Only computational scalability.** Preserve trained M37
AlternatingDilatedGeometricDiT weights, exact one-hop/exact-two-hop
incoming graph attention, local geometric displacement, ten-layer
Jacobi synchronization, Flow Matching input convention, and existing
257^3 HDF5-derived unstructured-capable CSR.

M41.5 Grace diagnostics measured ~87.22 seconds (32 partitions)
and ~91.96 seconds (64 partitions) inside the
partition_metadata_load stage per forward, versus ~6.89/6.56
seconds in sparse geometric attention. This stage includes NPZ
reading, int32->int64 conversion, CPU rectangular CSR construction
and source-minus-target geometric displacements.

The existing lazy loader rebuilds BOTH one-hop and exact-two-hop
CSR/geometry even when the current alternating layer requires only
one family. It also builds both for the input and output projections,
which only need owned-node IDs.

## Exactly three experimental implementations

1. **baseline**: unchanged semantics of M41.4/M41.5 lazy NPZ loading,
   except CPU-only profiler subtimings are now collected when requested.
   Both families are reconstructed for every visit, including
   initial/final. It remains the regression comparison.
2. **selective**: new SelectiveDiskPartitionCollection reads only
   owned IDs during initial/final, and only the active family
   ('local' or 'dilated') during transformer layers. It rebuilds
   only that family's original exact rectangular CSR and raw-coordinate
   source-minus-target displacement. Owned/local/dilated stage objects
   are not retained between partitions or layers.
3. **mmap**: an OPTIONAL one-time per-layout preprocessing pass
   produces immutable .npy arrays for owned/read/owned_local indices,
   both precomputed rectangular CSR row pointer/row/column and
   the two exact displacement matrices, **one partition at a time**.
   Then MappedDiskPartitionCollection opens only the relevant arrays
   with numpy mmap_mode='c' (copy-on-write) on demand. This does NOT
   preload the whole graph or its CSR into CPU RAM.
   It does consume additional scratch storage (potentially tens
   of GiB across the 32- and 64-partition layouts), in exchange for
   avoiding CPU CSR/displacement reconstruction.

Both optimized paths exploit a proven invariant of the saved
M41.4 partition files: their edges are already sorted by owned
target then source, so the identity CSR edge permutation is
unnecessary. The sparse context now checks the authoritative
CSR column count for empty neighborhoods; its edge_index itself
may be omitted for optimized disk providers. All ordinary M37
original/full and M41.4 model paths retain the same mathematics.

The mmap cache checks the original partition manifest SHA-256
and node/part counts on loading. Compilation refuses to overwrite
any previous directory, completes the manifest atomically and
preserves original graph files. If compilation fails, manually
inspect incomplete cache directories before restarting.

## More granular metadata instrumentation

The optional synchronized StreamedStageProfiler still measures
the partition_metadata_load parent stage (with CPU and GPU
synchronization on either side). Nested CPU metadata detail timers
are **separate, not additive with the parent stage totals**:

- metadata_npz_decode: individual .npz member extraction/array load
  for active pass (legacy or selective);
- metadata_index_conversion: int64 conversions and owned-local
  index lookup, where applicable;
- metadata_csr_assembly: exact target-major CSR build and validations;
- metadata_relative_displacement: coordinate gather and source-target;
- metadata_mmap_array_open: .npy memory-map opening (mmap path);
- metadata_mmap_csr_wrapper: CSR sparse tensor wrapping/validation.

A **major instrumentation caveat**: memory-mapped .npy file pages
may be faulted in only when later operations use the arrays,
including when staging GPU topology. mmap open time alone is
therefore not equivalent to total I/O time. Profiling also inserts
many synchronization calls and perturbs wall time. Use UNPROFILED
forward benchmarks to claim speed improvements.

## Acceptance sequence

1. Small randomized/shuffled irregular 3D hexahedral numerical
   parity checks for baseline/selective/mmap, including
   self-attention policy variants and under inference_mode().
   Verify exact tensors at atol=rtol=2e-4, stable model weights,
   ordinary versioned CSR tensors, cache freshness/identity.
2. Compile mmap-ready, per-partition metadata for the EXISTING
   32 and 64 layouts; no full graph rebuild and no modification
   to source HDF5/CSR/partition metadata.
3. One matched Grace allocation with three clean-process forward
   runs of each of six configurations (2 partitions × 3 modes).
   Rotate A/B/C order and reverse the layout order on alternating
   repetitions. Record median, mean, stddev, process RSS peak,
   PyTorch GPU allocation/reservation and partition plan open time.
4. Run one SEPARATE synchronized diagnostic forward per config
   (6 additional trials) to compare source of metadata time.
5. Compare **full saved first-trial outputs**, each
   [16,974,593,5] float32, against baseline 64-partition output
   elementwise in bounded 65,536-row chunks.
   Zero violations at atol=rtol=2e-4 required.

Expected total: 18 ordinary + 6 profiled full forwards.
No new training, Heun generation, spectra or numerical fidelity
claims are part of this benchmark.

## Cluster commands

Prepared graph remains:
 /scratch/coop/theret/GraphAttention_runs/m41_4_257_generic_graph

Prepared 32 layout remains:
 /scratch/coop/theret/GraphAttention_runs/m41_4_257_scaling/layouts/layout_032

Grace Stage 1: small regression gates and compile optional mmap cache:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_6_257_metadata_prepare.slurm
~~~

Output cache:
 /scratch/coop/theret/GraphAttention_runs/m41_6_257_metadata_cache/
     layout_032/cache.json
     layout_064/cache.json

Grace Stage 2: new controlled benchmark after Stage 1 has passed:

~~~bash
sbatch scripts/slurm/m41_6_257_metadata_bench.slurm
~~~

Output:
 /scratch/coop/theret/GraphAttention_runs/m41_6_257_metadata_benchmark/
     progress.json
     summary.json
     timings/layout_NNN/{baseline,selective,mmap}/repeat_XX/forward.json
     profiles/layout_NNN/{baseline,selective,mmap}/forward.json

Each first unprofiled trial also saves a full 5-channel forward.npy
for cross-layout/mode comparison. New output directory guards and
an exclusive lock prevent trial mixing and concurrent writes.
Completed process reports are reusable after a Slurm interruption.
A partial process directory must be inspected manually.

Both scripts use --partition=grace, --gres=gpu:1,
--cpus-per-task=4, --time=04:00:00 and **no --mem directive**.
Check available disk quota and scratch free space before caching.
Two layout caches can require tens of GiB of extra scratch storage.
All 24 forwards might exceed 4 hours in adverse conditions, but
the run records progress after each completed trial and is resumable.
Do not launch another job against an active output directory.

## What will be concluded

- Whether selective materialization alone eliminates a large
  proportion of 87–92 s metadata cost at 257^3.
- Whether precomputed memory-mappable CSR and displacement
  provide further acceleration without moving the bottleneck
  into memory page faults or CPU peak RSS.
- Trade-off between 32 (faster, bigger partitions) and
  64 (smaller GPU memory) under each static metadata strategy.
- Whether every optimized path preserves M37 exact output
  up to previously validated floating-point tolerance.

**Status (2026-10-16): M41.6 complete full-resolution benchmark received and analyzed.** Results follow.


## M41.6 GH200 outcome — 2026-10-16

Evidence submitted: summary(20261016-080737).json and progress.json.
The summary reports **status=complete**, 18 unprofiled (three repeats per
mode/layout) and six separately profiled forwards (24 total), all on
the SAME fixed 16,974,593-node physical graph, M37 checkpoint
SHA-256 01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8
and seed=5678.

### Controlled unprofiled forward timings and memory

Median statistics below are over **three trials each**. No profiled
diagnostic wall times are used in this latency comparison.

| Partitions | Metadata | Median forward (s) | Mean (s) | Std dev (s) | Range (s) | Peak process CPU RSS (GiB) | Peak PyTorch GPU allocated (GiB) |
|---:|---|---:|---:|---:|---|---:|---:|
| 32 | baseline | 112.6702 | 112.8523 | 1.3961 | 111.5563-114.3305 | 21.0725 | 4.67175 |
| 32 | selective | **60.6660** | 61.8852 | 3.2025 | 59.4714-65.5182 | 20.0785 | 4.67175 |
| 32 | mmap | 61.7896 | 60.9036 | **13.1249** | 47.3582-73.5630 | **19.3442** | 4.67175 |
| 64 | baseline | 119.7291 | 118.9934 | 1.6185 | 117.1378-120.1134 | 19.8607 | 2.38414 |
| 64 | selective | 63.9071 | 64.4568 | 1.7861 | 63.0103-66.4532 | 19.2952 | 2.38414 |
| 64 | mmap | **59.6087** | 67.0968 | **16.5819** | 55.5796-86.1021 | **18.8331** | 2.38414 |

Relative to same-layout baseline median: selective is **46.16%**
faster (1.86x) at 32 and **46.62%** faster (1.87x) at 64.
Mmap median reductions are 45.16% (32) and 50.21% (64), but
its **run-to-run timing variability is much larger**; mmap's
mean at 64 (67.10 s) is slower than selective mean (64.46 s).
It is NOT yet established that mmap is consistently faster.
All three metadata modes within one partition count have
identical reported GPU allocated and GPU reserved peaks
(18.8223 GiB reserved at 32; 7.6797 GiB at 64).
Selective decreases CPU RSS by 0.994 GiB (32) / 0.565 GiB (64).
Mmap decreases CPU RSS by 1.728 GiB (32) / 1.028 GiB (64).

### What the synchronized stage instrumentation measured

| Synchronized stage (seconds) | 32 baseline | 32 selective | 32 mmap | 64 baseline | 64 selective | 64 mmap |
|---|---:|---:|---:|---:|---:|---:|
| Full profiled forward | 114.787 | 62.830 | 49.898 | 119.494 | 67.023 | 51.248 |
| Partition metadata load | **89.798** | 39.126 | 8.142 | **91.370** | 40.014 | 11.533 |
| Sparse geometric attention | 6.882 | 6.834 | 6.865 | 6.573 | 6.532 | 6.540 |
| H2D CSR / geometry stage | 0.379 | 0.324 | **11.301** | 0.441 | 0.394 | **11.411** |
| D2H owned-state readback | 3.882 | 3.801 | **9.937** | 5.661 | 5.857 | **8.048** |

These are synchronized *wall times*, NOT GPU kernel durations.
Profiler stages are disjoint but additional nested CPU substep
timers belong within metadata and MUST NOT be added again.
The large mmap H2D and D2H stage penalties show that reduced
explicit metadata preparation **does not remove all costs**.
Potential cause: mmap's deferred file-page faults when CUDA
staging or CPU consumption touches arrays. This is a hypothesis;
page-fault counters and per-stage I/O are not in the report.
The 32-partition mmap H2D geometry staging is 11.30 s
versus 0.32 s for selective, and D2H readback is 9.94 s
versus 3.80 s for selective. No change in geometric sparse
attention time is apparent.

Nested CPU metadata substeps, in seconds:

| CPU substep | 32 baseline | 32 selective | 64 baseline | 64 selective |
|---|---:|---:|---:|---:|
| CSR construction | 49.043 | 19.384 | 50.799 | 19.740 |
| Relative displacement | 16.998 | 7.127 | 16.445 | 6.863 |
| NPZ decoding | 16.516 | 7.896 | 17.510 | 8.868 |
| Index conversion | 6.205 | 4.199 | 5.672 | 4.051 |

Mmap nested subtimings cover *mapping opening* and wrapping CSR
but not physical page residency: for 32, mmap array opening
7.836 s and CSR wrapper 0.236 s; for 64, opening 11.131 s
and wrapper 0.291 s. Therefore simple "mmap metadata time"
is not a complete measure of all metadata-related I/O.

### Numerical integrity

All six saved first-trial outputs were compared element-by-element
against 64-partition baseline with shape [16,974,593, 5],
84,872,965 float32 values, atol=rtol=2e-4.
**Zero tolerance violations in all comparisons**.
The maximum cross-configuration absolute difference was
3.337860107421875e-6 and all non-reference relative L2
differences were approximately 2.91e-7. This demonstrates
parity **between the tested partitioned implementations**,
not an unpartitioned M37 257^3 gold standard, bitwise
determinism across all repeats, or physical model validity.
The uploaded files do not include Stage 1 unit-test logs;
do not assert an exact pytest pass count from these JSONs.

### IMPORTANT: host/provenance discrepancy

The uploaded progress.json records its initial host as
calypso-grace02.calypso.cerfacs.fr, but the final summary.json
reports host calypso-grace01.calypso.cerfacs.fr.
The benchmark code preserves the initially recorded progress
host when resuming but writes the currently executing host in
the final summary. This is consistent with a run resumed on
a different allocation/host. **Individual trial host IDs
are not recorded**, so the eighteen measurements should NOT
be characterized as all collected under one fixed Grace
allocation, despite the original design objective.
This matters especially for mmap, which depends on host
page cache and file-read locality.

### Decision and next computational experiment

- Prefer **32 selective** for reproducible low forward latency
  (median 60.67 s, peak CPU 20.08 GiB, GPU allocated 4.67 GiB).
- Prefer **64 selective** when GPU headroom matters (median
  63.91 s, CPU 19.30 GiB, GPU allocated 2.38 GiB).
- Keep **mmap experimental**: lower CPU RSS and occasional
  faster runs do not outweigh observed extreme timing
  dispersion without per-host and page-cache control.
- Next narrow A/B test: run selective vs mmap at 32/64
  on one pinned allocation if feasible, record per-trial
  hostname and Slurm job ID, process minor/major page faults,
  /proc/self/io read_bytes and rchar, cold versus warmed page
  cache conditions, and H2D/D2H stage variation.
  Do not change numerical semantics, M42 code, or the global
  one-hop CSR. Avoid dropping shared system caches on HPC.
- A further speedup avenue is reducing CPU CSR reconstruction
  in selective mode without incurring mmap's page-fault
  variability. Any caching policy must bound CPU RSS.

**Result: M41.6 achieved about 1.9x forward throughput in
selective mode while retaining previously established 257^3
partitioned numerical tolerance.**
