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

**Status: source changes and Grace jobs prepared but the new
M41.6 benchmark has NOT yet run.** Baseline performance numbers
are from previous M41.5 results, not the new optimized code.
