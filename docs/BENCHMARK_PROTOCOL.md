# Benchmark Protocol

## 1. Purpose

Performance is a first-class project requirement, but performance claims are valid only when the measured workload, hardware, software state, and timing method are explicit.

This document defines the repository-wide protocol used from M7 onward.

Correctness gates take precedence over performance. A faster implementation that changes the scientific or numerical contract is not a valid optimization unless that change is independently specified and validated.

## 2. Evidence levels

Every performance statement must use one of these evidence labels.

### `ANALYTICAL`

The statement is based only on complexity, tensor sizes, memory estimates, algorithm structure, or another non-measured argument.

Examples:

- an edge-local attention formulation is `O(E)` in the number of attended pairs;
- a dense all-pairs score matrix requires `O(N^2)` storage.

`ANALYTICAL` is not a latency, throughput, or peak-memory claim.

### `LOCAL_BENCHMARK`

The quantity was measured with the repository benchmark protocol, but not on the designated target hardware/workload combination.

The exact machine, workload, dtype, software versions, git state, and benchmark parameters must accompany the result.

### `TARGET_VALIDATED`

The quantity was measured with the repository benchmark protocol on the designated target environment and the stated target workload.

`TARGET_VALIDATED` applies only to the exact class of workload and execution mode represented by the measurement. A single-GPU result does not establish multi-GPU scaling. A synthetic workload does not establish real-data I/O performance unless that I/O was part of the measurement.

## 3. Required benchmark metadata

Every persisted benchmark result must record at least:

- benchmark name/version or script identity;
- evidence label;
- timestamp;
- git SHA, branch, and dirty/clean state;
- Python version;
- PyTorch version;
- CUDA version and availability;
- GPU model when applicable;
- device count relevant to the benchmark;
- workload source/type;
- number of physical graphs/samples;
- total nodes;
- total edges when graph topology is present;
- ordered input/target channel semantics;
- dtype/precision mode;
- model identity and parameter count;
- optimizer identity for training benchmarks;
- warmup count;
- measured repetition count;
- scheduler/job metadata when available on a cluster.

Benchmark outputs should be machine-readable JSON so future implementations can be compared without manually transcribing results.

## 4. Timing rules

GPU work is asynchronous. CUDA timings therefore require synchronization before the timed region begins and after the timed region ends.

M7 uses wall-clock `perf_counter` timing with explicit `torch.cuda.synchronize()` around each measured call. This deliberately favors transparent, end-to-end Python-call latency over isolated CUDA-event kernel timing.

Each benchmark consists of:

1. setup outside the timed region;
2. warmup calls;
3. synchronization;
4. repeated synchronized measurements;
5. summary statistics.

The primary latency statistic is the **median** measured latency. Mean, population standard deviation, minimum, and maximum are also recorded for diagnostic purposes.

Warmup iterations are not included in timing statistics.

## 5. What must be timed separately

Do not combine unrelated costs into one number without naming the scope.

At minimum distinguish where relevant:

- storage/HDF5 reading;
- mesh/topology construction;
- host-side packing/task preparation;
- host-to-device transfer;
- model forward;
- forward + loss + backward + optimizer step;
- distributed communication.

The M7 baseline benchmark intentionally excludes storage I/O and one-time mesh-edge construction from repeated timings. It separately measures host packing/task preparation and GPU/CPU model execution.

## 6. Throughput

Throughput must state its denominator.

For variable graphs, report at least one physically interpretable count such as:

- physical samples/second;
- nodes/second;
- edges/second when the measured algorithm actually consumes edges.

Do not report edge throughput for a model that ignores connectivity. Do not compare samples/second between workloads with materially different mesh sizes without also reporting node/edge counts.

For M7, nodes/second and graphs/second are derived from median latency.

## 7. CUDA memory

When CUDA is used, record allocator state after warmup and peak allocator state during measured calls:

- allocated bytes before measurement;
- reserved bytes before measurement;
- peak allocated bytes;
- peak reserved bytes;
- incremental peak allocated bytes above the resident pre-measurement allocation.

The benchmark must not call `torch.cuda.empty_cache()` between repetitions because doing so changes allocator behavior and does not represent normal training execution.

PyTorch allocator measurements are not equivalent to total process or device memory reported by system tools. Claims must state which quantity is being used.

## 8. Benchmark workload classes

### 8.1 Synthetic framework workload

The deterministic `SyntheticMeshDataset` is permitted for framework scaling and regression benchmarking. It is explicitly non-physical.

The M7 default synthetic workload uses fixed per-graph node counts so the requested scale is unambiguous. The generated chain/cycle/star topologies are retained only as structural inputs; the M5 null baseline does not consume them.

### 8.2 Real AVBP workload

The real AVBP benchmark uses an explicitly declared snapshot, mesh, case definition, field set, and connectivity indexing convention.

For M7 the benchmark task is deliberately a **performance workload**, not a scientific prediction claim: the five conservative state channels are mapped to the same five channels with the null affine model after M3.3 physical nondimensionalization and M6 train-only-style statistical standardization.

This workload establishes framework/runtime overhead at the real mesh size. It does not establish CFD predictive accuracy.

## 9. Comparison rules

A performance comparison is valid only when the compared runs have compatible:

- hardware;
- software environment;
- dtype/precision;
- workload graph/sample composition;
- channel counts;
- timing scope;
- warmup/repetition procedure.

If one of those differs, present the measurements separately and do not attribute the difference solely to the implementation under study.

For architecture comparisons in M8 and later, use the same persisted M7-style workload definition and benchmark procedure whenever possible.

## 10. Single-GPU and distributed evidence

Single-GPU latency/throughput/memory and distributed scaling are separate claims.

Calypso currently provides one GPU per node for the user's target workflow. M7 therefore establishes a single-device baseline. Multi-node NCCL scaling requires a separate benchmark and is not inferred from the M6 CPU/Gloo DDP correctness validation.

## 11. M7 reference executable

The baseline executable is:

```bash
python scripts/benchmark_m7.py ...
```

It measures:

- host `pack_and_prepare` latency from already materialized samples;
- null-baseline forward latency;
- M6 reference training-iteration latency;
- graph/node throughput;
- CUDA allocator peaks when applicable;
- complete runtime and scheduler provenance.

The training benchmark uses AdamW with zero learning rate and zero weight decay. This exercises optimizer-state and step mechanics without allowing repeated benchmark iterations to drift the model parameters.

## 12. Assumptions and limitations

The M7 protocol assumes:

- benchmark inputs fit on the selected device;
- the device is not intentionally shared with unrelated workloads during measurement;
- system load, power state, and GPU clock behavior may still introduce run-to-run variability;
- synchronized Python wall-clock latency includes framework dispatch overhead;
- PyTorch CUDA allocator counters do not capture all driver/library memory;
- the null baseline does not measure sparse graph-kernel cost because it does not consume coordinates or edges.

For important model comparisons, repeat the benchmark as independent runs and retain all result JSON rather than relying on one favorable run.
