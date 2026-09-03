# M7 Baseline Benchmarks

## Status

M7 introduces the reproducible performance-evidence protocol and the first benchmark executable for the framework/null-model baseline.

M6 correctness is considered target-validated on Calypso from the reported gate:

- 122 unit tests passed;
- Ruff checks passed after the formatter-only follow-up;
- two-rank CPU/Gloo DDP validation matched the single-process global reference with rank sample counts 3 and 2 and unequal microbatch counts.

M7 is **not complete** until the benchmark executable itself has passed the software gate and target single-GPU measurements have been collected on Calypso.

## 1. Goal

M7 establishes a measurement baseline before introducing sparse graph attention in M8.

The purpose is not to argue that the M5 `NodeLinearBaseline` is computationally representative of a graph transformer. It is deliberately too simple for that. Instead M7 establishes:

- a stable timing/memory methodology;
- persisted runtime/hardware/workload provenance;
- host packed-batch/task-preparation cost;
- current M6 training-path overhead;
- the real HIT mesh-size benchmark workload;
- a reproducible reference against which later graph models can be measured.

All graph-kernel performance claims remain `ANALYTICAL` until a graph-aware model exists and is measured.

## 2. Measured regions

The M7 executable measures three regions independently.

### Host packing and task preparation

```text
materialized Sample objects
    -> pack_samples
    -> NodeRegressionTask.prepare
```

Storage I/O is excluded.

### Forward

```text
frozen standardized task batch
    -> NodeLinearBaseline.forward
```

The forward benchmark uses `torch.inference_mode()`.

### Training iteration

```text
prepared task batch
    -> frozen M6 statistical standardization
    -> model forward
    -> per-sample MSE
    -> backward
    -> AdamW.step
```

The optimizer uses `lr=0` and `weight_decay=0` during timing. AdamW state is initialized during warmup and therefore exists before measured CUDA peak-memory statistics are reset.

## 3. Timing convention

The default benchmark uses:

```text
warmup:      10 calls
measurements: 50 calls
primary statistic: median latency
```

CUDA is synchronized before and after every measured call.

For each region M7 records:

- median latency;
- mean latency;
- population standard deviation;
- minimum and maximum latency;
- graphs/second;
- nodes/second;
- CUDA allocated/reserved memory before measurement;
- peak allocated/reserved memory;
- incremental peak allocated memory.

The null baseline does not consume edges, so M7 deliberately does not report an edge-throughput number for its forward/training measurements.

## 4. Synthetic reference workloads

The synthetic dataset is non-physical. Its role is controlled systems benchmarking.

A useful first single-GPU sweep is:

| Workload | Graphs | Nodes/graph | Total nodes | Purpose |
|---|---:|---:|---:|---|
| S1 | 1 | 8192 | 8192 | small single graph |
| S2 | 1 | 32768 | 32768 | large single graph |
| S3 | 4 | 8192 | 32768 | same approximate node load, multiple graphs |
| S4 | 8 | 4096 | 32768 | higher graph-count reduction/packing overhead |

This sweep is useful for separating total-node cost from graph-count overhead in the current per-sample loss implementation.

It is **not** an edge-scaling study because the affine baseline ignores topology.

Example:

```bash
python scripts/benchmark_m7.py \
  --device cuda \
  --dtype float32 \
  --warmup 10 \
  --repetitions 50 \
  --evidence TARGET_VALIDATED \
  --output ../GraphAttention_benchmarks/m7_s3.json \
  synthetic \
  --graphs 4 \
  --nodes-per-graph 8192 \
  --spatial-dim 3
```

## 5. Real HIT workload

M7 also supports the real `HIT_LES_FORCED` AVBP snapshot already used for M3.3 validation.

The benchmark:

1. loads the explicitly paired snapshot and mesh;
2. applies the explicit one-based connectivity interpretation for this file;
3. converts native hex connectivity into the current sparse directed edge list;
4. uses the five conservative fields `rho`, `rhou`, `rhov`, `rhow`, `rhoE`;
5. applies M3.3 physical nondimensionalization;
6. fits M6 sample-balanced statistics on the benchmark sample solely to exercise the frozen training path;
7. benchmarks a five-channel-to-five-channel affine null model.

The mapping of the five state channels to themselves is a **benchmark workload only**. It is not proposed as a useful CFD learning task and no predictive-accuracy claim follows from it.

Example on Calypso:

```bash
python scripts/benchmark_m7.py \
  --device cuda \
  --dtype float32 \
  --warmup 10 \
  --repetitions 50 \
  --evidence TARGET_VALIDATED \
  --output ../GraphAttention_benchmarks/m7_hit.json \
  avbp \
  --snapshot /gpfs-calypso/scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT/solut_hit_00000770.h5 \
  --mesh /gpfs-calypso/scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5 \
  --case-file cases/HIT_LES_FORCED.yaml \
  --case-id HIT_LES_FORCED \
  --mesh-id HIT_LES_FORCED \
  --connectivity-indexing one
```

The HDF5 read and one-time edge construction occur before repeated timings and are explicitly excluded from latency claims.

## 6. Calypso execution model

The current target workflow has one GPU per Calypso node. M7 therefore establishes **single-GPU** target evidence only.

A Slurm allocation needs one GPU. For example, inside an allocated GPU shell:

```bash
nvidia-smi
python scripts/benchmark_m7.py ...
```

The benchmark records available Slurm variables and `CUDA_VISIBLE_DEVICES` in its JSON output.

Multi-node NCCL scaling remains a separate future benchmark. The M6 two-process CPU/Gloo test validates distributed weighting correctness, not GPU communication performance.

## 7. Benchmark result artifact

Each invocation writes one JSON file and prints the same content to stdout.

Keep benchmark output outside the git working tree when possible so repository provenance remains clean. The result includes:

- exact git state;
- Python/PyTorch/CUDA versions;
- GPU identity and memory;
- relevant Slurm environment;
- workload shape and semantic channels;
- model/optimizer identity;
- timing/memory results;
- evidence label.

The evidence label is supplied explicitly by the person running the benchmark. The executable cannot prove that a machine is the designated target environment merely from its hostname.

## 8. Assumptions

M7 introduces or inherits these assumptions:

- the selected workload fits in device memory;
- benchmark jobs are not intentionally sharing the selected GPU with unrelated work;
- samples are materialized before host packing is timed;
- host-to-device transfer is outside the current repeated benchmark regions;
- float32 is the initial target benchmark precision;
- model-facing task tensors, coordinates, sparse index tensors, and node weights are moved to the selected device, while redundant raw source fields remain host-resident;
- the real HIT benchmark uses the established one-based raw connectivity interpretation;
- the one-sample real-data scaler is a performance fixture, not a scientifically reusable training scaler.

## 9. Handled edge cases

The benchmark utilities explicitly handle:

- CPU execution without CUDA memory reporting;
- CUDA unavailability when a CUDA benchmark is requested;
- invalid warmup/repetition counts;
- synthetic graph/node-count validation;
- output-directory creation;
- optional Slurm metadata;
- either float32 or float64 baseline execution;
- real AVBP connectivity construction before packing.

## 10. Deferred or unsupported cases

M7 does not yet establish:

- graph-attention or message-passing kernel performance;
- independent edge-density scaling;
- host-to-device transfer throughput;
- HDF5 storage throughput;
- asynchronous data loading/prefetching;
- BF16/FP16 benchmark evidence;
- multi-GPU or multi-node performance;
- CUDA graph capture or compiled execution;
- full Lightning trainer throughput;
- energy/power measurements.

These should be added only when a concrete implementation or performance question requires them.

## 11. Failure behavior

The benchmark fails rather than silently adapting when:

- CUDA is requested but unavailable;
- workload counts are invalid;
- the AVBP case/mesh/snapshot contract fails;
- physical nondimensionalization requirements are not satisfied;
- a standardizer cannot be fitted;
- the model/task dtype contract is violated;
- device memory is insufficient.

An out-of-memory failure is not converted into a benchmark result.

## 12. Efficiency and numerical trade-offs

Explicit synchronization makes latency measurement more reliable but prevents overlap across measured calls. This is intentional for the M7 reference protocol.

The current per-sample M6 loss and finite-value validation use transparent Python/reduction logic and may be slower than future fused implementations. M7 measures that cost rather than optimizing it before evidence exists.

`NodeLinearBaseline` is intentionally compute-light. Consequently Python dispatch, standardization, loss reduction, and optimizer overhead can dominate its training latency. That result should not be extrapolated to a sparse transformer.

## 13. M7 completion gate

Before M7 is considered complete:

```bash
pytest
ruff check .
ruff format --check .
python scripts/inspect_config.py
```

must pass on Calypso, followed by at least:

- one synthetic single-GPU `TARGET_VALIDATED` run;
- one real HIT single-GPU `TARGET_VALIDATED` run.

The persisted JSON outputs should then be reviewed before any baseline performance statement is added to traceability.
