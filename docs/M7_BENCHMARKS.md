# M7 Baseline Benchmarks

## Status

M7 introduces the reproducible performance-evidence protocol and the first benchmark executable for the framework/null-model baseline.

M6 correctness is considered target-validated on Calypso from the reported gate:

- 122 unit tests passed;
- Ruff checks passed after the formatter-only follow-up;
- two-rank CPU/Gloo DDP validation matched the single-process global reference with rank sample counts 3 and 2 and unequal microbatch counts.

M7 is **complete and target-validated** for its frozen single-GPU framework/null-baseline scope. On 2026-09-04 the software gate passed and both required `TARGET_VALIDATED` workloads were measured in Slurm job `400132` on one NVIDIA GH200 480GB GPU on Calypso. The exact reference measurements are recorded in §13.

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
warmup:       10 calls
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

The current hex-derived edge list does **not** add the deferred periodic cross-boundary graph connections. Its reported edge count therefore describes the present M3.2 connectivity representation, not a claim of complete periodic physical topology. This does not affect M7 null-model forward/training timing because the baseline does not consume edges.

The mapping of the five state channels to themselves is a **benchmark workload only**. It is not proposed as a useful CFD learning task and no predictive-accuracy claim follows from it.

Example on Calypso outside a container:

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

### 6.1 Singularity path visibility

The established Calypso job workflow launches Python through a Singularity image with bindings such as:

```bash
singularity exec --nv -B /home -B /scratch IMAGE PYTHON ...
```

Paths supplied to the benchmark must therefore be valid **inside the container**, not only on the login-node host filesystem. On the current setup, using the host-style `/gpfs-calypso/scratch/...` paths from inside that container can fail with `FileNotFoundError` because `/gpfs-calypso` is not bound into the image.

When `/scratch` is bound, use the container-visible `/scratch/...` form for the repository, AVBP files, case file, and benchmark output. Alternatively, bind `/gpfs-calypso` explicitly, but the project baseline is to reuse the established `/scratch` binding and avoid introducing a second path convention inside the job.

For example, the real HIT arguments inside the established container workflow are:

```text
snapshot:  /scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT/solut_hit_00000770.h5
mesh:      /scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5
case-file: /scratch/coop/theret/GraphAttention/cases/HIT_LES_FORCED.yaml
```

A representative one-GPU Slurm wrapper is:

```bash
#!/bin/bash
#SBATCH --job-name=m7_benchmark
#SBATCH --time=00:30:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --output=/scratch/coop/theret/GraphAttention_benchmarks/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_benchmarks/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
RESULTS=/scratch/coop/theret/GraphAttention_benchmarks
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/<python-environment>/bin/python3

mkdir -p "$RESULTS"
cd "$REPO"

singularity exec --nv -B /home -B /scratch \
  "$IMAGE" \
  "$PYTHON" -u "$REPO/scripts/benchmark_m7.py" \
  --device cuda \
  --dtype float32 \
  --warmup 10 \
  --repetitions 50 \
  --evidence TARGET_VALIDATED \
  --output "$RESULTS/m7_hit_${SLURM_JOB_ID}.json" \
  avbp \
  --snapshot /scratch/coop/theret/HIT_LES_FORCED/RUN/SOLUT/solut_hit_00000770.h5 \
  --mesh /scratch/coop/theret/HIT_LES_FORCED/MESH/mesh.mesh.h5 \
  --case-file "$REPO/cases/HIT_LES_FORCED.yaml" \
  --case-id HIT_LES_FORCED \
  --mesh-id HIT_LES_FORCED \
  --connectivity-indexing one
```

`PYTHON` must point to the same validated Python environment used for the repository checks. It is left explicit rather than silently falling back to the container system Python.

The benchmark records available Slurm variables and `CUDA_VISIBLE_DEVICES` in its JSON output.

For `--evidence TARGET_VALIDATED`, the executable additionally requires:

- a CUDA device;
- a clean git working tree.

Those checks prevent obvious evidence-label mistakes. The executable still cannot prove that the current cluster is the designated target environment merely from a hostname; that classification remains an operator responsibility.

Multi-node NCCL scaling remains a separate future benchmark. The M6 two-process CPU/Gloo test validates distributed weighting correctness, not GPU communication performance.

## 7. Benchmark result artifact

Each invocation writes one JSON file and prints the same content to stdout.

Keep benchmark output outside the git working tree so repository provenance remains clean. The result includes:

- exact git state;
- Python/PyTorch/CUDA versions;
- GPU identity and memory;
- relevant Slurm environment;
- workload shape and semantic channels;
- model/optimizer identity;
- timing/memory results;
- evidence label.

The evidence label is supplied explicitly by the person running the benchmark.

## 8. Assumptions

M7 introduces or inherits these assumptions:

- the selected workload fits in device memory;
- benchmark jobs are not intentionally sharing the selected GPU with unrelated work;
- samples are materialized before host packing is timed;
- host-to-device transfer is outside the current repeated benchmark regions;
- float32 is the initial target benchmark precision;
- model-facing task tensors, coordinates, sparse index tensors, and node weights are moved to the selected device, while redundant raw source fields remain host-resident;
- the real HIT benchmark uses the established one-based raw connectivity interpretation;
- periodic cross-boundary graph edges remain absent from the present real-HIT edge count;
- Singularity paths must resolve in the container namespace; the established Calypso wrapper binds `/scratch` and uses `/scratch/...` paths inside the image;
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
- real AVBP connectivity construction before packing;
- rejection of `TARGET_VALIDATED` on CPU or a dirty/unknown git worktree.

Container filesystem visibility itself is not auto-detected by the benchmark. Missing binds therefore surface as normal file-not-found errors before a result is produced.

## 10. Deferred or unsupported cases

M7 does not establish:

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
- `TARGET_VALIDATED` is requested on CPU;
- `TARGET_VALIDATED` is requested from a dirty or unverifiable git worktree;
- workload counts are invalid;
- the AVBP case/mesh/snapshot contract fails;
- a required host path is not visible inside the Singularity container;
- physical nondimensionalization requirements are not satisfied;
- a standardizer cannot be fitted;
- the model/task dtype contract is violated;
- device memory is insufficient.

An out-of-memory failure is not converted into a benchmark result.

## 12. Efficiency and numerical trade-offs

Explicit synchronization makes latency measurement more reliable but prevents overlap across measured calls. This is intentional for the M7 reference protocol.

The current per-sample M6 loss and finite-value validation use transparent Python/reduction logic and may be slower than future fused implementations. M7 measures that cost rather than optimizing it before evidence exists.

`NodeLinearBaseline` is intentionally compute-light. Consequently Python dispatch, standardization, loss reduction, and optimizer overhead can dominate its training latency. That result should not be extrapolated to a sparse transformer.

Using the existing `/scratch` container binding avoids extra mount configuration and matches the established Calypso workflow, at the cost of container-visible path strings differing from the `/gpfs-calypso/scratch/...` paths commonly shown on the login node.

## 13. Target-validated reference results

### 13.1 Environment and provenance

The completed M7 target run used:

```text
date:                 2026-09-04
Slurm job:            400132
node:                 calypso-grace03
GPU:                  NVIDIA GH200 480GB
compute capability:   9.0
PyTorch-visible VRAM: 102005473280 bytes
Python:               3.12.3
PyTorch:              2.7.0a0+7c8ec84dab.nv25.03
CUDA:                 12.8
dtype:                float32
git SHA:              79b156e27842618a54a0be18a81ea76c994ac140
git branch:           main
git dirty:            false
warmup:               10
repetitions:          50
```

The same Slurm allocation and software environment were used for both required workloads. The repository test suite passed in this Singularity runtime, and `ruff check .` plus `ruff format --check .` passed for the validated source state.

### 13.2 Synthetic S3 reference

The synthetic S3 workload contains four non-physical graphs of 8192 nodes each:

```text
physical graphs: 4
nodes:           32768
edges:           65530
input channels:  momentum.x, momentum.y, momentum.z
target channel:  rho.value
model parameters: 4
```

Measured reference:

| Region | Median latency | Mean ± population std | Node throughput | Incremental CUDA peak allocation |
|---|---:|---:|---:|---:|
| host pack + task prepare | 2.0803 ms | 2.0816 ± 0.0788 ms | 15.752 Mnodes/s | n/a |
| forward | 0.0760 ms | 0.0766 ± 0.0056 ms | 431.160 Mnodes/s | 131072 B |
| training iteration | 2.4206 ms | 2.4156 ± 0.0430 ms | 13.537 Mnodes/s | 1312256 B |

### 13.3 Real `HIT_LES_FORCED` reference

The real HIT workload uses the established snapshot/mesh pair and current hex-derived directed edge representation:

```text
physical graphs: 1
nodes:           35937
edges:           209088
input channels:  rho.value, rhou.x, rhov.y, rhow.z, rhoE.value
target channels: rho.value, rhou.x, rhov.y, rhow.z, rhoE.value
model parameters: 30
periodic cross-boundary edges: not augmented (M3.2 deferred scope)
```

Measured reference:

| Region | Median latency | Mean ± population std | Node throughput | Incremental CUDA peak allocation |
|---|---:|---:|---:|---:|
| host pack + task prepare | 19.0759 ms | 19.0874 ± 0.2030 ms | 1.884 Mnodes/s | n/a |
| forward | 0.0549 ms | 0.0558 ± 0.0039 ms | 654.828 Mnodes/s | 1767424 B |
| training iteration | 1.5733 ms | 1.5766 ± 0.0215 ms | 22.841 Mnodes/s | 5752320 B |

For the real HIT training measurement, PyTorch reported 72,617,984 B allocated before the measured region, 78,370,304 B peak allocated, and 100,663,296 B peak reserved.

### 13.4 Interpretation boundary

These values establish a reproducible framework/null-model reference on the GH200 target environment. They do **not** establish graph-attention throughput or edge-scaling behavior because `NodeLinearBaseline` consumes neither coordinates nor `edge_index`.

The real HIT training iteration being faster than synthetic S3 must not be interpreted as HIT being intrinsically cheaper. The null model is too small to saturate the GPU, and the reference loss performs per-physical-graph reductions. S3 contains four graphs while the HIT benchmark contains one, so graph-count-dependent framework overhead is visible at this scale.

Forward times are only tens of microseconds. CUDA synchronization, launch/dispatch overhead, and clock state are therefore a material part of the measurement; the difference between S3 and HIT forward latency is not an architecture-level result.

The approximately 9.2x higher real-HIT host preparation latency reflects the complete measured host preparation path for a richer five-field physically nondimensionalized task and a larger packed topology. HDF5 reading and one-time hex-to-edge construction remain outside that repeated timing region.

The small CUDA memory footprint is also a property of the affine null model. The real HIT edge tensor is present in the prepared batch but is not consumed by the model. M8 must establish its own memory evidence once sparse graph attention actually operates on edges.

## 14. Completed M7 validation gate

The M7 completion gate is satisfied:

```text
software gate:
  pytest                       PASS
  ruff check .                 PASS
  ruff format --check .        PASS
  python scripts/inspect_config.py  PASS

target benchmark gate:
  synthetic S3, single GH200   TARGET_VALIDATED
  real HIT, single GH200       TARGET_VALIDATED
```

M7 is therefore complete for the frozen single-GPU null-baseline scope. M8 may use these measurements as the reference floor, but every graph-aware model must be benchmarked separately before receiving any measured performance claim.
