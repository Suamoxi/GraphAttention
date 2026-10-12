# M41.1 — single-GPU layerwise halo DiNAT-DiT inference

Status: **TARGET_VALIDATED for single-GPU numerical equivalence on GH200**, job `461023`,
2026-10-12, commit `2c28a89`. Selected unit tests: 47 passed, one optional DGL
backend test skipped. The full repository-wide test suite was not run in this job.
Scope: existing M37 `AlternatingDilatedGeometricDiT` checkpoint, single physical graph,
single GPU, inference only. M42 dataset preparation is out of scope.

## Mathematical contract

The global directed edges use `edge_index[0] = j` (key/value source) and
`edge_index[1] = i` (query target). Let `O_p` be the disjoint nodes owned by
partition `p`, with each node owned exactly once. Let `E_local` and
`E_dilated` be the **original** one-hop and exact-two-hop edge sets, including
only self edges permitted by the saved M37 model configuration.

The halo for partition `p` contains source nodes in either edge set whose
target is owned by `p`. The plan caches `R_p = O_p union halo_p`, and for
each topology caches **all** edges with target in `O_p`. Their global
source/target indices are remapped to the sorted local list `R_p`. No
exact-two-hop edge is recomputed after partitioning, and no edge is lost merely
because the source and target have different owners.

Each transformer layer reads only the *previous* global hidden tensor,
gathers `h[R_p]`, evaluates the original DiT block with its original weights
and owned-target incoming edges, then scatters **only owned outputs** to a
separate global buffer. The buffer is swapped only after *all* partitions have
finished. Graph softmax is target-wise; each owned target sees precisely its
original key set and denominator. LayerNorm, adaLN conditioning, residual
connections, MLP and the final head are nodewise after the attention operation.
Consequently, ignoring halo-target outputs is exact up to finite-precision
kernel differences.

Global `centered_bbox` coordinate normalization is performed **once on the
entire physical graph**; it is not recomputed for each partition. Relative
edge displacement uses raw global geometry `coords[source] - coords[target]`.
The geometric M37 attention bias and optional self-loop policies are unchanged.
CSR uses target rows and source columns; `edge_order` applies the same
per-edge permutation to the learned geometric bias.

The plan and its CSR tensors are built with inference mode disabled and
gradients disabled. This allows a plan built inside an enclosing
`torch.inference_mode()` scope to be reused outside it. Rebuild the plan when
coordinates, connectivity, attention self-loop policy, dtype or device change.
The benchmark sampler adapter rejects switching graph tensor objects.

**Numerical fidelity of node-local stages:** input projection, coordinate
projection and final output projection run once on the complete resident
global tensor, exactly as in the original M37 forward. Only the alternating
DiT attention blocks are evaluated by owned/halo partition. This removes
avoidable differences from partition-dependent CUDA GEMM shapes in these
strictly nodewise projections, without altering M37 model weights, attention
topology or condition embeddings. Partition-local DiT blocks can still select
different CUDA kernels than the full-graph call; equivalence is therefore
evaluated numerically, not assumed bitwise.

**Known overhead:** for now, each owned-plus-halo gather evaluates the full DiT
block and discards results at halo query nodes, including unnecessary QKV,
normalization and MLP work. Further, the same read set contains both one-hop
and exact-two-hop keys. The total gathered nodes divided by global nodes
(`nodewise_work_ratio`) quantifies duplicated per-layer nodewise work, not
an ideal communication-only halo cost. This milestone does **not** promise
lower peak GPU memory or out-of-core execution.

## Source files

- `src/graph_attention/geometry/partitioning.py` — deterministic spatial ownership.
- `src/graph_attention/models/partitioned_dinat.py` — cached partition plan
  and original-block layerwise evaluation.
- `tests/unit/test_partitioned_dinat.py` — synthetic graph, directed,
  optional-self, per-layer, 50-step Heun, and inference-mode regression tests.
- `scripts/benchmark_m41_partitioned.py` — real M37 checkpoint numerical
  validation and CUDA timing/memory measurements.
- `scripts/summarize_m41_partitioned.py` — multi-layout CSV comparison.
- `scripts/slurm/m41_1_halo_partitioned_m37.slurm` — GH200 launch.

## Calypso execution

The default M37 run is:

```text
/scratch/coop/theret/GraphAttention_runs/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16
```

It must contain `best.pt`, `resolved_config.yaml`,
`standardizers.pt`, and `dataset_split_manifest.json`; the configured
dataset must still be readable. The benchmark selects the first held-out test
sample and checks that it contains exactly `33^3 = 35,937` nodes. It
reconstructs the same dataset/task/collator/standardizers/model and loads
`model_state_dict` with `strict=True`. It refuses any backend other than
M37's `torch_sparse`.

Run the M41.1 test gate and all spatial partitions on Calypso:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_1_halo_partitioned_m37.slurm
```

To change the layouts or replicate timings:

```bash
export M41_BLOCKS="1 1 1;2 1 1;2 2 1;2 2 2;4 2 2"
export M41_REPEATS=5
export M41_WARMUP=2
sbatch scripts/slurm/m41_1_halo_partitioned_m37.slurm
```

To run a single layout directly in the existing Calypso environment:

```bash
singularity exec --nv -B /home -B /scratch \
  /softs/local_arm/singularity/images/pyg25.03.sif \
  /scratch/coop/theret/avbpML_env/bin/python3 \
  -m scripts.benchmark_m41_partitioned \
  --run-dir /scratch/coop/theret/GraphAttention_runs/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16 \
  --blocks 2 2 2 --warmup 2 --repeats 5 --sampling-steps 50 \
  --output /scratch/coop/theret/GraphAttention_runs/m41_manual_2x2x2.json
```

The Slurm script launches the existing unit tests before the real checkpoint
benchmark, including the CUDA `torch_sparse` equivalence test. The reference
model is **not retrained or modified**.

## What is compared

The same model, standardized inputs, original graph, and conditioning are
used for both paths, under FP32 with TF32 disabled.

1. **Connectivity:** every original local/exact-two-hop incoming edge,
   including self edges when configured, occurs exactly once in an owned
   partition, with identical source/target and raw displacement. Partitioned
   CSR row/column mappings and row counts are checked.
2. **Single forward:** output arrays plus all intermediate transformer-layer
   representations, using hooks on the *original* M37 blocks.
3. **Full generation:** the task's original `sample_standardized` Heun solver
   performs 50 steps / 100 network evaluations, seed 5678. Forward-pre-hooks
   verify the first sampled state is bitwise identical and both models receive
   100 evaluations. Final generated arrays are compared numerically.
4. **Efficiency:** synchronous wall-clock forward latency with warmups and
   repetitions; complete sampler latency; peak allocated/reserved CUDA
   memory. Data loading, host-to-device transfer, plan construction and
   numerical comparisons are outside timed forwards.

Defaults for the numerical comparison are `atol=2e-4` and `rtol=2e-4`
for FP32: each component must satisfy
`abs(partitioned - reference) <= atol + rtol * abs(reference)`.
The report also records max absolute, RMS and relative L2 errors, and the
number of violated components. These tolerances acknowledge floating-point
changes from altered GEMM shapes and reduction grouping; they are not a claim
of bitwise identity. The initial generated Gaussian state **must** be bitwise
identical.

To avoid conflating unrelated persistent memory, the benchmark first measures
reference-only inference with its full-graph CSR cache, then releases that
cache before building the partition-only CSR plan. `plan_persistent_allocated_gib`
reports device allocations retained by the plan, and each mode reports its
total CUDA peak plus transient incremental peak. The values are empirical
within one process, not a theoretical memory bound.

## Outputs and interpretation

Default job outputs go to:

```text
/scratch/coop/theret/GraphAttention_runs/m41_1_halo_partitioned_m37/<job-id>/
    git_commit.txt
    gpu_info.csv
    blocks_1x1x1.json
    blocks_2x1x1.json
    blocks_2x2x1.json
    blocks_2x2x2.json
    blocks_4x2x2.json
    comparison.csv
```

Individual JSON reports preserve file hashes, resolved configuration,
test-sample ID, geometry size, block partition counts, topology checks,
per-layer/output/Heun error statistics, measured timings and CUDA memory
statistics. The summary CSV refuses to combine different checkpoints,
test samples, precision configurations or generation settings.

**Acceptance:** all per-layout `passed` flags must be true, the reference
M37 tests must still pass, and the actual checkpoint must load successfully
on Calypso. `forward_overhead_pct > 0` means partitioning is slower; this
is expected for small meshes because M41.1 still computes unused halo-node
outputs and adds Python/kernel-launch overhead. Compare both median latency
and `nodewise_work_ratio` as the number of partitions rises. Do not interpret
a lower incremental peak as lower total GPU memory without also comparing
`peak_allocated_gib` and retained plan memory.

If output parity fails, inspect the earliest failing layer in
`errors.layers`; if connectivity checks fail, inspect edge direction,
self loops and CSR row counts first. If only the generated output fails,
inspect `generation_checks` and the Heun trajectory. Record the job log
and JSON files for review. The section below records the first **successful**
full M37 checkpoint validation.

## Target GH200 numerical and performance evidence — 2026-10-12

Source: Calypso Slurm job **`461023`** (provided stdout and stderr logs);
clean `main` commit `2c28a89bb6504cdf8e42d7ee154ccde6c3756cca`.
All five JSON reports share the same M37 `best.pt` SHA-256:
`01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8`.
The held-out sample is `solut_hit_00007500`, with 35,937 nodes, the original
209,088 local directed edges and 608,058 exact-two-hop directed edges.
Other artifact hashes and the full resolved configuration are preserved in
the per-layout JSON reports. Hardware is NVIDIA GH200 480GB, CUDA 12.8,
PyTorch `2.7.0a0+7c8ec84dab.nv25.03`, single GPU. All measurements are
FP32 with TF32 disabled.

The selected Slurm test gate completed **47 passed, 1 skipped**; the skipped
test concerns optional DGL dependencies (missing `torchdata.datapipes`), not
M37's `torch_sparse` backend. All five checkpoint reports have both
`passed=true` and `evidence=TARGET_VALIDATED`. Each reports exact ownership
coverage, unchanged incoming local/exact-two-hop topology and geometry,
zero numerical tolerance violations after each of 10 transformer blocks,
and a passing 50-step Heun comparison. Both paths made exactly 100 model
evaluations with bitwise-identical initial Gaussian noise, seed 5678.

Largest observed absolute error across **all** tested layouts:

- Individual transformer layer: `5.9425831e-5`
- Single forward output: `5.3167343e-5`
- Final Heun-generated field: `1.9073486e-6`

All compared tensor elements satisfy
`abs(partitioned-reference) <= 2e-4 + 2e-4*abs(reference)`.
The errors are not zero: this is tolerance-based mathematical equivalence,
not bitwise identity or a proof of arbitrary-grid accuracy.

Measured medians (five repetitions for each mode and layout):

| Spatial blocks | P | Read-node ratio | Original forward (ms) | Partitioned forward (ms) | Forward overhead | Original Heun (s) | Partitioned Heun (s) | Partitioned peak allocated (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1×1×1 | 1 | 1.000 | 31.45 | 28.95 | −7.9% | 2.743 | 2.528 | 0.433 |
| 2×1×1 | 2 | 1.121 | 32.01 | 40.28 | +25.9% | 2.786 | 3.614 | 0.353 |
| 2×2×1 | 4 | 1.246 | 30.90 | 65.58 | +112.2% | 2.736 | 5.879 | 0.310 |
| 2×2×2 | 8 | 1.375 | 33.31 | 126.99 | +281.3% | 2.783 | 12.099 | 0.251 |
| 4×2×2 | 16 | 1.632 | 31.50 | 232.88 | +639.4% | 2.750 | 22.954 | 0.216 |

The unpartitioned reference peak allocated memory is approximately **0.440 GiB**
in every run. The 16-partition result uses 0.216 GiB peak allocated, a measured
50.8% reduction in this specific setting. Plan-construction time ranges from
0.071 to 0.150 seconds and retained plan allocations are approximately
0.060–0.065 GiB. Allocated CUDA memory is not identical to total device memory
usage, and this result is not an out-of-core memory bound.

The `1×1×1` plan is faster than the original forward in this experiment,
plausibly because relative geometry is precomputed and reused. More partitions
are substantially slower; at `4×2×2`, forward latency is about 7.4 times the
original despite just 1.63 times total gathered nodewise work. The data are
consistent with repeated small CUDA kernels, reduced GPU occupancy, Python
launch overhead and redundant halo-target computation, but this breakdown is
**not** established without profiler evidence. This is a correctness milestone,
not a favorable partition-scaling performance result.

Results directory:

```text
/scratch/coop/theret/GraphAttention_runs/m41_1_halo_partitioned_m37/461023/
    blocks_1x1x1.json ... blocks_4x2x2.json
    comparison.csv
    git_commit.txt
    gpu_info.csv
```

**M41.1 scope status:** the single-GPU real-checkpoint equivalence, 50-step
Heun comparison and specified initial GH200 benchmarks are validated.
The full repository-wide regression suite was **not** run in this job;
the `torch_sparse` path and relevant original M37/Flow Matching tests were
included in the selected gate. Future optimization (owned-query-only
computation, fewer launches or multi-GPU exchange) belongs to a separate
milestone; M41.1 alone is not a practical speedup for this 35,937-node mesh.

## Calypso diagnostic history

The first supplied Calypso logs (`m41_1_halo_460988.out` and `.err`)
showed that the job executed **older commit `8f9ae949`**, not the later
M41.1 validation commit `1fe30cef`. A GH200 GPU ran 13 pytest items: 12
passed and `test_torch_sparse_gpu_parity` failed. The largest output error
was `3.805309534e-4` (2/90 tensor elements outside `atol=rtol=2e-4`).
The checkpoint benchmark never ran in that job. This alone does not
establish a faulty partition topology or a passing M37 comparison.

The follow-up changes keep attention partitioned but compute node-local input
and output projections globally, disable TF32 explicitly in the CUDA parity
test, and capture reference-vs-partition errors for each intermediate layer.
The revised CUDA test and real-checkpoint validation passed on 2026-10-12
(job `461023`) without changing numerical tolerances. The prior failure's
individual cause was not isolated because both projection shape and TF32
settings changed. This does not establish bitwise reproducibility across
GPU kernels.

For diagnostic data collection **only**, the Slurm script accepts
`M41_CONTINUE_AFTER_FAILURE=1`; it then attempts each benchmark layout
even if tests or an earlier layout fail. Crucially, the job **still exits
nonzero** if any validation failed, and individual JSON reports retain
their `passed` status. By default, failures continue to stop the job.

