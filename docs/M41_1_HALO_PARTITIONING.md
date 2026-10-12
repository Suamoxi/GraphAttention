# M41.1 — Exact single-GPU halo partitioning for M37

## Scope and scientific contract

M41.1 adds a **partitioned inference executor**, not a new model or
super-resolution task. It does not modify M37's forward path, trained weights,
checkpoint format, exact-two-hop graph topology, Flow Matching loss, or Heun
solver. The original, non-partitioned call remains the numerical reference.

This work is independent of `dnstoles` (the DNS-to-LES data pipeline / M42).

**Invariant:** for the same original physical graph, state and conditioning,
a partitioned DiNAT-DiT forward must agree with the original M37 forward to
floating-point tolerance on every node. The same is tested after each DiT
block, not only at the final output.

M41.1 supports **one physical graph on one GPU**. Global hidden states remain
resident on that GPU. Partitioned training, multi-GPU halo communications and
out-of-core execution are later milestones.

## Why layerwise haloing?

M37 alternates the original local graph topology and **exact-two-hop**
topology across 10 DiT blocks. With alternating one- and two-hop attention the
network can depend on nodes up to ~15 original graph hops away. Using one
static halo that is this deep would be prohibitively duplicative.

Instead, at every transformer block:

1. Obtain a read-only layer-`l` hidden field.
2. For every ownership partition, gather its owned nodes and the halo nodes
   needed for **that block's** incoming attention edges (the cached plan
   currently uses the union of one-hop and two-hop sources).
3. Compute the existing DiT block on the local read tensor, supplying **all
   original directed incoming edges for owned targets**. Attention scores and
   the softmax denominator are computed over the full original neighborhood.
4. Copy only the owned-node block outputs into a separate layer-`l+1` tensor.
5. After all partitions finish, replace the previous hidden field. This is a
   layerwise halo refresh; no Gauss-Seidel/in-place cross-partition updates.

The current prototype invokes the full DiT block over owned-plus-halo nodes,
then discards outputs for halo nodes. This is simple and numerically faithful,
but performs some redundant nodewise normalization and MLP computation.
Query-only / owned-only optimization belongs to M41.2.

## Graph topology, geometry and conditioning

- The `edge_index` convention is **source/key -> target/query**.
- The exact-two-hop topology is generated from the **global original graph**
  by the normal geometry pipeline. Do **not** rebuild it from partitions.
- Local and dilated self-token inclusion follows the existing M37/M40 model
  flags. M41 does not silently insert periodic boundary edges.
- A node is owned by exactly one partition; halo nodes do not produce
  separately owned outputs or separately counted losses.
- The original physical graph's `centered_bbox` coordinate normalization is
  computed once, not independently per partition.
- Original physical coordinates are used for the geometric edge bias.
- One physical-graph condition embedding (including Flow Matching time)
  is shared across all partitions.
- Each partition caches read/owned mappings, local/dilated incoming edges,
  relative displacements and, for the `torch_sparse` backend, CSR attention
  topology. These are reused at each ODE evaluation for a fixed mesh.

The initial partitioning policy uses axis-aligned spatial bins. The interfaces
operate on global node IDs and directed graph edges, so a METIS or other
unstructured graph partitioning policy can be substituted later.

## Modules

- `src/graph_attention/geometry/partitioning.py`: deterministic ownership IDs.
- `src/graph_attention/models/partitioned_dinat.py`: cached halo plan and
  inference-only layerwise executor.
- `tests/unit/test_partitioned_dinat.py`: attention-enabled per-layer parity,
  cross-partition exact-two-hop edge verification, Heun-like repeated
  evaluation test, and CUDA `torch_sparse` test.
- `scripts/benchmark_m41_partitioned.py`: load the real M37 checkpoint,
  compare forward outputs and timings on one held-out graph.
- `scripts/slurm/m41_1_halo_partitioned_m37.slurm`: test gate + benchmark
  for the Calypso Grace GH200.

## Basic API

```python
from graph_attention.geometry.partitioning import coordinate_block_partition_ids
from graph_attention.models.partitioned_dinat import (
    build_dinat_partition_plan, forward_partitioned_dinat,
)

model.eval()
# coords / edge_index / attention_edge_indices represent ONE original graph.
owners = coordinate_block_partition_ids(coords, blocks=(2, 2, 2))
plan = build_dinat_partition_plan(
    model,
    coords=coords,
    edge_index=edge_index,
    attention_edge_indices=attention_edge_indices,
    partition_ids=owners,
)

# Repeat at each Flow Matching ODE evaluation with the current state/time:
with torch.inference_mode():
    prediction = forward_partitioned_dinat(
        model,
        inputs=current_state,
        plan=plan,
        conditioning=physical_graph_condition_and_time,
    )
```

The builder disables inference mode internally and owns its coordinate snapshot.
Plans can be built inside `torch.inference_mode()` and reused outside it. Treat
all plan tensors as immutable. Rebuild after changes to geometry, edges, model
self-loop policy, device or dtype; dtype mismatch is rejected explicitly.
Empty spatial bins are removed while retaining deterministic spatial ordering.

The benchmark uses a small model-call adapter to pass partitioned execution to
**the unchanged `FlowMatchingTask.sample_standardized` implementation**. Both
paths receive the same physical batch, sampling key and seed. Thus the complete
50-step Heun comparison uses identical Gaussian noise, conditioning and 100
model evaluations. This is a validation adapter, not a new production sampler.

## Run on Calypso

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_1_halo_partitioned_m37.slurm
```

Default layouts are `1x1x1`, `2x1x1`, `2x2x1`, `2x2x2`, `4x2x2`.
Each runs in a separate process with the same first held-out sample and weights.
The Slurm job first runs partition, original DiNAT and Flow Matching task tests.
It stops on any failed test or numerical gate. To select layouts/repetitions:

```bash
M41_BLOCKS="1 1 1;2 2 2" M41_REPEATS=3 CHECKPOINT=best.pt \
    sbatch scripts/slurm/m41_1_halo_partitioned_m37.slurm
```

Reports are stored in
`GraphAttention_runs/m41_1_halo_partitioned_m37/<job-id>/blocks_XxYxZ.json`.
Set `OUT` to override the directory. The dataset, frozen standardizers, split
manifest and model settings come from the saved training run. Checkpoint loading
is strict. The benchmark requires the saved `torch_sparse` backend, one physical
graph and 35,937 nodes (override `--expected-nodes` only for an intentional mesh
change). The input probe uses the task's deterministic validation construction.

Each JSON records:

- Output, every transformer-layer, and complete Heun errors: maximum absolute,
  RMS, relative L2, number of tolerance violations and finite-value checks.
- Exact incoming-edge order/multiplicity checks for both effective topologies,
  including configured self loops, and exact node-ownership coverage.
- Individual synchronized wall-clock timings, mean, median and sample standard
  deviation for forward and complete generation; two forward warmups by default.
- Baseline, peak allocated, incremental peak allocated and peak reserved GiB.
- Plan construction time, owned/halo nodes and edge counts per partition,
  nodewise work ratio `sum(read_nodes)/N`, and forward overhead percentage.
- Git SHA/dirty state, runtime/GPU information, precision policy, all arguments,
  resolved training config, split manifest, sample ID and SHA256 hashes of the
  checkpoint, config, manifest and frozen standardizers.

The numerical gate uses elementwise `abs(new-ref) <= 2e-4 + 2e-4*abs(ref)`.
This is a provisional FP32 acceptance tolerance, allowing different GEMM shapes
and reduction roundoff; it is not a measured error bound. TF32 and autocast are
disabled. The same gate applies to each layer and the accumulated Heun result.
Do not loosen it automatically after a failure: inspect the first divergent
layer and absolute/RMS errors first. Near zero, the absolute term controls the
comparison. NaN/Inf is always a failure. Failed numerical comparisons are saved
before nonzero exit; setup/runtime failures instead appear in Slurm logs.

Timings exclude loading, graph construction, transfers and plan construction;
partition timings include gathers, duplicated halo nodewise work and writes.
The original path runs before allocating a partition plan. Partition memory
includes the resident plan **and the reference model's warmed CSR cache**, so
this is a conservative comparison, not an isolated minimum-memory claim.
CPU diagnostic snapshots are outside timings and do not contribute to GPU peak
memory. Reserved memory is allocator-dependent; compare allocated memory first.
No isolated kernel attribution is claimed by the aggregate overhead percentage.
A speedup below one means partitioning is slower; that is acceptable for M41.1.

## Local validation and evidence status

```bash
PYTHONPATH=src OMP_NUM_THREADS=1 python -m pytest \
    tests/unit/test_partitioned_dinat.py tests/unit/test_dinat_dit.py \
    tests/unit/test_dinat_self_attention_ablation.py \
    tests/unit/test_sparse_transformer.py tests/unit/test_flow_matching_task.py -q
```

The new tests cover asymmetric directed edges (which detect edge reversal),
existing self loops, isolated targets/empty CSR rows, all self policies,
inference-created plans, empty spatial bins, dtype mismatch and the actual
50-step sampler with identical first states and exactly 100 calls per path.
CPU `torch_sparse` tests exercise the same native PyTorch CSR code; they do not
substitute for the CUDA test or pretrained checkpoint validation.

Local verification (2026-10-12): **59 passed, 2 skipped** with PyTorch on CPU.
Skipped: CUDA partition parity and unavailable DGL backend. Benchmark CLI help,
Python compilation, Ruff and Slurm shell syntax were also checked.

Calypso GH200 **TARGET_VALIDATED** on 2026-10-12 in job `461023` (clean
`main` commit `2c28a89`): 47 selected tests passed, one optional DGL test was
skipped; all five requested partition layouts passed real M37 checkpoint,
per-layer, graph-connectivity and 50-step Heun numerical gates. Peak allocated
GPU memory decreased with additional partitions, but forward and Heun time
increased substantially. See `docs/M41_1_HALO_PARTITIONED_DINAT.md` for the
measured performance table, numerical errors and provenance. This is a
single-graph / FP32 / GH200 numerical result, not a general guarantee of
speedup or memory savings on other meshes. A repository-wide pytest run
remains advisable.

## Validation criteria

1. All synthetic, partition and CUDA sparse-attention regression tests pass.
2. A real trained M37 checkpoint agrees on the held-out mesh to floating
   tolerance, including nonzero trained attention gates.
3. Changing partition layouts does not change the physical prediction.
4. No periodic edges, self-loops, geometry transforms or graph attention
   neighbors change except as already defined by the existing model.
5. Runtime and peak memory are measured and documented. On a `33^3` graph,
   the partitioned implementation may be slower and need not save memory
   overall due to cached halos and increased kernel launches.

## Limitations and next steps

- M41.1 retains the full hidden field on GPU and uses a local read buffer for
  each partition. It cannot yet execute a graph whose global hidden state is
  too large for a single GPU.
- Halo representations in M41.1 are **reads**, not independent stochastic
  samples. The future conditional generator must maintain globally
  consistent noise across partitions.
- There is no partitioned backward or gradient exchange yet. Do not use this
  executor for training.
- Complete 50-step Heun generation was target-validated on the trained M37
  checkpoint in Calypso job `461023`: identical initial Gaussian noise,
  100 network evaluations per path, and no numerical tolerance violations.
- M41.2 should avoid redundant halo MLP work and study more memory-efficient
  storage and execution. M41.3 can add multi-GPU communication after parity.
