# M41.3 — single-GPU, CPU-host-streamed layerwise DiNAT-DiT

**Status: IMPLEMENTED, GH200 CHECKPOINT VALIDATION PENDING.**

This is a third, explicitly opt-in inference executor for the original
trained M37 `AlternatingDilatedGeometricDiT`; it does not change M37,
M41.1, M41.2, Flow Matching, model weights, graph topology, self-loop
policies or the original sampler. M42/dnstoles is not involved.

## Objective and what it does *not* claim

M41.1 and M41.2 hold the full transformer hidden tensor in GPU memory;
M41.2 also computes global QKV and global MLP activations. At
`257^3 = 16,974,593` nodes with `hidden_dim=128`, one global FP32
hidden tensor is 8.09 GiB, global QKV 24.28 GiB and a global
`mlp_ratio=4` intermediate 32.38 GiB. The sum is **not** a peak-memory
prediction, but full-graph activations become problematic at this scale.

M41.3 holds the entire *previous* and *next* hidden states on the CPU,
and processes one owned-plus-halo partition at a time on the GPU. No
global hidden, global QKV or global MLP tensor is created on the GPU
**by the executor**. The GPU receives only the partition-local hidden
reads, rectangular CSR topology, relative displacements and weights;
the CPU receives only newly updated *owned* hidden values.

The static full original graph, exact-two-hop edge list, coordinates,
global coordinate normalization and all CPU partition-plan metadata
still must exist on **host RAM**. M41.3 does not implement out-of-core
CPU graph construction, lazy HDF5 graph streaming, multi-GPU execution
or training/backpropagation. Large-graph preprocessing time and peak
host memory are **not established**.

## Mathematical contract

The original attention topology is directed `(source, target)`. Every
owned target has *all* of its original incoming sources for its active
attention layer: one-hop for layers 0,2,4,... and exact-two-hop for
layers 1,3,5,.... All original geometry and self-loop options are used.

For physical sample `x`, compute each node's initial embedding only
when its owning partition is processed:

```text
host_hidden_0[owned] = input_proj(x[owned])
                       + coord_proj(global_normalized_coords[owned])
```

The global `centered_bbox` statistics are taken across **all nodes**,
not separately per partition. At transformer layer `l`:

```text
old_host_hidden = complete CPU hidden buffer, immutable in this layer
new_host_hidden = separate, uninitialized complete CPU hidden buffer

for each partition:
    r = owned nodes union complete one-hop/two-hop source halo
    local_hidden = old_host_hidden[r] -> GPU
    local_norm1, local_qkv = original block modules on local_hidden
    attention = M41.2 rectangular owned-query CSR,
                original relative edge displacements, per-target softmax
    owned_updated = original out projection / adaLN gates /
                    residual / norm2 / MLP applied ONLY to owned nodes
    new_host_hidden[owned] = owned_updated -> CPU

old_host_hidden = new_host_hidden    # only AFTER all owners are updated
```

Every partition reads **the same previous-layer hidden state**. The
executor never uses partially updated halos from the current layer.
The block's `adaLN_modulation` receives the same graph-level
conditioning as M37, once per layer. Learned parameters are unchanged.

The final conditioned output projection is computed partition by
partition; the full physical output is assembled in CPU memory, then
returned to the caller's input device. Existing 50-step Heun sampling
keeps only the low-channel-count global physical field on GPU, not
full transformer hidden fields.

Because the per-partition linear operations use different GEMM shapes
from the original unpartitioned M37, FP32 results may not be bitwise
identical. The benchmark tests **every layer**, final output and
50-step Heun under `atol=rtol=2e-4`, and must report **zero elementwise
violations** to be accepted. This is pending GH200 validation.

## Implementation

- `src/graph_attention/models/streamed_dinat.py`:
  `StreamedDiNATPlan`, `build_streamed_dinat_plan()`,
  `forward_streamed_dinat()`. The planner takes CPU coordinates,
  one-hop and exact-two-hop edges, and CPU partition IDs. A temporary
  CPU model clone reuses the M41.2 rectangular CSR builder; **the
  original GPU model is neither moved nor mutated**. Cached M37
  full-graph attention topologies must be cleared before plan build.
- Plan metadata (`read_global`, `owned_global`, per-partition
  rectangular CSR and edge displacements) is stored on CPU.
  Staging one partition's CSR and geometry does **not** build all
  partition CSRs on GPU.
- The executor accepts a physical input tensor on CPU or GPU.
  The returned physical output is on the input's original device.
  Debug `return_layer_states=True` creates CPU copies of every
  intermediate hidden state and is unsuitable for a large-volume run.
- Default complete CPU hidden buffers are **not pinned**.
  `--pin-host-hidden` is an opt-in benchmark experiment, which
  page-locks approximately two full hidden-state tensors and may be
  unsuitable for large meshes or cluster pinning limits.
- M41.3 is **single GPU, one graph, FP32, inference-only, torch_sparse**.
  It is not a GPU-aware MPI implementation. CPU/host metadata is
  immutable and should be rebuilt for each different mesh topology.

## Memory budget (theoretical, no large-grid measurements yet)

For `N=257^3`, `H=128`, FP32:

| Allocation | Approximate size | Location |
|---|---:|---|
| Two complete layerwise hidden buffers | 16.19 GiB combined | CPU |
| Complete five-field physical state | 0.316 GiB | CPU or GPU |
| One partition's read hidden, QKV, MLP intermediates | Depends on partition/halo size | GPU |
| Original one-hop and two-hop graph edge lists | ~6 GiB combined if int64 | CPU |
| Normalized coordinates, halo CSR/displacements | Depends on mesh and partition layout | CPU |

Additional host graph-construction intermediates, CUDA-resident model
parameters, allocated sparse kernels, workspace and allocator caching
are **not** included in this table. A 16-way decomposition still has
roughly 1 million owned nodes per partition, so local peak memory and
transfer time need benchmarking. Under memory pressure use more
partitions or eventually group/stream graph metadata.

At `257^3`, the static CPU plan builder currently derives from the
M41.1/M41.2 global-edge planner, including repeated per-partition edge
selection and temporary CSR construction. Its large-mesh runtime and
CPU memory footprint are a known limitation, not a solved production
problem. A future scaling milestone must replace this with partition
graph files / a one-pass edge bucketing planner, ideally integrating
mesh partitions already generated by the HPC solver.

## Real M37 GH200 numerical gate and A/B benchmark

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_3_host_streamed_m37.slurm
```

The Slurm script runs selected M37/M41.1/M41.2/M41.3 unit tests first,
then runs M41.2 `owned_query` and M41.3 `streamed` in **separate
processes for each spatial layout** on the single GH200. Both compare
with the same exact trained M37 checkpoint and held-out graph.
The cases are `1x1x1`, `2x1x1`, `2x2x1`, `2x2x2`,
`4x2x2`. Every run uses 50 Heun steps/100 evaluations with seed 5678,
including per-layer intermediate-state checks.

The initial comparison uses one warmup and **two timed repetitions**.
This is an exploratory performance experiment, not a final statistically
robust performance claim. For additional repetitions set
`M41_3_REPEATS=5`.

Optional controls:

```bash
export M41_3_BLOCKS="1 1 1;2 2 2;4 2 2"
export M41_3_REPEATS=5
export M41_3_WARMUP=2
export M41_3_PIN_HOST=0
sbatch scripts/slurm/m41_3_host_streamed_m37.slurm
```

Use `M41_3_CONTINUE_AFTER_FAILURE=1` only to collect diagnostic
reports after a failed gate; the job still returns **nonzero** and the
failed layouts are not a performance success.

Artifacts under
`/scratch/coop/theret/GraphAttention_runs/m41_3_host_streamed_m37/<job-id>/`:

```text
git_commit.txt
git_status.txt
gpu_info.csv
owned_blocks_1x1x1.json
streamed_blocks_1x1x1.json
...
owned_blocks_4x2x2.json
streamed_blocks_4x2x2.json
comparison.csv
```

**Benchmark memory limitation:** the M41 shared benchmark loads an
unpartitioned M37 *reference* graph and standardized physical fields
onto GPU to compare numerical parity. These input graph and physical
tensors remain resident when measuring M41.3. Therefore, its CUDA
peak does **not** measure a standalone large-graph CPU-input deployment.
Report these memory numbers as a conservative reference-loaded
benchmark. They cannot establish a `257^3` memory peak.

The new report explicitly identifies `hidden_storage=cpu_double_buffer`
and records two CPU hidden buffers' expected size and maximum per-part
read nodes. The strict summary refuses mismatched checkpoints/samples,
missing reports, and any numerical-gate failure.

## Acceptance and next steps

1. **Numerical gate**: selected tests pass and original real M37
   checkpoint parity holds for every intermediate transformer layer,
   final output, and complete Heun generation across all selected
   layouts.
2. **Execution/memory gate**: no global hidden/QKV/MLP tensors on GPU
   inside the streamed executor, a bounded active partition size,
   correct CPU double-buffered layer updates; compare GPU allocated
   memory and CPU host storage explicitly.
3. **Large-volume follow-up (not implemented)**: support a CPU-only
   real dataset loading path without materializing the reference
   model/graph on CUDA, replace the global-edge CPU partition-plan
   builder with a more scalable one-pass / solver-partition reader,
   then measure `65^3`, `129^3`, `193^3`, `257^3`.
   A mesh that cannot fit the original M37 cannot use M37's full-GPU
   parity run; validate equivalent subgraphs and small meshes first.
4. Later: pinned async copies/double-buffer prefetch, grouping multiple
   local GPU partitions, geometry cache, CUDA Graph replay, and MPI
   ownership/halo exchanges. These are **not** part of M41.3 v1.
