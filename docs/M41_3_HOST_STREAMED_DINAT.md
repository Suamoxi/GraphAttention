# M41.3 — single-GPU, CPU-host-streamed layerwise DiNAT-DiT

**Status: IMPLEMENTED; M37 numerical parity VALIDATED on GH200 job 461341 at 35,937 nodes. Real 257^3 large-mesh execution remains UNTESTED.**

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
violations** to be accepted. This was validated on GH200 job 461341 (2026-10-12) for all five partition layouts. The original M37 checkpoint is reproduced under the specified tolerances.

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

## M41.3 actual GH200 evidence — job 461341, 2026-10-12

Original source: Google Drive
<https://drive.google.com/drive/folders/1u3ASJVKPZ9UQ0ic0QzL9eUvSBqBM4kTa>.
Repository commit recorded in the run:
`6b6324d1138deaba22bf4d9955573eab495d4ec4`.
All five layouts passed M37 parity for every transformer layer,
final model output and 50-step/100-NFE Heun generation.
At tolerance `atol=rtol=2e-4` all were finite and had zero violations.

| Blocks | M41.2 forward median (ms) | M41.3 forward median (ms) | M41.2 peak GPU GiB | M41.3 peak GPU GiB |
|---|---:|---:|---:|---:|
| 1x1x1 | 34.81 | 55.98 | 0.485 | 0.345 |
| 2x1x1 | 44.92 | 75.63 | 0.414 | 0.258 |
| 2x2x1 | 69.19 | 113.22 | 0.398 | 0.210 |
| 2x2x2 | 103.24 | 190.02 | 0.383 | 0.154 |
| 4x2x2 | 161.96 | 348.08 | 0.377 | 0.119 |

For P16: M41.3 uses 0.119 GiB vs 0.377 GiB peak allocated
(about **68.4% less**), but takes **2.15x longer** for a forward.
The corresponding Heun medians are 33.446 s vs 12.366 s.
M41.3's largest layer error (absolute) was 5.86e-5,
final-output max error 4.07e-5, Heun max error 9.54e-7.

**Important**: the paired script keeps original M37 reference data
on GPU, so these allocations overestimate a standalone CPU-only
loading configuration and cannot predict the peak on a 257^3 mesh.
Only two timed repetitions were used; do not overinterpret small
cross-run performance differences.

## Actual 257^3 mesh and snapshot — metadata-only next gate

Uploaded to Drive on 2026-10-12:

- Mesh: `Re150/Ma0.3/MESH/mesh.mesh.h5`, about **1.68 GB**
  (1,678,544,536 bytes). Drive ID:
  `1DSeN5Lp9NK6iRPleogvq2tyoq8GILika`.
- Snapshot: `Re150/Ma0.3/SOLUT/solut_hit_00000001.h5`,
  about **2.65 GB** (2,648,061,484 bytes). Drive ID:
  `1cg3HmzfyDTh3S2OHdO636BbAeK2Ji1RU`.
- The mesh's `hip.in` describes 257x257 structured-plane
  generation and a 256-interval extrusion, consistent with a
  `257^3` node lattice. This does **not** establish actual HDF5
  field shapes, vertex ordering, indexing base or periodic connectivity.

Do **not** call the existing GraphTaskCollator or
`build_streamed_dinat_plan` on this full mesh until graph preparation
is redesigned or memory-tested. The current
`hex_connectivity_to_edge_index` materializes 12 candidate undirected
hex edges per cell, uses `torch.unique` across the entire mesh, and
`exact_two_hop_edge_index` materializes length-two walks globally.
Even on a perfectly regular cubic mesh, intermediate memory and
sorting costs can be enormous. The old M41.3 planner also scans
global edges once per partition. No actual 257^3 end-to-end inference
has been attempted.

A new **metadata-only, bounded-memory CPU preflight** is available:

- `scripts/preflight_m41_3_large_hdf5.py`
- `tests/unit/test_m41_3_large_hdf5_preflight.py`
- `scripts/slurm/m41_3_257cube_preflight.slurm`

On Grace, the archive root is `/archivenfs/phlusim/HIT_DNS_COMP`.
The M41.3 preflight defaults to
`/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3/MESH/mesh.mesh.h5`
and
`/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3/SOLUT/solut_hit_00000001.h5`.
The script explicitly binds `/archivenfs` into Singularity and checks
both paths before launching HDF5 inspection. If the Re150/Ma0.3
subdirectory differs on Grace, override `M41_3_DATA_BASE` or set
`MESH_FILE` and `SNAPSHOT_FILE` individually. The job is CPU-only;
supply `sbatch --partition=<cpu partition>` if the default partition
requires a GPU.

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_3_257cube_preflight.slurm
```

The report `preflight.json` records HDF5 path/shape/dtype/chunking,
node and cell counts, a bounded sample of field/coordinate values,
connectivity extrema scanned without reading entire arrays into RAM,
and theoretical GPU/CPU buffer budgets. A successful preflight
establishes **metadata compatibility only**; it does not validate
periodic boundaries, geometric ordering, graph topology, trained weights
or 17-million-node end-to-end memory scaling.

Next implementation gate after seeing this preflight: if the mesh is
regular, construct per-partition 1-hop/exact-two-hop CSR directly from
verified Cartesian lattice node IDs and only retain CPU partition
metadata, rather than materializing global graph paths; otherwise
implement a general streamed edge-bucketing builder. Then test small
subdomains against the validated M37 model before attempting 257^3.

