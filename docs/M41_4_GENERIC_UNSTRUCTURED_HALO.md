# M41.4 — Generic unstructured-mesh halo construction + 257^3 experiment

## Scientific objective

Enable the original M37 AlternatingDilatedGeometricDiT to perform exactly
the same one-hop and exact-two-hop owned-query attention on unstructured
mesh topologies, without constructing the global exact-two-hop graph.
No Cartesian lattice, node numbering, or element shape assumption enters
the model executor.

For target j, one-hop sources k are incoming graph edges k->j.
Exact-two-hop sources i satisfy an original-graph walk i->k->j, excluding
self and direct incoming edges and deduplicating multiple paths. Optional
self-attention respects the checkpoint configuration. The layerwise
double-buffer synchronization of M41.3 is unchanged.

## Code and architecture

- src/graph_attention/geometry/external_csr.py: bounded-HDF5-chunk
  reading and target-bucket external sorting of physical element edges.
  Hex meshes use M37's exact twelve canonical element edges. Tetrahedral
  meshes use the six unique pairwise edges. Arbitrary directed edge
  blocks are supported by build_disk_csr. The one-hop CSR is stored on
  disk in target-major order, with indptr.npy and indices.bin.
- src/graph_attention/geometry/partition_csr.py: ownership may be any
  contiguous integer partition labels, including graph partitioners.
  spatial_owners reproduces the existing coordinate bounding-box bins
  as one option. Exact-two-hop edges are computed in bounded batches
  only for OWNED query targets. Per-partition files are written separately.
- src/graph_attention/models/disk_streamed_dinat.py: adapter that loads
  each part into the existing M41.2 rectangular CSR and M41.3 partition
  types. Relative displacements use actual coordinates. The original
  forward_streamed_dinat is reused without changing trained model weights.
  Plan files may be lazy-loaded or cached in CPU RAM.
- tests/unit/test_m41_4_generic_csr.py: irregular perturbed and shuffled
  hexahedral meshes, torch M37 graph-edge parity, exact-two-hop parity,
  full-forward parity, self-loop cases, tetrahedra and directed edges.
- scripts/m41_4_generic_large.py: full 257^3 preparation, strict M37
  checkpoint loading, measured full forward, and resumable 50-step Heun.

## Acceptance sequence

1. First, run irregular/unstructured numerical regression tests and
   require zero failures and M37 full-output parity at 2e-4.
2. Build the real 257^3 disk one-hop CSR from one-based connectivity
   and extract partition-local two-hop edges. Default coordinate
   ownership is 4 x 4 x 4 (64 parts), with a maximum of 400,000 owned
   nodes per partition. No Cartesian index map is used.
3. Run a single complete 257^3 M37 forward using streamed hidden states
   and record runtime, finiteness and GPU memory.
4. Then attempt 50-step Heun (100 NFE). A step completes both model
   evaluations and writes a resumable checkpoint. The default is one
   step per Slurm allocation until timings are measured.

## Grace commands

Each submitted script requests partition=grace, gpu:1, four CPUs,
four hours and **no explicit memory limit**.

First run the unstructured tests + complete generic preparation:

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_4_generic_257_prepare.slurm
~~~

Inspect:
 /scratch/coop/theret/GraphAttention_runs/m41_4_257_generic_graph/prepare.json

If it passed, run the first measured full-resolution forward:

~~~bash
sbatch scripts/slurm/m41_4_generic_257_infer.slurm
~~~

Inspect:
 /scratch/coop/theret/GraphAttention_runs/m41_4_257_inference/forward/forward.json

If that passed, begin generation:

~~~bash
export M41_4_MODE=heun
export M41_4_STEPS_THIS_JOB=1
sbatch scripts/slurm/m41_4_generic_257_infer.slurm
~~~

Resubmit the same command (same checkpoint, seed, graph, and step
count) to continue from progress.json. Do not submit simultaneous
jobs to the same generation directory.

## Important limitations and scientific caveats

- Generic mesh topology means arbitrary irregular hexahedral or
  tetrahedral elements and arbitrary node order. Other cell families
  require a defined local-edge template or externally supplied edges.
- Current ownership uses geometric bins (not METIS). Unstructured
  meshes are supported but load balance and halo sizes are not yet
  optimized by graph partitioning.
- Inference only, one physical graph, one GPU. No global exact-two-hop
  graph and no full GPU hidden tensor are constructed.
- Coordinate units must match the physical-nondimensionalization
  convention used in M37 training. The preparation runner supports
  M41_4_COORD_DIVISOR. For the supplied Re150/Ma0.3 hip.in, the
  box side is 0.0006227710672453675 m; this is the default
  coordinate divisor in the Grace preparation job to reproduce
  a unit-box coordinate convention. M37 used its own source
  HIT_LES_FORCED box length 0.0005668079275737893 m as L_ref.
  Matching these dimensionless coordinate conventions is necessary
  but does NOT establish statistical transfer across different
  Reynolds and Mach numbers.
- The M37 checkpoint was trained on the much smaller HIT_LES_FORCED
  dataset, not this Re150/Ma0.3 257^3 simulation. A successful forward
  or Heun rollout demonstrates execution feasibility, not quality.
- No real 257^3 graph preparation or inference results have been
  obtained yet. The external sort requires scratch disk space and
  the Heun runtime may require multiple Grace allocations.
