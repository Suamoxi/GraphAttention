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
- The external sort requires scratch disk space. Generation throughput
  has not yet been measured, and the Heun runtime may require multiple
  Grace allocations. Peak CPU RAM was **not recorded** in job 464853;
  new runner instrumentation records current/peak process RSS before
  and after partition-plan loading and after the forward.

## Actual GH200 forward evidence — 2026-10-14, job 464853

- Host: `calypso-grace02.calypso.cerfacs.fr`, code revision
  `ebbb732d56d186f3c68059f326e7df6e0fcd1761`.
- The `prepare.json` from the previous stage was present and marked
  complete; the full preparation log was not uploaded with this report.
  The caller requested preloaded CPU partition plans.
- Partition plan loaded in **10.4 s**, with `preload_partitions=true`.
- Original M37 checkpoint `best.pt` SHA-256
  `01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8`.
- Single full-resolution forward at flow time `t=0.0`:
  **158.22969341 seconds**, `[16,974,593, 5]` output shape,
  finite output. Total Python runtime: **170.08620595 seconds**.
- GPU peak **allocated 2.25520754 GiB**, GPU peak **reserved 7.6796875 GiB**
  (PyTorch allocator). The GH200 reports 97,871 MiB total HBM.
- stderr contained only a Singularity read-only CUDA-compat cleanup
  warning and the PyTorch sparse-CSR beta warning.
- Important qualification: no unpartitioned M37 `257^3` output exists,
  so numeric parity was established only on earlier small test graphs.
  The forward demonstrates execution feasibility, not physical validity
  or full-population generation throughput. Actual peak host RAM and
  CPU/GPU transfer time breakdown remain unmeasured.

Next computational gate: measure process peak resident CPU RAM and
compare lazy on-disk partitions vs preloaded CPU partitions on the **same
prepared 64-partition graph**. The updated runner now records
`host_memory_before_plan`, `host_memory_after_plan`,
`host_memory_after_forward`, and `plan_load_seconds`.
Execute separately for the two modes:

~~~bash
# Current preloaded baseline, now with RSS instrumentation
M41_4_PRELOAD=1 OUTPUT=/scratch/coop/theret/GraphAttention_runs/m41_4_257_inference/forward_preload_profile sbatch scripts/slurm/m41_4_generic_257_infer.slurm

# Lazy, disk-backed partition metadata, same mesh and checkpoint
M41_4_PRELOAD=0 OUTPUT=/scratch/coop/theret/GraphAttention_runs/m41_4_257_inference/forward_lazy_profile sbatch scripts/slurm/m41_4_generic_257_infer.slurm
~~~

The scripts reserve Grace, 1 GPU, 4 CPUs, 4 hours, and do not add a
`--mem` directive. They do not rebuild the graph.

## CPU/GPU memory profiling and lazy-CSR failure — 2026-10-14

Evidence: Grace jobs 464863 (preload, passed) and 464864 (lazy,
failed), both on code revision 7dd94c0b.

The real 257^3 preparation manifest reports 64 parts using 4x4x4
coordinate bins, 101,451,264 directed one-hop edges and 303,167,994
directed exact-two-hop edges stored per partition. Maximum owned
partition size is 274,625 nodes; maximum read set is 312,064 nodes.
Total preparation took **96.63197 seconds**, including coordinates,
disk one-hop CSR construction and per-partition exact-two-hop extraction.

| Metric | Preloaded (464863) | Lazy (464864) |
|---|---:|---:|
| Plan setup | 8.854 s | 0.2 s |
| Host RSS after plan | 23.0956 GiB | 1.24 GiB |
| Peak CPU RSS through complete forward | 31.7054 GiB | unknown (failed) |
| Full forward | 171.2683 s, passed | failed |
| Peak GPU allocated | 2.2552 GiB | unavailable |
| Peak GPU reserved | 7.6797 GiB | unavailable |
| Output | [16,974,593,5], finite | none |

The preloaded process began at 1.091 GiB RSS; host RSS at the end
of the forward was 15.313 GiB despite a peak of 31.705 GiB.
This measures process memory, not total node memory. Lazy memory
numbers were sampled only after the plan was opened, not during
a successful forward, so the lazy peak is still unknown.

The lazy run failed inside _gpu_topology with:
RuntimeError: Cannot set version_counter for inference tensor.

Cause: the lazy partition loader assembled CSR component tensors
while PyTorch inference mode was active. Temporarily disabling
inference mode around only the sparse CSR wrapper was insufficient,
because its component tensors were still inference tensors.
The repair disables inference mode for the entire lazy __getitem__
operation, so all tensors are ordinary/versioned. This does not
change M37 graph edges, model weights or numerical behavior.

A local PyTorch CPU minimal reproduction confirmed the diagnosis.
Unit regression runs irregular hex/tet M37 forward parity under
inference mode and checks CSR components. The corrected lazy full
forward subsequently **passed on Grace job 464888**, described below.

### Next job: corrected lazy path (same prepared graph)

~~~bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
M41_4_PRELOAD=0 \
OUTPUT=/scratch/coop/theret/GraphAttention_runs/m41_4_257_inference/forward_lazy_fixed \
sbatch scripts/slurm/m41_4_generic_257_infer.slurm
~~~

Job 464888 completed successfully; the exact run command above is
preserved for reproducibility. No global unpartitioned M37 257^3
reference is available for a same-size numerical comparison.


## Confirmed lazy GH200 full-resolution forward — 2026-10-14, job 464888

- Host: calypso-grace04.calypso.cerfacs.fr. Code revision:
  2ffa461802903f80c4eb9c1e86be69a47b03c81d.
- The new inference-mode numerical regression gate passed:
  **7 passed, 1 PyTorch sparse-CSR beta warning** in 7.81 s.
- Original M37 checkpoint SHA-256:
  01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8.
  Same checkpoint as the previous preloaded run.
- Reused the real HDF5-derived generic 64-partition CSR plan. Node
  count 16,974,593; largest owned/read partitions:
  274,625 / 312,064 nodes. No Cartesian topology optimization.
- Plan opening: 0.14267 seconds. Host RSS before plan 1.09088 GiB;
  after plan 1.24048 GiB; peak before forward 1.61804 GiB.
- Full model forward: **108.94849 seconds**; finite output with
  shape [16,974,593, 5], flow_time=0.0.
  Total Python runtime: **110.43804 seconds**, excluding separate
  pytest gate and process/SLURM container startup.
- Peak process RSS throughout execution: **19.98260 GiB**;
  end-of-forward RSS: 3.04419 GiB.
  Peak GPU allocation: **2.25521 GiB**; reserved: **7.67969 GiB**.
- Reported channel means and standard deviations agree very closely
  with job 464863's preloaded results. This is statistical summary
  consistency, not an elementwise M37 equivalence test.
- Nonfatal stderr: CUDA compatibility read-only filesystem warning
  from Singularity cleanup and PyTorch sparse-CSR beta warning.

### Memory/latency tradeoff at 64 partitions

| Metric | Preload (464863) | Lazy (464888) | Change |
|---|---:|---:|---:|
| Plan setup | 8.854 s | 0.143 s | ~62x shorter |
| Host RSS after plan | 23.0956 GiB | 1.2405 GiB | -94.6% |
| Peak process RSS | 31.7054 GiB | 19.9826 GiB | -37.0% |
| Full forward | 171.2683 s | 108.9485 s | -36.4% |
| Peak GPU allocated | 2.2552 GiB | 2.2552 GiB | unchanged |
| Peak GPU reserved | 7.6797 GiB | 7.6797 GiB | unchanged |

**Interpretation:** the lazy partition mode demonstrated much lower
CPU memory consumption and shorter wall time in these *single*
successful trials on different Grace hosts. These are measured
differences, not yet a causal performance claim: hardware locality,
filesystem caching and CPU memory pressure may confound the speed
comparison. Full run-to-run latency variance is not yet established.

**Next computational experiment:** repeat successful forwards using
identical partitions and both load strategies (distinct report
folders), preferably on the same Grace host when available. Next,
reuse the global one-hop CSR but repartition 257^3 into varying
block counts (e.g. 16/32/64/128) and measure peak CPU RSS, GPU peak,
preparation time and per-forward runtime. For smaller partition
counts, explicitly raise the per-partition owned-node guard only
if the available Grace memory budget justifies it. Avoid any
generative-quality or spectrum claims.
