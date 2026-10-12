# M41.2 — owned-query halo attention (single GPU, inference-only)

Status: **TARGET_VALIDATED for numerical equivalence and measured on GH200**,
Calypso Slurm job `461158` (2026-10-12, code commit `f3d9e5bf`).
Performance gains versus M41.1 are measured; the high-partition latency
remains above M37 and memory use is greater than M41.1.
This is a separate executor and opt-in benchmark, **not a replacement** for
the validated M41.1 `forward_partitioned_dinat` reference.
Neither the M37 training path, the original M37 model, checkpoints nor
the M41.1 default benchmark mode is changed. M42 is out of scope.

## Why optimize this?

On the 35,937-node HIT graph, M41.1 completed the 50-step Heun parity gate.
However 16 partitions gave **232.88 ms** per forward, versus **31.50 ms**
for original M37, with a total per-partition read-node ratio of only 1.632.
This indicates that redundant halo computations, repeated small linear layers
and kernel launch overhead deserve attention. Precise kernel attribution
remains a profiling task.

M41.1 evaluates each full DiT block over owned **and** halo states, writes
only owned outputs and discards intermediate halo-node outputs. In M41.2,
the model evaluates every strictly nodewise operation *once globally* on the
resident hidden tensor. Only the **attention reduction** is partitioned.
This is possible because M41.1 explicitly allows a full hidden field on GPU.

## Numerical contract

For layer input `h`, M37 computes the global adaLN modulation, pre-normalized
features, and joint `Q,K,V` projection. M41.2 computes these with the
**same original M37 modules** and whole-graph matrix shapes, once per layer:

```text
z = modulate(norm1(h), shift_attn, scale_attn)
Q, K, V = original_attention.qkv(z)
```

Partition `p` has owner nodes `O_p`, the union of original local and
exact-two-hop incoming source halos `R_p`, and a cached rectangular CSR
of shape `|O_p| x |R_p|`. Each incoming graph edge `(j -> i)` maps to
`row = index_of(i, O_p)`, `col = index_of(j, R_p)`.

The local attention score is the unchanged geometric M37 score:

```text
score_ij = (Q_i . K_j) / sqrt(head_dim) + geometry_mlp(coords_j - coords_i)
alpha_ij = exp(score_ij) / sum_{k in incoming(i)} exp(score_ik)
context_i = sum_{j in incoming(i)} alpha_ij V_j
```

All original incoming neighbors, including cross-partition exact-two-hop
neighbors, enter the same **per-target softmax denominator**. Empty incoming
rows generate zero context. Original local/dilated self-loop policies remain
unchanged and no synthetic periodic edges are inserted. CSR sorting reorders
geometric score biases by the identical `edge_order`.

The complete context tensor is assembled by owner index. The original M37
attention output projection, gated residual, `norm2` and gated MLP operate
**once globally** after all partition contexts have been written. This
restores the same per-layer synchronization as M41.1 and avoids the repeated
halo MLP and repeated small GEMMs.

The original global `centered_bbox` normalized coordinates are reused for
coordinate embeddings; geometric edge biases use the original raw relative
displacements, exactly as in M37. Trained weights and Flow Matching
conditioning are unchanged. As in M41.1, minor FP32 differences due to
reduction grouping and sparse kernels are allowed only if they meet
predeclared tolerances.

## Implementation

- `src/graph_attention/models/owned_query_dinat.py` defines the rectangular
  `OwnedQueryTopology`, an immutable `OwnedQueryPlan`, its plan builder and
  `forward_owned_query_dinat()`.
- The builder reuses M41.1 ownership and original topologies, then drops
  unused *square* full-block CSR caches from its returned plan.
- It caches rectangular CSR patterns and their edge permutation. No native
  M37 attention or model API is modified. The plan is immutable and must be
  rebuilt after a geometry/topology/dtype/device change. No model mutation is
  permitted during inference.
- The original `scripts/benchmark_m41_partitioned.py` now accepts
  `--executor layerwise_full` (default, unmodified M41.1 behavior) or
  `--executor owned_query` (new M41.2 behavior). Both use the same strict
  M37 checkpoint, input graph, conditioning, reference hooks and original
  50-step task sampler.
- `scripts/summarize_m41_2.py` only compares matched checkpoint artifacts
  and test samples from paired full/owned runs, refusing incomplete or
  incompatible comparisons.
- `tests/unit/test_owned_query_dinat.py` covers all self-loop policies,
  arbitrary ownership, directed/empty input rows, intermediate-layer parity,
  plan creation under inference mode, no mutation of learned weights,
  50-step Heun parity and CUDA `torch_sparse` parity. It explicitly asserts
  that `norm1`, `qkv`, `out_proj`, `norm2` and `mlp` run exactly once
  per global graph per transformer layer.

Only the existing `torch_sparse` M37 backend is supported initially.
The CPU PyTorch CSR implementation can run small regression tests. The
optimized path rejects training-mode models; backward/multi-GPU/out-of-core
execution are not implemented.

## GH200 A/B benchmark

On Calypso:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_2_owned_query_m37.slurm
```

Defaults: M37 checkpoint from
`/scratch/coop/theret/GraphAttention_runs/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16`,
single GH200, FP32 with TF32 disabled, full 35,937-node held-out graph,
50-step Heun with seed 5678 (100 model evaluations), two warmups and five
repetitions per mode/layout. Requested spatial layouts: `1x1x1`, `2x1x1`,
`2x2x1`, `2x2x2`, `4x2x2`.

Optional run controls:

```bash
export M41_2_BLOCKS="1 1 1;2 2 2;4 2 2"
export M41_2_REPEATS=5
export M41_2_WARMUP=2
sbatch scripts/slurm/m41_2_owned_query_m37.slurm
```

If the test gate fails and you explicitly need diagnostic reports from
additional layouts, set `M41_2_CONTINUE_AFTER_FAILURE=1`. The job then
attempts remaining cases but still exits **nonzero**. A failed numerical
gate must never be interpreted as a performance success.

Results are written under:

```text
/scratch/coop/theret/GraphAttention_runs/m41_2_owned_query_m37/<job-id>/
    git_commit.txt
    gpu_info.csv
    full_blocks_1x1x1.json
    owned_blocks_1x1x1.json
    ...
    full_blocks_4x2x2.json
    owned_blocks_4x2x2.json
    comparison.csv
```

Per-case `.log` files contain complete validation diagnostics; the main
Slurm stdout remains compact. Reports record checksum-provenanced M37
weights, sample identity, original edge counts, layerwise errors, 100 Heun
calls and identical initial noise, median forward/Heun latency, and peak
allocated CUDA memory. Every comparison must satisfy
`|owned - M37| <= 2e-4 + 2e-4*|M37|` for each component.
The summary also computes M41.1/M41.2 forward and Heun speedup, retaining
both memory totals and plan allocations.

The benchmark runs M41.1 and M41.2 in **separate processes** for each
layout to isolate persistent topology. Timing and peak-memory comparisons
exclude plan construction and dataset loading, but include per-forward halo
gathers. Actual peak memory may rise relative to M41.1 because full-global
QKV and MLP activations are retained. Smaller attention aggregation alone
does not guarantee smaller overall memory.

## Acceptance and next optimization

The M41.2 experiment is accepted as *numerically valid* only if all
selected tests and real-checkpoint layouts pass, every intermediate layer
and the final Heun state meet the original tolerances, and original M41.1
still passes. Accept a *performance improvement* only if the paired medians
show positive speedup without unacceptable memory growth. The results in
the next section report the measured speed-memory trade-off, not an
unqualified optimization win.

The next step is to profile the remaining sampled-CSR heads,
per-partition gather/softmax and aggregation kernels. Candidate later
optimizations include caching geometry-only biases while weights are frozen,
batching compatible partitions, and a fused rectangular sparse attention
kernel. The current M41.2 implementation is deliberately preserved
as the validated reference during this profiling stage.

## GH200 measured results — job 461158 (2026-10-12)

The Calypso run used clean Git commit
`f3d9e5bfcf174858698cdaf5f9f1deabc901ada8`, NVIDIA GH200 480GB,
single GPU, FP32 with TF32 disabled, two warmup steps and five
timed repetitions per configuration. Checkpoint
`best.pt` SHA-256:
`01cf4dd6e6199d8b086d5737e6399059f520f25e386c02f6ab4eb4e7cc5764e8`.
All ten reports used held-out sample `solut_hit_00007500`, with 35,937
nodes, 209,088 directed one-hop edges, and 608,058 directed exact-two-hop
edges. The selected regression gate completed **68 passed / one skipped**
(optional DGL/torchdata backend unavailable). This selected gate was
not a full repository-wide test run.

Every M41.1 and M41.2 layout has `passed=true` and
`evidence=TARGET_VALIDATED`. Every edge topology check passed. Across
the five M41.2 layouts the maximum absolute error versus unpartitioned M37
was `5.888939e-5` for intermediate layers, `4.443526e-5` for the
final network output, and `1.907349e-6` for complete Heun generation.
All individual component comparisons have zero violations under
`atol=rtol=2e-4`; initial Gaussian noise is bitwise identical, and
both models perform 100 model evaluations with 50 Heun steps.

### Timings and memory

All numbers are **measured medians** over five repetitions, without plan
construction or data transfer. Each executor was run in a separate process.
Latency comparisons use the original M37 time measured alongside the
corresponding executor, not the reference timing from another job.

| Spatial blocks | Partitions | M41.1 forward (ms) | M41.2 forward (ms) | M41.2 speedup vs M41.1 | M41.2 / M37 forward | M41.1 Heun (s) | M41.2 Heun (s) | M41.1 peak GiB | M41.2 peak GiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1x1x1 | 1 | 28.96 | 23.09 | 1.254x | 0.733x | 2.501 | 2.248 | 0.433 | 0.485 |
| 2x1x1 | 2 | 42.96 | 31.89 | 1.347x | 1.014x | 3.639 | 2.518 | 0.353 | 0.414 |
| 2x2x1 | 4 | 69.91 | 44.55 | 1.569x | 1.412x | 6.106 | 3.567 | 0.310 | 0.398 |
| 2x2x2 | 8 | 122.94 | 73.26 | 1.678x | 2.334x | 11.675 | 6.419 | 0.251 | 0.383 |
| 4x2x2 | 16 | 238.88 | 126.44 | 1.889x | 4.110x | 23.518 | 11.622 | 0.216 | 0.377 |

Original M37's reference allocated-memory peak is approximately
**0.440 GiB** in each per-layout run. As intended, M41.2 is faster
than M41.1 at every tested partition count. With 16 partitions,
forward speedup is 1.889x, generation speedup is 2.024x, but M41.2
is still 4.11x slower than unpartitioned M37. Total per-partition
read-node ratio at 16 partitions is 1.632.

The improvement **comes with a memory cost**. At 16 partitions,
M41.2 peaks at 0.377 GiB versus M41.1 at 0.216 GiB (+74%),
although it remains about 14% below unpartitioned M37. At one
partition, M41.2's 0.485 GiB peak exceeds M37's 0.440 GiB.
These are `torch.cuda.max_memory_allocated` measurements, not total
device usage or an out-of-core upper bound.

**Interpretation:** for this 35,937-node problem, one or two
M41.2 partitions are preferable when latency matters.
Two partitions have approximately the same forward time as M37
(31.89 versus 31.46 ms) with a 0.414 GiB allocation peak.
Finer partitions may be useful only when larger meshes or hard
working-memory constraints justify their extra runtime.
Potential costs from sampled CSR kernel launches, halo gathers,
geometry-bias MLP, row-wise normalization and index-add reduction
are **hypotheses** until operator-level profiling measures them.

### Opt-in operator-level profiling

An additional, non-default profiler now instruments **one forward per
executor after all unprofiled median timings have completed**. It
captures PyTorch CPU/CUDA operator names, call counts, self GPU and CPU
times, and memory activity. No CUDA trace is exported by default.

For a targeted experiment, profile the one-partition and 16-partition
cases first; this avoids another full five-layout run:

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch --export=ALL,M41_2_BLOCKS="1 1 1;4 2 2",M41_2_REPEATS=2,M41_2_PROFILE=1 \
    scripts/slurm/m41_2_owned_query_m37.slurm
```

Set `M41_2_TRACE=1` as well to export per-forward Chrome traces, which
may consume substantial scratch space. Profiles are saved as
`profile_full_...` and `profile_owned_...` directories with
`original_operations.json` and either
`layerwise_full_operations.json` or `owned_query_operations.json`.
The corresponding benchmark JSON includes a `profiles` summary with the
top 30 GPU operators. Profiled GPU operation sums are *not* a substitute
for unprofiled wall-clock medians.

This profiling is an **unvalidated additional instrumentation path**.
The numerical implementation measured in job `461158` has not
changed. Further numerical or performance claims require the
new Calypso profiler run results.

## GH200 CPU/CUDA operator profile — job 461247 (2026-10-12)

Evidence: clean main commit `eb1935f2ca798c3ca48c0301c8bb169c72ead9a0`,
NVIDIA GH200 480GB, FP32/TF32 disabled, the identical M37 checkpoint and
35,937-node test graph as job `461158`. Calypso path:
`/scratch/coop/theret/GraphAttention_runs/m41_2_owned_query_m37/461247/`.
The job ran 68 selected tests successfully with one optional DGL test
skipped; both `1x1x1` and `4x2x2` executors passed all checkpoint,
intermediate-layer, and complete 50-step Heun numerical gates. Unprofiled
timings used two warmups and **two** timed repeats. Profiling captured one
additional forward per executor after those timings. No Chrome trace was
requested.

| Blocks | M37 reference forward (ms) | M41.1 forward (ms) | M41.2 forward (ms) | M41.2/M41.1 speedup | M41.1 Heun (s) | M41.2 Heun (s) |
|---|---:|---:|---:|---:|---:|---:|
| 1x1x1 | 42.82 | 44.96 | 41.19 | 1.092x | 2.825 | 2.313 |
| 4x2x2 | 42.16 | 307.82 | 199.56 | 1.542x | 26.647 | 15.063 |

Both M37 and partitioned forwards were slower than in job `461158`.
The M37 reference rose from approximately 31 ms to 42 ms. These
cross-job differences may be hardware frequency or runtime variability;
this profile alone does not isolate their cause and is **not** evidence of
a regression caused by the opt-in profiler instrumentation. Within-run
numerical checks and speedup ratios remain valid.

### Operator counts on a single forward

| Operation | Original M37 | M41.2 1 partition | M41.2 16 partitions |
|---|---:|---:|---:|
| `aten::sparse_sampled_addmm` | 40 | 40 | 640 |
| `aten::scatter_reduce_` | 12 | 10 | 160 |
| `aten::index_select` | 3 | 20 | 320 |
| `aten::copy_` | 250 | 241 | 3,841 |
| `aten::addmm` | 57 | 57 | 207 |

The 640 sampled CSR matrix multiplications correspond to
`10 layers x 16 partitions x 4 heads`, rather than 40 for the original
whole-graph M37. The `addmm` difference (207 versus 57) is consistent
with 150 extra per-partition geometry-MLP evaluations, one for each
additional partition per layer.

The 16-partition M41.2 profile records, among other PyTorch
`aten::` operator *self CPU* times: `empty` 19.65 ms (6,512 calls),
`copy_` 17.79 ms (3,841 calls), `sparse_sampled_addmm` 12.96 ms
(640 calls), `index` 7.98 ms (640 calls), and `addmm` 6.83 ms
(207 calls). The sampled-addmm CUDA execution record totals only
approximately 2.82 ms. Recorded `aten::` self CPU time summed across
all `aten::` operators is ~135.1 ms for M41.2 and ~197.3 ms for
M41.1; separately recorded raw CUDA kernel activity sums to ~43.6 ms
and ~62.8 ms, respectively. **Do not add CPU and CUDA totals**: they are
overlapping timing domains, and the profiler also changes execution.

These counts and self-times indicate many small dispatched operations,
temporary allocations, and per-partition sparse launches. They do not
prove that a particular future fusion or caching implementation will
produce a speedup. The original whole-graph M37 forward remains much
faster than either 16-partition implementation on this mesh.

### Next performance experiment (not yet implemented or validated)

Prioritize **opt-in CUDA Graph capture/replay** of the existing
owned-query executor for a fixed graph and fixed tensor shapes. The
single-GPU flow-matching Heun solver invokes the same graph 100 times,
so replay could reduce host-side launch overhead without modifying
attention, self-loop topology, learned weights, or time-conditioning
mathematics. Inputs and conditioning must be copied into stable replay
buffers each evaluation. Numerical checks must include all layers,
the identical Heun noise and output, and memory retained by the
captured graph.

**Unknown until a GH200 experiment:** whether this particular cuSPARSE
sampled-CSR path and dynamic allocations can be captured safely, how much
GPU memory replay buffers/pools consume, and whether end-to-end Heun
improves. Preserve M41.2 as the unmodified numerical reference and
fail explicitly on unsupported capture, rather than silently switching
executors. A fallback experimental avenue is reducing per-partition CSR
kernel calls (e.g., head batching or a purpose-built fused sparse
attention kernel). Caching static geometry MLP outputs is a simpler,
lower-risk change, but does not address the 640 sampled-matmul calls.
