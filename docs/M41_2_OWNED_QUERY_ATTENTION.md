# M41.2 — owned-query halo attention (single GPU, inference-only)

Status: **implemented, numerical/performance validation pending on GH200**.
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
show positive speedup without unacceptable memory growth. No such measurement
is asserted before Calypso execution.

A useful next iteration, **after obtaining the A/B results**, is profiling
the remaining sampled-CSR heads, per-partition gather/softmax and
aggregation kernels. Candidate later optimizations include caching
geometry-only biases while weights are frozen, batching compatible
partitions, and a fused rectangular sparse attention kernel. None is
introduced in M41.2's first implementation.
