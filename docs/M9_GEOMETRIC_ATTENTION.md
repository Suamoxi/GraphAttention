# M9 Relative-Displacement Geometric Attention

## Status

M9 introduces the first geometry-aware attention mechanism in the repository. It extends the frozen M8 sparse one-hop Transformer by adding a learned per-head attention-score bias derived only from relative edge displacement.

The scientific/software gate is complete and the frozen FP32 reference is target-performance-validated on one NVIDIA GH200 480GB. Slurm job `403939` ran from clean SHA `83e160846badb151eef0a09c2f1e2234da22bc24` with Python 3.12.3, PyTorch `2.7.0a0+7c8ec84dab.nv25.03`, CUDA 12.8, 10 warmups, and 50 measured repetitions.

## 1. Scientific change from M8

M8 attention on a supplied directed edge `j -> i` uses

$$
s_{ij}^{(h)}=
\frac{q_i^{(h)\mathsf T}k_j^{(h)}}{\sqrt{d_h}}.
$$

M9 changes exactly one scientific mechanism:

$$
\boxed{
s_{ij}^{(h)}=
\frac{q_i^{(h)\mathsf T}k_j^{(h)}}{\sqrt{d_h}}
+b_{ij}^{(h)}
}
$$

with

$$
\boxed{
b_{ij}=\phi_\theta(\Delta r_{ij})
}
$$

and

$$
\boxed{
\Delta r_{ij}=r_j-r_i
}
$$

for source node `j` and target/query node `i`.

M9 does not add explicit edge distance, normalized direction, absolute coordinates, self-loops, new edges, k-hop connectivity, random/global edges, or geometry-conditioned values/messages.

## 2. Why relative displacement only

The M9 geometry feature is the complete displacement vector:

$$
\Delta r_{ij}=(\Delta x,\Delta y,\Delta z)
$$

in 3-D, or the corresponding 2-D vector.

Euclidean distance is mathematically recoverable from the displacement:

$$
\|\Delta r_{ij}\|=
\sqrt{\Delta x^2+\Delta y^2+\Delta z^2}.
$$

M9 therefore does not duplicate distance as an input feature. Adding explicit distance later would be a separate inductive-bias ablation rather than part of the baseline geometric mechanism.

For physically nondimensionalized tasks, `NodeRegressionTask` already produces coordinates scaled by the case length reference:

$$
r^*=\frac{r}{L_{\mathrm{ref}}}.
$$

The M9 displacement is therefore

$$
\Delta r^*_{ij}=
\frac{r_j-r_i}{L_{\mathrm{ref}}}.
$$

No additional coordinate standardizer is introduced.

## 3. Translation invariance

For a global translation `a`,

$$
r'_i=r_i+a,
\qquad
r'_j=r_j+a,
$$

so

$$
r'_j-r'_i=r_j-r_i.
$$

The M9 geometric signal is therefore translation invariant. With unchanged physical node features and topology, the complete M9 model is expected to produce the same prediction up to floating-point tolerance after a global coordinate translation.

M9 does not claim rotation, reflection, or scale invariance/equivariance. The Cartesian displacement components transform under those operations, and the geometry MLP is an ordinary learned MLP rather than an equivariant tensor construction.

## 4. Geometry ownership

Relative displacement is a deterministic geometry transform implemented in:

```text
src/graph_attention/geometry/relative.py
```

For

```text
edge_index[0, e] = j
edge_index[1, e] = i
```

the geometry layer returns

```text
coords[j] - coords[i]
```

for edge `e`.

The model consumes this geometry through an explicit coordinate/topology interface. Geometry construction does not know the learning objective or optimizer.

## 5. Learned geometric bias

Each M9 attention layer has its own small geometry MLP:

```text
spatial_dim
    -> num_heads
    -> GELU
    -> num_heads
```

The final linear map has no bias because a head-wise constant added identically to every incoming edge cancels under the target-wise softmax and is therefore unidentifiable.

The geometry MLP hidden width equals `num_heads`. This is intentional for the first M9 reference: it avoids materializing a second `O(E * hidden_dim)` geometric activation while still allowing a nonlinear function of the displacement vector.

The resulting vector

$$
b_{ij}\in\mathbb R^{H_{\mathrm{heads}}}
$$

adds one learned geometric scalar to each attention head before softmax.

## 6. Message/value path

M9 changes only the attention logits. The value/message path remains the M8 path:

$$
m_i^{(h)}=
\sum_{j:(j\rightarrow i)\in E}
\alpha_{ij}^{(h)}v_j^{(h)}.
$$

Relative displacement is not concatenated to values and does not directly alter the MLP/residual branches. This isolates geometric neighbor weighting from a simultaneous geometric message-content change.

## 7. Topology policy

M9 uses exactly the supplied M8 topology.

The current hexahedral CFD geometry transform already:

- extracts native physical hexahedral node edges;
- canonicalizes physical pairs;
- removes duplicates shared by adjacent cells;
- represents each physical adjacency in both attention directions;
- introduces no self-loops.

M9 does not alter that topology.

Periodic cross-boundary HIT edges remain deferred, so the current real HIT topology is still incomplete with respect to periodic physical adjacency.

## 8. Scientific genealogy

M9 is a **project adaptation** combining established ideas:

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, arXiv:1706.03762 — scaled dot-product attention;
- Pfaff et al., *Learning Mesh-Based Simulation with Graph Networks*, ICLR 2021, arXiv:2010.03409 — relative mesh-position information as graph-edge geometry in learned mesh simulation;
- relative positional information in local attention is also established in point/vision Transformer literature.

The exact repository equation

$$
q_i^T k_j/\sqrt{d_h}+\phi_\theta(r_j-r_i)
$$

is a project-specific adaptation and is not attributed verbatim to one of those papers.

## 9. Required scientific properties

M9 validation includes:

1. deterministic relative-displacement construction with the frozen source/target sign convention;
2. reverse directed edges produce opposite displacement vectors;
3. global translation leaves relative displacement and model output unchanged within tolerance;
4. geometric sparse attention matches an explicit small-neighborhood reference;
5. changing relative geometry can change the prediction;
6. disconnected packed execution matches independent graph execution;
7. consistent node renumbering remains equivariant;
8. the model integrates with the M6 equal-sample training step;
9. invalid coordinate shapes, dtypes, indices, NaN, and Inf fail explicitly.

The full repository software gate reached 165 passing tests before the final formatting-only cleanup, and the exact-runtime M9 benchmark test subset also passed inside job `403939`.

## 10. Configuration

The synthetic smoke configuration is:

```yaml
_target_: graph_attention.models.GeometricSparseGraphTransformer
in_channels: 2
out_channels: 1
hidden_dim: 64
num_heads: 4
num_layers: 2
spatial_dim: 2
mlp_ratio: 4
conditioning_channels: 0
```

`spatial_dim` must match the coordinate dimension supplied at runtime.

For target performance comparison with M8, the benchmark defaults remain:

```text
hidden_dim = 128
num_heads  = 8
num_layers = 4
mlp_ratio  = 4
```

so the benchmark comparison isolates the added relative-displacement bias rather than changing the backbone width/depth.

## 11. Target-validated benchmark results

`scripts/benchmark_m9.py` uses the same benchmark protocol and workload construction as M8. Job `403939` collected both required `TARGET_VALIDATED` workloads on one NVIDIA GH200 480GB in FP32.

Synthetic S3:

- 4 graphs, 32,768 nodes, 65,530 directed edges;
- 793,857 parameters;
- forward median `3.413842 ms`;
- training-iteration median `90.945784 ms`;
- forward incremental CUDA allocation `229,376,000 B`;
- training incremental CUDA allocation `1,586,070,016 B`.

Real `HIT_LES_FORCED`:

- 1 graph, 35,937 nodes, 209,088 directed edges;
- 794,629 parameters;
- forward median `5.798864 ms`;
- training-iteration median `18.704375 ms`;
- forward incremental CUDA allocation `553,855,488 B`;
- training incremental CUDA allocation `2,978,613,248 B`.

Against the frozen M8 real-HIT reference, the M9 relative-geometry path adds about 8.6% forward latency, 5.5% training-iteration latency, and 1.8% incremental training allocation. The four-layer reference adds only 384 parameters. The synthetic high-degree stress workload remains dominated by the inherited M8 scatter/reduction backward pathology; M9 does not introduce a new observed pathology.

These measurements establish implementation cost only. They do not show that M9 improves predictive quality; that learning question is the purpose of the subsequent M10 ablation.

## 12. Assumptions

M9 introduces or inherits these assumptions:

- `edge_index[0]` is source and `edge_index[1]` is target;
- relative displacement is defined as source minus target coordinates;
- coordinates and physical inputs are expressed in one consistent Cartesian frame;
- when physical nondimensionalization is enabled, coordinates have already been divided by the case `L_ref`;
- supplied topology is scientifically intended and already canonicalized by geometry construction;
- no implicit self-loop is required because the residual path preserves local state;
- a small per-layer geometry MLP is sufficient as the first nonlinear displacement encoder;
- M6 task scaling, sample weighting, and optimizer semantics remain unchanged.

## 13. Handled edge cases

Handled cases include:

- 2-D and 3-D coordinates through explicit `spatial_dim`;
- variable node/edge counts;
- multiple disconnected packed graphs;
- directed or symmetric edge lists;
- empty edge lists;
- nodes with no incoming edges;
- reverse edge directions;
- global translation;
- optional graph conditioning;
- coordinate/index validation and non-finite coordinate rejection.

## 14. Deferred or unsupported cases

M9 deliberately defers:

- explicit distance as a redundant/inductive-bias feature;
- normalized direction vectors;
- absolute-position features;
- rotation/reflection-equivariant geometric representations;
- geometry-conditioned value/message vectors;
- local mesh-scale or metric-tensor features;
- periodic cross-boundary HIT augmentation;
- fused specialized sparse-geometric kernels;
- CUDA BF16/FP16 target validation;
- multi-node performance scaling.

## 15. Failure behavior

The model fails rather than silently correcting inputs when:

- coordinate count does not match node count;
- coordinate dimension does not match `spatial_dim`;
- coordinate/input dtype or device differs;
- coordinates contain NaN or Inf;
- `edge_index` is malformed, has the wrong dtype/device, or references an invalid node;
- model dimensions/head divisibility are invalid.

No automatic coordinate casting, centering, normalization, edge augmentation, or topology repair is performed.

## 16. Efficiency, memory, and numerical trade-offs

M9 computes relative displacement once per model forward and reuses it across all Transformer layers. Each layer applies its own small geometry MLP and materializes a per-edge/per-head bias.

The geometric addition therefore introduces roughly `O(E * num_heads)` score/bias activations, rather than another `O(E * hidden_dim)` geometry feature tensor. The existing M8 message path still materializes `O(E * hidden_dim)` values/messages and remains the dominant sparse activation class.

The current implementation is a transparent native-PyTorch scientific reference, not a claim of kernel optimality. M8 target validation already showed strong sensitivity of backward scatter/reduction performance to extreme degree concentration; M9 inherits that limitation because it retains the same sparse aggregation backend.

For FP16/BF16 attention projections, M8's FP32 score/softmax reduction policy remains active. The learned geometric bias is promoted to the score dtype before addition.
