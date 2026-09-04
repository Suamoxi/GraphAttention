# M8 Sparse One-Hop Transformer

## Status

M8 introduces the first graph-aware learnable model in the repository: a transformer whose attention is evaluated only on explicitly supplied directed one-hop mesh edges.

Implementation and scientific-property validation are present. Target GPU performance evidence remains pending until the M8 benchmark is run on Calypso. Until then, M8 performance evidence is `ANALYTICAL` only.

## 1. Scope

M8 is deliberately narrow. It introduces exactly one architectural scientific change relative to the M5 null baseline:

> node states may exchange information through scaled dot-product attention on the supplied sparse one-hop graph topology.

M8 does **not** yet introduce:

- coordinate features;
- relative positions or distances;
- geometric attention bias;
- edge features;
- k-hop dilation;
- random or global edges;
- learned topology;
- graph pooling/hierarchy;
- periodic cross-boundary edge augmentation.

Those are separate geometry/model changes, beginning with M9 geometry-aware attention.

## 2. Scientific genealogy

The attention score follows scaled dot-product attention from:

- Vaswani et al., *Attention Is All You Need*, NeurIPS 2017, arXiv:1706.03762.

Restricting attention to graph neighborhoods follows the established graph-attention/message-passing idea that each node aggregates information only from explicitly connected neighbors. Graph Attention Networks provide relevant neighborhood-attention context:

- Veličković et al., *Graph Attention Networks*, ICLR 2018, arXiv:1710.10903.

The repository implementation is a **project adaptation**: it uses Transformer-style multi-head scaled dot-product scores, not the additive GAT scoring function, and evaluates them only on the supplied `edge_index` pairs.

## 3. Directed-edge convention

For

```text
edge_index[0, e] = j
edge_index[1, e] = i
```

edge `e` is interpreted as a directed information path

$$
j\rightarrow i.
$$

Node `i` forms the query and node `j` supplies the key and value.

For head `h` with head dimension `d_h`:

$$
q_i^{(h)} = W_Q^{(h)} h_i,
\qquad
k_j^{(h)} = W_K^{(h)} h_j,
\qquad
v_j^{(h)} = W_V^{(h)} h_j.
$$

The edge score is

$$
s_{ij}^{(h)} =
\frac{q_i^{(h)\mathsf T} k_j^{(h)}}{\sqrt{d_h}}.
$$

The normalization is over incoming supplied edges for target node `i`:

$$
\alpha_{ij}^{(h)} =
\frac{\exp(s_{ij}^{(h)})}
{\sum_{\ell:(\ell\rightarrow i)\in E}\exp(s_{i\ell}^{(h)})}.
$$

The head message is

$$
m_i^{(h)} =
\sum_{j:(j\rightarrow i)\in E}
\alpha_{ij}^{(h)}v_j^{(h)}.
$$

The multi-head output is the concatenated head message followed by a learned output projection.

## 4. Topology policy

M8 consumes the supplied `edge_index` exactly as model topology.

The model does not:

- add self-loops;
- remove self-loops if the geometry layer explicitly supplies them;
- deduplicate edges;
- symmetrize directed edges;
- create cross-sample edges.

The current hexahedral geometry transform already supplies each native physical mesh edge in both directions and removes duplicates. The model itself does not assume every future graph is symmetric.

No implicit self-loop is added because topology augmentation is a scientific/architectural decision. A node retains a local information path through the transformer residual connection. If a node has no incoming attention edges, its sparse-attention message is exactly zero before the residual addition.

If duplicate directed edges are supplied, they are treated as distinct attention entries and therefore change the normalization. Canonical edge construction/deduplication belongs to the geometry layer.

## 5. Transformer block

M8 uses a pre-normalization residual block:

$$
\tilde h = h + \operatorname{SparseMHA}(\operatorname{LN}(h), E),
$$

$$
h' = \tilde h +
\operatorname{MLP}(\operatorname{LN}(\tilde h)).
$$

The MLP is

$$
\operatorname{MLP}(x)
= W_2\operatorname{GELU}(W_1x+b_1)+b_2.
$$

The hidden MLP width is `mlp_ratio * hidden_dim`.

M8 intentionally has no dropout. This keeps the first sparse scientific reference deterministic and avoids introducing a second architectural/training change at the same milestone.

## 6. Inputs, outputs, and conditioning

The task still owns named field selection and statistical scaling. The sparse transformer receives only the already prepared model-facing tensors.

For node input `x_i` and optional graph-level conditioning `c_g`, the initial node representation is

$$
h_i^{(0)} = W_{\mathrm{in}}[x_i,c_{g(i)}]+b_{\mathrm{in}}.
$$

When no conditioning is configured, only `x_i` is projected.

After the configured transformer blocks:

$$
\hat y_i = W_{\mathrm{out}}\operatorname{LN}(h_i^{(L)}) + b_{\mathrm{out}}.
$$

Coordinates are deliberately not consumed in M8. Therefore M8 is topology-aware but not yet geometry-aware.

## 7. Sparse implementation

The implementation uses native PyTorch scatter/index reductions on edge tensors rather than a padded neighborhood tensor or dense `[N,N]` mask.

For `N` nodes, `E` supplied directed edges, hidden dimension `H`, and a fixed number of heads, the attention path is analytically linear in supplied edges rather than quadratic in node count:

```text
score/normalization storage: O(E * num_heads)
message materialization:     O(E * H)
node hidden storage:          O(N * H)
```

This is an analytical complexity statement, not a target-throughput claim.

The current implementation materializes per-edge value messages before `index_add_`. That is simple and transparent but may not be the optimal memory/kernel strategy. M8 target benchmarking is required before deciding whether a specialized scatter/sparse backend is justified.

## 8. Numerical softmax convention

Attention softmax is stabilized per target node and head by subtracting the maximum incoming score before exponentiation.

For FP16/BF16 projected query/key tensors, edge score multiplication, score reduction, exponentiation, and normalization are performed in FP32. The resulting attention weights are cast back to the value dtype before message multiplication.

This increases temporary score/normalizer precision and memory relative to fully low-precision reduction. CUDA mixed-precision performance and accuracy remain unvalidated until explicitly benchmarked.

Edge-list order may produce tiny backend-dependent floating-point differences because scatter accumulation order is not mathematically ordered. Scientific property tests therefore use numerical tolerances rather than exact equality.

## 9. Required scientific properties

M8 validation includes:

1. sparse attention matches an explicit small-neighborhood reference implementation;
2. reordering the edge list does not change the result beyond floating-point tolerance;
3. packed disconnected graphs match independent graph execution;
4. consistent node renumbering is equivariant;
5. empty-edge graphs remain finite;
6. the model integrates with the M6 equal-sample training step.

## 10. Configuration

The Hydra model configuration is:

```yaml
_target_: graph_attention.models.SparseGraphTransformer
in_channels: 2
out_channels: 1
hidden_dim: 64
num_heads: 4
num_layers: 2
mlp_ratio: 4
conditioning_channels: 0
```

This file is a small 2-D synthetic smoke configuration. It is not a recommended scientific model size.

The M8 benchmark executable uses an explicit reference architecture by default:

```text
hidden_dim = 128
num_heads  = 8
num_layers = 4
mlp_ratio  = 4
```

Those values define a benchmark workload only; they are not claimed to be optimal.

## 11. Benchmark gate

M8 adds `scripts/benchmark_m8.py`, using the M7 measurement protocol and the same synthetic S3 / real HIT workload classes.

Unlike M7, M8 forward/training measurements legitimately report edge throughput because the model consumes `edge_index` in every transformer layer.

Before any M8 target performance claim, collect at least:

- synthetic S3 single-GPU `TARGET_VALIDATED` result;
- real `HIT_LES_FORCED` single-GPU `TARGET_VALIDATED` result;
- exact model hyperparameters and parameter count;
- latency, node throughput, edge throughput, and CUDA allocator peaks.

Comparison with M7 is valid only as a comparison with the null-model framework floor, not as an assertion that both models perform equivalent computation.

## 12. Assumptions

M8 introduces or inherits these assumptions:

- `edge_index[0]` is source and `edge_index[1]` is target;
- packed graphs are disconnected, as guaranteed by M4 packing;
- canonical geometry construction is responsible for unwanted duplicate-edge removal;
- the supplied topology is scientifically intended for attention;
- no implicit self-loop is required because the residual path preserves local state;
- task input/output channel semantics remain explicit and compatible with the configured channel counts;
- global conditioning, when configured, has already passed M5 inference-availability/definition checks;
- model hidden dimension is divisible by the head count.

## 13. Handled edge cases

The implementation handles:

- variable node and edge counts;
- multiple disconnected graphs in one packed microbatch;
- directed or symmetric edge lists;
- nodes with no incoming edges;
- globally empty edge lists;
- optional graph conditioning;
- invalid edge shape, dtype, device, and node indices;
- invalid model dimensions/head divisibility;
- edge-list reordering within numerical tolerance.

## 14. Deferred/unvalidated edge cases

M8 deliberately defers:

- geometric/positional attention information;
- periodic cross-boundary edge augmentation for HIT;
- duplicate-edge canonicalization inside the model;
- dropout/stochastic depth;
- specialized fused sparse-attention kernels;
- graph partitioning for one graph exceeding device budget;
- CUDA BF16/FP16 target validation;
- multi-node sparse-transformer scaling;
- deterministic bitwise CUDA scatter reductions.

## 15. Failure behavior

The model fails explicitly when:

- input channel count does not match configuration;
- model/input dtype or device is incompatible;
- `edge_index` is not `[2,E]` `torch.long` on the input device;
- an edge references a node outside the packed node range;
- configured dimensions/head counts are invalid;
- required conditioning tensors are absent or incompatible.

It does not silently add edges, alter topology, cast model/input dtypes, or change hidden dimensions.

## 16. Efficiency, memory, and scientific trade-offs

The native-PyTorch implementation minimizes dependencies and gives a transparent scientific reference, but it materializes `O(E * hidden_dim)` messages. A fused/specialized backend may later reduce memory traffic or kernel-launch overhead.

No backend replacement should be made on advertisement or asymptotic arguments alone. The M8 reference must first be target-benchmarked, after which a specialized implementation can be compared under identical scientific equations and workloads.

The deliberate absence of geometry makes M8 scientifically incomplete as a final CFD attention model, but it isolates the effect of sparse one-hop attention. This follows the repository rule that scientifically meaningful architecture changes should normally be introduced and ablated one at a time.
