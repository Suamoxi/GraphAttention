# M4 Packed Variable-Graph Batching

## Status

M4 is complete for its frozen packed-graph and computational-budget scope from `docs/ARCHITECTURE.md` and `docs/NUMERICAL_CONVENTIONS.md`.

The merged implementation was target-validated on Calypso on 2026-09-03. The repository test suite reported 81 passing tests, `ruff check .` passed, `python scripts/inspect_config.py` passed, and the only initial `ruff format --check .` findings were formatter-only changes in three files. Those formatter changes were merged in PR #19, after which the final formatter check passed on Calypso.

M4 is deliberately limited to computational representation and grouping. It does not define task channel order, training loss weighting, gradient accumulation, DDP behavior, or a model architecture.

## 1. Purpose

Variable mesh size is a core repository requirement. A computational microbatch therefore represents several physical samples as one disconnected sparse graph rather than padding them to a common node count or neighborhood size.

For samples `g = 0, ..., B-1`:

$$
N_{\mathrm{total}} = \sum_g N_g,
\qquad
E_{\mathrm{total}} = \sum_g E_g.
$$

M4 materializes:

```text
coords       [N_total, D]
edge_index   [2, E_total]
batch_index  [N_total]
ptr          [B + 1]
```

Selected node fields are concatenated along their leading node dimension while remaining explicitly named.

## 2. Why M4 does not create one anonymous `X`

The data layer reports named fields; the later task layer owns ordered input/target selection. M4 must not prematurely decide that order or silently concatenate scientifically different quantities.

`pack_samples(..., node_field_names=...)` therefore requires the caller to explicitly identify which supplied tensors are node-supported fields. The returned `PackedBatch.fields` mapping keeps each field named:

```text
rho       -> [N_total]
momentum  -> [N_total, D]
...
```

M5 may later select an ordered subset and form model state such as `X [N_total, C]`. Keeping the M4 representation named preserves field semantics and avoids moving task ownership into the packer.

The packer never infers node support from a filename, field name, tensor range, or scientific convention. It only verifies that a field explicitly supplied through `node_field_names` has leading dimension `N` in every sample.

## 3. Disconnected-union construction

Samples are concatenated in the supplied order. For graph `g`, its local edge indices are shifted by the cumulative number of preceding nodes.

If

$$
o_g = \sum_{j<g} N_j,
$$

then each local edge `(i, j)` becomes `(i + o_g, j + o_g)`.

No cross-sample edges are introduced.

`ptr` records graph boundaries:

$$
\mathrm{ptr}[0]=0,
\qquad
\mathrm{ptr}[g+1]=\mathrm{ptr}[g]+N_g.
$$

`batch_index[i] = g` identifies the physical sample owning packed node `i`.

## 4. Preserved per-graph semantics

`PackedBatch` retains ordered per-graph tuples for:

- `sample_id`;
- `mesh_id`;
- `case_id`;
- `ReferenceScales`;
- `RegimeParameters`;
- sample metadata;
- mesh metadata.

Reference and regime objects are not converted into anonymous tensors in M4. Selecting model-visible global conditioning remains a task/model-interface decision.

## 5. Node weights

If every graph supplies `node_weights [N]`, M4 concatenates them to `[N_total]` without renormalizing them.

If no graph supplies node weights, the packed value is `None`.

A microbatch that mixes graphs with and without node weights fails explicitly. M4 does not invent fallback weights because that would silently choose a statistical/spatial loss policy before the task and loss contracts exist.

## 6. Compatibility requirements

One packed microbatch requires:

- non-empty graphs;
- one coordinate dimension `D`;
- one coordinate dtype;
- one device for coordinates, connectivity, selected node fields, and node weights;
- for each selected named field, one dtype and one trailing shape across graphs;
- either node weights on all graphs or on none.

Scalar node fields such as `[N]` and vector/tensor node fields such as `[N, C]` remain in their native named shapes. Only the leading node dimension is concatenated.

M4 deliberately does not cast or promote dtypes during packing.

## 7. Computational budgets

`MicrobatchBudget` defines positive integer limits:

```text
max_nodes
max_edges
```

`partition_samples_by_budget` consumes an already selected sample sequence and partitions it into contiguous groups satisfying:

$$
N_{\mathrm{total}} \le N_{\max},
\qquad
E_{\mathrm{total}} \le E_{\max}.
$$

The function does not shuffle, sample, bucket, duplicate, or drop samples. The input order is preserved exactly.

A single graph that exceeds either limit fails before packing with a diagnostic containing its sample ID, node/edge counts, and configured limits.

Concrete budget values are hardware/model-specific and are intentionally not repository constants.

## 8. Sampler and packer remain separate

M4 does not implement a statistical sampler. The caller decides which physical samples are selected and in what order. The budget partitioner only decides where computational microbatch boundaries fall in that already selected sequence.

This separation is required so future size bucketing or hardware packing does not silently alter sample probabilities or create a resolution-dependent curriculum.

Integration with optimizer-level statistical batches, gradient accumulation, and DDP belongs to M6.

## 9. Geometry/topology assumption

The edge budget counts the `Mesh.edge_index` that exists at the time `partition_samples_by_budget` is called.

Therefore all sparse topology that the model will actually consume for that microbatch must be constructed before the budget is evaluated. Budgeting a raw AVBP mesh whose `edge_index` has not yet been constructed would undercount `E_total` and is not a valid performance contract.

`cell_connectivity` is not packed into the model-facing disconnected node graph. Cell connectivity remains source/geometry information and should be converted into the required node topology or geometric descriptors before ordinary graph packing when needed.

Future graph partitioning for a single sample that itself exceeds device capacity is explicitly outside ordinary M4 batching.

## 10. Tests

`tests/unit/test_packed_batching.py` covers:

- variable node and edge counts in one packed microbatch;
- exact `ptr` and `batch_index` construction;
- edge-index offsetting;
- absence of cross-sample edges;
- named scalar and vector field concatenation;
- node-weight preservation;
- per-graph semantic metadata preservation;
- node- and edge-limited budget partitioning;
- exact-budget boundary behavior;
- oversized-graph diagnostics;
- invalid budget rejection;
- empty-selection behavior for partitioning;
- explicit failures for missing fields, incompatible field shapes, mixed node-weight availability, incompatible coordinate dimensions, and empty graphs.

A packed-vs-independent **model-output** equivalence test is deferred until M5 introduces a model/task baseline. There is currently no model whose output could be compared without inventing M5 prematurely.

## 11. Efficiency and evidence status

Packing performs linear concatenation/index-offset work in the amount of supplied data:

$$
O(N_{\mathrm{total}} + E_{\mathrm{total}})
$$

plus concatenation of explicitly selected node fields. It does not materialize padded `[B, N_max, ...]` tensors.

This complexity statement is `ANALYTICAL`. M4 does not make a throughput, latency, or peak-memory performance claim. Hardware benchmarking belongs to M7 after the training/model path exists.

## 12. Assumptions and deferred edge cases

### Assumptions introduced or inherited

- A computational graph contains at least one node.
- Samples packed together use compatible coordinate dimensions, dtypes, devices, and selected field layouts.
- `Mesh.edge_index` represents the actual sparse topology whose edge count should constrain the microbatch.
- Caller-supplied `node_field_names` are semantically node-supported; M4 verifies shape consistency but does not infer scientific support.
- The input sample order already reflects the intended sampling policy.

### Explicitly handled

- different node counts;
- different edge counts and graph topologies;
- graphs with zero edges but at least one node;
- scalar and multi-component named node fields;
- all-present or all-absent node weights;
- exact node/edge budget boundaries;
- individual graphs larger than a configured budget;
- empty selected sequences for budget partitioning.

### Unsupported or deferred

- zero-node graphs in a packed microbatch;
- mixing samples with and without node weights under an implicit fallback policy;
- dtype/device promotion during packing;
- automatic field support inference;
- task-level channel concatenation and conditioning tensors;
- cell-level packed model entities;
- topology generated only after budget calculation;
- single-graph partitioning/domain decomposition;
- size-aware statistical sampling/bucketing policy;
- optimizer-batch, loss-weighting, gradient-accumulation, and DDP semantics;
- model-output packed-vs-independent equivalence until a model exists.

## 13. Completion gate

The M4 completion gate was exercised on Calypso on 2026-09-03:

```text
pytest                          -> 81 passed in 17.28 s
ruff check .                    -> passed
python scripts/inspect_config.py -> passed
ruff format --check .           -> passed after merged formatter-only PR #19
```

The validated repository used Python 3.11.13 and PyTorch 2.10.0+cu128. The checks were run from the Calypso login environment with no CUDA device visible; therefore this is target-environment software/numerical validation, not a GPU performance benchmark.

M4 is complete for its frozen scope. This validation introduces no throughput, latency, or peak-memory performance claim; performance evidence remains `ANALYTICAL` until M7 benchmarking.
