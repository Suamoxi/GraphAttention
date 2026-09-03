# M5 Task Contract and Trivial Baseline

## Status

M5 implements the first explicit deterministic node-regression task contract, a minimal train/validation/test sample split manifest, and a node-local affine baseline model. Target-environment validation on Calypso is required before M5 is marked complete.

M5 deliberately stops before optimizer stepping, distributed loss aggregation, gradient accumulation, or statistical train-set scaling. Those mechanisms remain outside the task/baseline contract.

## 1. Purpose

M4 preserved named node fields in packed batches so task semantics would not be hidden inside the batching layer. M5 now defines the first concrete task boundary:

$$
\boxed{\text{named packed fields} \rightarrow \text{explicit task inputs/targets/conditioning}}
$$

The baseline task is deterministic node regression:

$$
X \rightarrow Y.
$$

The task is intentionally generic with respect to the model architecture. It does not assume attention, message passing, diffusion, or flow matching.

## 2. Explicit field and component ordering

`NodeRegressionTask` receives ordered names for:

```text
input_fields
target_fields
conditioning_parameters
```

The task resolves each selected field through the supplied `FieldCatalog`. Only node-supported fields are accepted in this M5 task.

For a scalar field:

```text
rho -> rho.value
```

For a multi-component field with declared components:

```text
momentum -> momentum.x, momentum.y, momentum.z
```

The resulting task tensors are:

```text
inputs  [N_total, C_in]
targets [N_total, C_out]
```

and `NodeRegressionBatch.input_channels` / `target_channels` preserve the exact semantic channel order.

M5 selects complete declared field groups. Partial slicing of one stored multi-component field is deferred until a real task requires it. Separately stored components such as `rhou`, `rhov`, and `rhow` remain independently selectable and their task order is exactly the configured field order.

## 3. Task-facing packed representation

`NodeRegressionBatch` retains the original `PackedBatch` as `source` and adds task-facing tensors:

```text
coords        [N_total, D]
inputs        [N_total, C_in]
targets       [N_total, C_out]
conditioning  [B, C_cond]
```

The packed graph topology, `batch_index`, `ptr`, node weights, sample IDs, case IDs, reference scales, regime descriptors, and metadata remain available through the preserved source batch.

Keeping the original packed representation avoids erasing provenance when the task creates transformed model inputs.

## 4. Physical preprocessing composition

When `physical_nondimensionalization=true`, M5 applies the existing M3.3 `ConvectiveNondimensionalizer` independently to each graph using that graph's declared `ReferenceScales`.

For each graph `g`:

$$
q_{g,\mathrm{dim}} \rightarrow q_g^*,
\qquad
x_{g,\mathrm{dim}} \rightarrow x_g^*.
$$

The transformed fields are then concatenated into the configured task channel order.

This is important when one packed microbatch contains several physical cases with different numerical reference values. A single microbatch-level reference scale is not assumed.

The original packed tensors are not mutated. The task representation therefore distinguishes raw packed provenance from transformed task inputs.

For the default synthetic smoke configuration, physical nondimensionalization is disabled because the synthetic fields are deliberately non-physical and do not carry an M3.3 reference-state contract.

## 5. Global regime conditioning

Requested `conditioning_parameters` are read from each graph's `RegimeParameters` and materialized as:

$$
c \in \mathbb{R}^{B\times C_{\mathrm{cond}}}.
$$

The configured order is preserved exactly.

M5 requires every requested conditioning quantity to:

- exist for every graph in the task batch;
- have `inference_available=true`;
- use the same explicit `definition` string across all graphs in the batch.

This prevents a task from silently concatenating quantities that share a name but have different physical meanings.

Conditioning remains graph-level. It is not repeated over nodes by the task layer.

## 6. Split manifest

`graph_attention.data.SplitManifest` provides an explicit immutable train/validation/test partition by sample ID.

The contract requires:

- a non-empty training split;
- non-empty sample-ID strings;
- no duplicate sample membership within or across splits.

The manifest records membership only. It does not randomly generate splits, inspect target values, or infer membership from file paths.

This establishes the minimum split identity required before a future statistical scaler can be fitted strictly on training samples.

## 7. Trivial baseline model

`NodeLinearBaseline` is deliberately geometry-blind. For node `i` in graph `g(i)` it implements:

$$
\boxed{
\hat y_i = W\,[x_i, c_{g(i)}] + b
}
$$

when global conditioning is enabled, and:

$$
\boxed{
\hat y_i = W x_i + b
}
$$

otherwise.

The baseline does not consume coordinates or `edge_index`.

This is intentional. It provides a null geometric baseline that validates:

- task channel construction;
- packed versus independent graph execution;
- global-conditioning plumbing;
- node-renumbering equivariance;
- explicit model input/output channel compatibility.

It must not be interpreted as a competitive CFD model or as evidence about sparse attention quality.

## 8. Packed versus independent equivalence

Because the baseline acts independently on each node, packing disconnected graphs must not change its result.

For samples `G_1, ..., G_B`, the test requires:

$$
f(\mathrm{pack}(G_1,\ldots,G_B))
=
\mathrm{concat}(f(G_1),\ldots,f(G_B))
$$

up to the numerical tolerance appropriate for the dtype.

M5 adds this test, closing the model-output equivalence test that was intentionally deferred in M4 until a model existed.

## 9. Node-renumbering equivariance

The M5 baseline is pointwise and therefore satisfies the repository's required node-renumbering property:

$$
f(PX)=P f(X).
$$

A dedicated permutation test verifies the implemented baseline.

This test does not prove the property for future graph models. Every later model must carry its own permutation/equivariance validation.

## 10. Configuration

The default synthetic configuration now instantiates the actual task and model:

```yaml
# task/regression.yaml
_target_: graph_attention.tasks.NodeRegressionTask
input_fields:
  - momentum
target_fields:
  - rho
conditioning_parameters: []
physical_nondimensionalization: false
```

```yaml
# model/baseline.yaml
_target_: graph_attention.models.NodeLinearBaseline
in_channels: 2
out_channels: 1
conditioning_channels: 0
bias: true
```

The default model channel counts correspond to the default 2D synthetic smoke task. Changing the selected fields, spatial dimension, or conditioning requires updating the model channel counts explicitly. A mismatch fails at runtime rather than being silently adapted.

## 11. Statistical scaling remains deferred

M5 establishes the two prerequisites that M3.3 lacked:

1. explicit task channel semantics;
2. explicit training-split identity.

It still does **not** fit statistical normalization values.

For variable meshes, fitting training statistics requires an explicit weighting convention. A global node-wise mean would give larger meshes more statistical influence, while sample-balanced or physically weighted estimators represent different numerical/scientific choices. That policy is not silently chosen in M5.

A train-only statistical scaler must therefore be implemented and documented before meaningful M6 training begins, after its weighting convention is explicitly selected.

## 12. Tests

M5 adds tests covering:

- exact scalar/vector channel ordering;
- named input and target concatenation;
- per-case physical nondimensionalization in one packed batch;
- preservation of the original dimensional packed source;
- ordered regime conditioning;
- inference-availability enforcement;
- inconsistent conditioning-definition rejection;
- non-node and non-floating task-field rejection;
- explicit split membership and overlap rejection;
- baseline input/output shape validation;
- packed-versus-independent baseline equivalence;
- node-renumbering equivariance;
- graph-level conditioning expansion inside the baseline;
- Hydra instantiation and end-to-end default synthetic task/model connection.

## 13. Assumptions and edge cases

### Assumptions introduced or inherited

- The initial M5 task is node-to-node deterministic regression on one packed node graph.
- Selected task fields are represented as `[N]` scalar fields or `[N, C]` complete declared component groups.
- All selected input fields share one floating dtype/device; all targets share that same dtype/device.
- The model configuration explicitly matches the task's resulting channel counts.
- Requested conditioning names have identical physical definitions across graphs.
- Physical nondimensionalization uses the existing M3.3 field mappings and per-graph references.
- The training experiment has at least one explicitly identified training sample.

### Explicitly handled

- scalar and multi-component node fields;
- separate stored vector components selected in explicit order;
- different numerical reference scales across graphs in one packed batch;
- no-conditioning tasks through an empty `[B, 0]` tensor;
- missing/unavailable/inconsistently defined regime conditioning;
- overlapping split membership;
- packed and independent baseline execution;
- node permutations for the baseline.

### Unsupported or deferred

- partial component selection within one stored multi-component field;
- cell/face/global prediction targets;
- heterogeneous source/query graphs for super-resolution or wall tasks;
- temporal pairing/sequence semantics;
- diffusion and flow-matching task mathematics;
- learned/statistical train-set scaling;
- loss reduction and node/quadrature weighting;
- optimizer batching, gradient accumulation, AMP, and DDP;
- graph-aware or geometry-aware learned transformations;
- automatic inference of model channel counts from data/task configuration.

### Failure behavior

Invalid field support, channel layout, dtype/device compatibility, missing conditioning, unavailable conditioning, inconsistent conditioning semantics, overlapping split membership, and model channel mismatches fail explicitly with exceptions. No fallback field ordering, dtype promotion, conditioning substitution, or split inference is performed.

## 14. Efficiency and evidence status

Task field concatenation and the node-local baseline are linear in the number of selected node values. Per-graph physical preprocessing currently iterates over graphs and slices using `ptr`; this is clear and correct for M5 but is not presented as an optimized GPU preprocessing kernel.

No throughput, latency, or peak-memory claim is made. Performance evidence remains `ANALYTICAL` until the benchmark milestone.

## 15. Completion gate

After merging, run on Calypso:

```bash
pytest
ruff check .
ruff format --check .
python scripts/inspect_config.py
```

M5 may be marked complete for its frozen task/baseline scope once these checks pass with the new tests included. Statistical scaling remains a required follow-on before meaningful M6 optimization work.
