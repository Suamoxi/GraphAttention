# M2 Core Scientific Data Contracts

## Purpose

M2 turns the frozen M0 field/mesh/reference semantics into minimal runtime contracts without implementing a concrete CFD reader, preprocessing pipeline, packed batching, task, or model.

The central rule remains:

$$
\boxed{\text{data describes what exists; it does not decide what is learned}}
$$

## Implemented contracts

### `FieldSpec`

A field is identified by a canonical name and explicit semantics:

- spatial support: node, cell, face, or global;
- scientific role;
- component names/order;
- raw source path when applicable;
- units/dimensional convention when known;
- provenance;
- stored-versus-derived state.

Tensor shape alone is never used to infer scientific meaning.

### `FieldCatalog`

A catalogue is a uniquely named collection of `FieldSpec` objects. It provides ordered lookup so later tasks can request named fields without anonymous channel slices.

M2 does not yet define task input/target selection; that belongs to the task layer.

### `ReferenceScale`, `ReferenceScope`, and `ReferenceScales`

Each physical reference quantity stores its numerical value separately from its semantic definition and provenance. M3.3 extends the original M2 contract with optional units, explicit scope, inference availability, and an optional derivation description. `ReferenceScales` can also name the reference scheme that gives the collection a coherent physical meaning.

For example:

```text
scheme: bulk_flow_reference
name: U_ref
value: 12.5
definition: bulk_velocity
units: m/s
provenance: case_metadata
scope: case
inference_available: true
```

Supported scopes are `case`, `operating_condition`, and `snapshot`. The baseline M3.3 preprocessing rejects snapshot-scoped references; supporting them later requires an explicit scientific objective rather than a silent normalization choice.

The low-level data contract still permits references whose units/provenance are not yet populated so readers can represent partially known source metadata. Physical preprocessing is stricter: every reference it actually uses must have the M3.3 semantic metadata required by that transformation.

### `Mesh`

The initial canonical mesh contract represents CFD mesh nodes as graph vertices:

- `coords [N, D]`;
- `edge_index [2, E]` using `torch.long`;
- optional `node_weights [N]` for future quadrature/control-volume use;
- optional mesh identifier and metadata.

M3.2 extends this runtime contract with optional native `cell_connectivity [C, K]`, also using zero-based `torch.long` node indices. This preserves cell-to-node topology supplied by a CFD source without making cells separate learned graph entities or forcing the data reader to construct model graph edges.

The contract validates dimensions, finite coordinates/weights, and connectivity bounds. It deliberately does not prescribe directed/undirected duplication, self-loop policy, edge ordering, geometric features, dilation, or long-range augmentation; those are geometry/model decisions.

### `Sample`

A sample contains:

- stable `sample_id`;
- one native `Mesh`;
- named loaded field tensors;
- case-level `ReferenceScales`;
- non-scientific/general sample metadata.

`Sample.validate_against(FieldCatalog)` validates declared field availability and the aspects of support that M2 can know without inventing cell/face topology. Node-supported fields must have leading dimension `N`; explicitly grouped multi-component fields must preserve their documented final component dimension.

## Deliberate non-goals

M2 does not implement:

- HDF5/AVBP readers;
- synthetic datasets;
- field derivation;
- nondimensionalization;
- statistical scaling;
- task channel selection;
- packed graph batching;
- sampler/packer logic;
- Lightning data modules;
- graph geometry transforms;
- sparse attention.

Those capabilities are later milestones and should consume these contracts rather than redefine them.

## Why runtime `Mesh`/`Sample` are not frozen dataclasses

`FieldSpec`, `FieldCatalog`, and reference-scale semantics are immutable value contracts. `Mesh` and `Sample` contain `torch.Tensor` objects, which are mutable regardless of dataclass freezing. Marking the containers frozen would therefore provide misleading immutability rather than a real scientific guarantee.

## M2 validation gate

M2 is complete when:

```bash
pytest
ruff check .
ruff format --check .
```

pass and the following behaviors are tested:

- duplicate field names are rejected;
- component order is explicit;
- reference values retain explicit definitions;
- canonical mesh tensor shapes and edge bounds are validated;
- node-supported sample fields are checked against mesh node count;
- unknown fields cannot silently pass catalogue validation.

Passing M2 validates the contracts only. It does not validate a reader, preprocessing transformation, model, or performance claim.
