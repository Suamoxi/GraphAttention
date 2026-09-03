# Scientific Traceability

## 1. Purpose

This document maps scientifically meaningful concepts to their specification, implementation, validation, and evidence.

The central rule is:

> No mathematically meaningful implementation may exist only in code.

Every scientifically meaningful feature should be traceable through:

$$
\boxed{
\text{scientific concept}
\rightarrow
\text{specification}
\rightarrow
\text{implementation}
\rightarrow
\text{test}
\rightarrow
\text{experiment/benchmark}
}
$$

## 2. Evidence categories

Use these labels in design notes and traceability entries.

### Established method

A mechanism substantially follows prior literature or an established numerical method.

Record the reference, relevant equation/section, and deviations in this repository.

### Project adaptation

A known method changed for this repository or CFD setting. Record source method, exact adaptation, reason, and validation.

### Project hypothesis

A new or unvalidated research idea. Record the hypothesis, intended mechanism, required ablation/comparison, and current evidence.

### Measured result

An empirical result obtained from a specified experiment or benchmark. Record enough context to reproduce it.

## 3. M0-M4 traceability table

| Concept | Type | Specification | Implementation | Validation | Evidence status |
|---|---|---|---|---|---|
| CFD field + mesh + task as core abstraction | Project design | `README.md`, `docs/SCIENTIFIC_SPEC.md` | Architecture-wide | Documentation review | Frozen M0 |
| Graph vertices represent CFD mesh nodes | Project design | `docs/SCIENTIFIC_SPEC.md` §3, `docs/M2_DATA_CONTRACTS.md` | `graph_attention.data.Mesh` | `tests/unit/test_data_contracts.py` | Implemented M2 contract |
| Native cell connectivity preserved from CFD source | Project design | `docs/SCIENTIFIC_SPEC.md` §2-4, `docs/M3_2_AVBP_HDF5.md` | `Mesh.cell_connectivity`, `AVBPHDF5Dataset` | unit tests + real AVBP check | Implemented M3.2; real-file validated 2026-09-02 |
| Explicit AVBP snapshot-to-mesh association | Project data/provenance contract | `docs/M3_2_AVBP_HDF5.md` | `AVBPSampleSpec`, `AVBPHDF5Dataset.sample_specs` | unit tests + real AVBP pair | Implemented M3.2; real-file validated 2026-09-02 |
| Per-process reuse of shared AVBP meshes | Software/data-I/O optimization | `docs/M3_2_AVBP_HDF5.md` | `AVBPHDF5Dataset._mesh_cache` | mesh-identity reuse test | Implemented M3.2; performance unclaimed |
| Hex cell connectivity to sparse node edges | Project geometry transform | `docs/M3_2_AVBP_HDF5.md` | `graph_attention.geometry.hex_connectivity_to_edge_index` | `tests/unit/test_geometry_connectivity.py` | Implemented M3.2; periodic cross-boundary topology excluded |
| AVBP periodic topology | Deferred geometry extension | `docs/M3_2_AVBP_HDF5.md` | not implemented | real HIT file establishes periodic metadata exists | Deferred; do not claim complete physical topology for periodic cases |
| AVBP named-field HDF5 reading | Project adaptation of existing project reader | `docs/M3_2_AVBP_HDF5.md`, `docs/SCIENTIFIC_SPEC.md` §5-7 | `AVBP_FIELD_CATALOG`, `AVBPHDF5Dataset` | unit/config tests + real AVBP pair | Implemented M3.2; real-file validated 2026-09-02 |
| Synthetic variable-mesh contract exerciser | Software test infrastructure | `docs/M3_1_SYNTHETIC_DATA.md` | `graph_attention.data.SyntheticMeshDataset` | `tests/unit/test_synthetic_data.py`, config tests | Implemented M3.1 / non-physical |
| Generic case-level reference-state directive | Project scientific convention using coherent convective scaling | `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/SCIENTIFIC_SPEC.md` §8-11 | `ReferenceScale`, `ReferenceScope`, `ReferenceScales`, `ConvectiveNondimensionalizer` | `tests/unit/test_data_contracts.py`, `tests/unit/test_nondimensionalization.py` | Field-transform runtime target-validated on Calypso; real HIT end-to-end validation 2026-09-03 |
| Authoritative declared case reference documents | Project scientific/provenance convention | `docs/M3_3_NONDIMENSIONALIZATION.md` §10-11 | `CaseDefinition`, `load_case_definition`, `AVBPHDF5Dataset.case_definitions`, `Sample.case_id` | `tests/unit/test_case_definition.py`, `tests/unit/test_avbp_hdf5.py`, real `HIT_LES_FORCED` case file | Implemented and target-validated M3.3 |
| Explicit reference values with non-evaluated derivation metadata | Project reproducibility convention | `docs/M3_3_NONDIMENSIONALIZATION.md` §10 | `load_case_definition` | literal-value/schema tests + real `HIT_LES_FORCED` case load | Implemented and target-validated M3.3 |
| Baseline coordinate nondimensionalization `x/L_ref` | Project scientific/numerical convention | `docs/SCIENTIFIC_SPEC.md` §8.1, `docs/NUMERICAL_CONVENTIONS.md` §13, `docs/M3_3_NONDIMENSIONALIZATION.md` §4, §13 | `ConvectiveNondimensionalizer.nondimensionalize_coordinates`, `dimensionalize_coordinates` | coordinate scale/failure/round-trip tests + real HIT validation | Target-validated 2026-09-03; real HIT dimensionless span = 1 on all three axes |
| `Re`, `Ma`, and related quantities as regime descriptors, not primary normalization scales | Project scientific convention | `docs/M3_3_NONDIMENSIONALIZATION.md` §5, §10, `docs/SCIENTIFIC_SPEC.md` §10 | `RegimeParameter`, `RegimeParameters`, `CaseDefinition.regime_parameters`, `Sample.regime_parameters` | contract/case/AVBP tests + real HIT attachment | Persistence target-validated M3.3; model conditioning deferred |
| `HIT_LES_FORCED` real nondimensionalization validation | Measured result | `docs/M3_3_NONDIMENSIONALIZATION.md` | `scripts/validate_m3_3_avbp.py`, `cases/HIT_LES_FORCED.yaml` | real AVBP snapshot + mesh on Calypso; 35,937 nodes, 32,768 hex cells; float64 round trips within `1e-12` tolerances | TARGET_VALIDATED 2026-09-03 for frozen M3.3 baseline preprocessing scope |
| Packed disconnected variable-graph batching | Established graph batching pattern + project requirements | `docs/ARCHITECTURE.md` §5, `docs/NUMERICAL_CONVENTIONS.md` §2-3, `docs/M4_PACKED_BATCHING.md` | `PackedBatch`, `pack_samples` | `tests/unit/test_packed_batching.py`: variable graph sizes, exact offsets/`ptr`/`batch_index`, no cross-sample edges, named field/metadata preservation | Implemented M4; target validation pending; performance evidence ANALYTICAL |
| Node/edge computational budgets | Project systems design | `docs/ARCHITECTURE.md` §6-7, `docs/NUMERICAL_CONVENTIONS.md` §4, `docs/M4_PACKED_BATCHING.md` | `MicrobatchBudget`, `partition_samples_by_budget` | `tests/unit/test_packed_batching.py`: exact boundaries, node/edge limiting, invalid and oversized cases, order preservation | Implemented M4; target validation pending; concrete hardware limits remain run configuration |
| Statistical batch independent of microbatch composition | Project numerical requirement | `docs/NUMERICAL_CONVENTIONS.md` §5 | trainer/task loss aggregation | gradient-equivalence tests | Planned M6 |
| Physical quadrature-weighted per-sample loss | Established numerical integration principle | `docs/NUMERICAL_CONVENTIONS.md` §6 | task/loss code | reference integration tests | Planned M6; AVBP `VertexData/volume` semantics unresolved |
| DDP global sample-weight consistency | Project distributed-training requirement | `docs/NUMERICAL_CONVENTIONS.md` §7 | trainer | multi-rank equivalence test | Planned M6 |
| Field catalogue with semantic roles | Project data contract | `docs/SCIENTIFIC_SPEC.md` §5-7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec`, `FieldCatalog`, `AVBP_FIELD_CATALOG` | field-contract and AVBP-reader tests | Implemented M2/M3.2 |
| Task-specific named channel selection | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §6 | task/data interface | ordering/provenance tests | Planned M5 |
| Stored vs derived field provenance | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec.stored`, `FieldSpec.provenance` | field-contract tests | Implemented M2 contract |
| Explicit case-level reference semantics | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §8-11, `docs/M2_DATA_CONTRACTS.md`, `docs/M3_3_NONDIMENSIONALIZATION.md` | `ReferenceScale`, `ReferenceScope`, `ReferenceScales.scheme`, `CaseDefinition` | reference-contract, case-definition, nondimensionalization tests + real HIT case | M3.3 runtime implemented and target-validated 2026-09-03 |
| Physical nondimensionalization before statistical scaling | Project scientific/numerical convention | `docs/SCIENTIFIC_SPEC.md` §8-12, `docs/NUMERICAL_CONVENTIONS.md` §8-10, `docs/M3_3_NONDIMENSIONALIZATION.md` | `ConvectiveNondimensionalizer` | field/coordinate explicit-scale and round-trip tests + real HIT validation | Physical runtime target-validated; statistical scaling deferred until task/split contracts |
| Inference-available reference quantities only | Project anti-leakage rule | `docs/SCIENTIFIC_SPEC.md` §9, `docs/M3_3_NONDIMENSIONALIZATION.md` | case-file schema + reference validation in `ConvectiveNondimensionalizer`; regime availability persisted for later task validation | unavailable/snapshot/schema failure tests | M3.3 baseline enforced for physical transforms; task conditioning validation deferred |
| Explicit resolution descriptor | Project multiresolution requirement | `docs/SCIENTIFIC_SPEC.md` §12 | geometry/preprocessing | resolution metadata tests | Planned |
| Node-renumbering equivariance | Fundamental graph-model property | `docs/SCIENTIFIC_SPEC.md` §13 | every graph model | permutation test | Mandatory when model exists |
| Translation/rotation/etc. claims require explicit proof/test | Project scientific rule | `docs/SCIENTIFIC_SPEC.md` §14 | model-specific | property-specific tests | Per model |
| Data/Geometry/Task/Model/Trainer ownership | Project software-science design | `docs/ARCHITECTURE.md` §3 | package architecture; native data, declared references/regimes, geometry transforms, and M4 computational packing remain separated | review/tests | Frozen M0 / instantiated M1-M4 |
| Performance evidence levels | Project engineering rule | `docs/BENCHMARK_PROTOCOL.md` | benchmark tooling | benchmark schema checks | Planned M7 |

## 4. Future model traceability template

When a scientific model component is added, create an entry containing:

```text
Concept:
Status: established method | project adaptation | project hypothesis
Reference:
Relevant paper equation/section:
Scientific definition:
Assumptions:
Repository modification:
Implementation path:
Configuration keys:
Scientific tests:
Numerical tests:
Benchmark evidence:
Known limitations:
```

## 5. Example: future sparse attention entry

The following is a template, not yet an implemented claim.

```text
Concept: sparse graph self-attention on mesh edges
Status: project adaptation / established sparse-attention mechanism
Reference: TBD at implementation time
Scientific definition:
    attention is evaluated only for explicitly supplied sparse edges
Implementation path:
    TBD
Reference implementation:
    dense masked attention on small graphs
Scientific tests:
    - sparse output matches dense masked reference
    - node-renumbering equivariance
    - packed-vs-independent graph equivalence
Performance evidence:
    ANALYTICAL until measured
```

## 6. Change procedure

When code changes a scientifically meaningful mechanism:

1. update the scientific definition first or in the same change;
2. update this traceability table;
3. update or add scientific tests;
4. record the evidence status;
5. do not mark a performance or scientific hypothesis as established merely because tests pass.

## 7. Citation policy

When a mechanism comes from literature, preserve enough citation detail to identify the exact source and idea used.

Do not attribute a project-specific synthesis to one paper unless that paper actually contains the claimed formulation.

When several ideas are combined, record the genealogy explicitly.
