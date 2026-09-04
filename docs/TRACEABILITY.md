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

## 3. M0-M8 traceability table

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
| `Re`, `Ma`, and related quantities as regime descriptors | Project scientific convention | `docs/M3_3_NONDIMENSIONALIZATION.md` §5, §10, `docs/SCIENTIFIC_SPEC.md` §10, `docs/M5_TASK_BASELINE.md` §5 | `RegimeParameter`, `RegimeParameters`, `NodeRegressionTask` conditioning selection | contract/case/AVBP tests + M5 conditioning order/availability/definition tests | Persistence target-validated M3.3; model-visible M5 selection target-validated 2026-09-03 |
| `HIT_LES_FORCED` real nondimensionalization validation | Measured result | `docs/M3_3_NONDIMENSIONALIZATION.md` | `scripts/validate_m3_3_avbp.py`, `cases/HIT_LES_FORCED.yaml` | real AVBP snapshot + mesh on Calypso; 35,937 nodes, 32,768 hex cells; float64 round trips within `1e-12` tolerances | TARGET_VALIDATED 2026-09-03 for frozen M3.3 baseline preprocessing scope |
| Packed disconnected variable-graph batching | Established graph batching pattern + project requirements | `docs/ARCHITECTURE.md` §5, `docs/NUMERICAL_CONVENTIONS.md` §2-3, `docs/M4_PACKED_BATCHING.md` | `PackedBatch`, `pack_samples` | `tests/unit/test_packed_batching.py` + full 81-test Calypso suite; exact offsets/`ptr`/`batch_index`, no cross-sample edges, named field/metadata preservation | TARGET_VALIDATED 2026-09-03 for M4 software/numerical scope; performance evidence ANALYTICAL |
| Node/edge computational budgets | Project systems design | `docs/ARCHITECTURE.md` §6-7, `docs/NUMERICAL_CONVENTIONS.md` §4, `docs/M4_PACKED_BATCHING.md` | `MicrobatchBudget`, `partition_samples_by_budget` | `tests/unit/test_packed_batching.py` + full Calypso suite; exact boundaries, node/edge limiting, invalid/oversized cases, order preservation | TARGET_VALIDATED 2026-09-03; concrete hardware limits remain run configuration; performance evidence ANALYTICAL |
| Explicit train/validation/test sample identity | Project anti-leakage/reproducibility contract | `docs/REPRODUCIBILITY.md` §3, §7, `docs/M5_TASK_BASELINE.md` §6 | `SplitManifest` | `tests/unit/test_data_splits.py` | TARGET_VALIDATED M5 2026-09-03; random split generation intentionally absent |
| Task-specific named channel selection | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §6, §16, `docs/M5_TASK_BASELINE.md` §2-3 | `NodeRegressionTask`, `NodeRegressionBatch` | `tests/unit/test_task_regression.py`, config integration | TARGET_VALIDATED M5 2026-09-03 |
| Per-graph physical preprocessing composed into task representation | Project scientific/numerical integration | `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/M5_TASK_BASELINE.md` §4 | `NodeRegressionTask._preprocess_physical` using `ConvectiveNondimensionalizer` | M5 two-case packed nondimensionalization test | TARGET_VALIDATED M5 2026-09-03; source batch remains dimensional |
| Node-local affine baseline `y_i = W[x_i,c_g] + b` | Null geometric baseline | `docs/M5_TASK_BASELINE.md` §7-9, `docs/SCIENTIFIC_SPEC.md` §16 | `NodeLinearBaseline` | baseline shape, conditioning, packed-vs-independent, and permutation tests | TARGET_VALIDATED M5 2026-09-03; no CFD performance claim |
| Sample-balanced train-only statistical scaling | Project numerical convention for variable meshes | `docs/NUMERICAL_CONVENTIONS.md` §18, `docs/M6_TRAINING_CORRECTNESS.md` §2-4 | `ChannelStandardizer`, `TaskStandardizers`, `fit_train_standardizers` | `tests/unit/test_training_scaling.py` + full 122-test Calypso gate | TARGET_VALIDATED M6 2026-09-03; performance evidence ANALYTICAL |
| Statistical batch independent of microbatch composition | Project numerical requirement | `docs/NUMERICAL_CONVENTIONS.md` §5, §18, `docs/M6_TRAINING_CORRECTNESS.md` §6 | `train_equal_sample_optimizer_step` | `tests/unit/test_training_step.py` gradient/update partition-equivalence test | TARGET_VALIDATED M6 2026-09-03 |
| Per-sample MSE with optional spatial weights | Project baseline regression objective + established weighted integration form | `docs/SCIENTIFIC_SPEC.md` §17, `docs/NUMERICAL_CONVENTIONS.md` §6, §18, `docs/M6_TRAINING_CORRECTNESS.md` §5 | `sample_reduced_mse` | `tests/unit/test_training_losses.py` + full M6 Calypso gate | TARGET_VALIDATED M6 2026-09-03; AVBP physical node-quadrature semantics remain unresolved |
| DDP global sample-weight consistency | Project distributed-training requirement | `docs/NUMERICAL_CONVENTIONS.md` §7, §18, `docs/M6_TRAINING_CORRECTNESS.md` §7-10 | `equal_sample_ddp_backward_scale`, `train_equal_sample_optimizer_step` | analytical unit test + `scripts/validate_m6_ddp.py` two-rank global-update comparison | TARGET_VALIDATED correctness on Calypso CPU/Gloo 2026-09-03; GPU/NCCL performance unclaimed |
| Autocast-compatible regression loss | Project numerical training convention using native PyTorch autocast | `docs/NUMERICAL_CONVENTIONS.md` §14, §18, `docs/M6_TRAINING_CORRECTNESS.md` §9 | `sample_reduced_mse`, `train_equal_sample_optimizer_step` | CPU BF16 unit smoke test | TARGET_VALIDATED for tested CPU path in M6; CUDA BF16/FP16 target validation deferred |
| Field catalogue with semantic roles | Project data contract | `docs/SCIENTIFIC_SPEC.md` §5-7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec`, `FieldCatalog`, `AVBP_FIELD_CATALOG` | field-contract and AVBP-reader tests | Implemented M2/M3.2 |
| Stored vs derived field provenance | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec.stored`, `FieldSpec.provenance` | field-contract tests | Implemented M2 contract |
| Explicit case-level reference semantics | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §8-11, `docs/M2_DATA_CONTRACTS.md`, `docs/M3_3_NONDIMENSIONALIZATION.md` | `ReferenceScale`, `ReferenceScope`, `ReferenceScales.scheme`, `CaseDefinition` | reference-contract, case-definition, nondimensionalization tests + real HIT case | M3.3 runtime implemented and target-validated 2026-09-03 |
| Physical nondimensionalization before statistical scaling | Project scientific/numerical convention | `docs/SCIENTIFIC_SPEC.md` §8-12, §17, `docs/NUMERICAL_CONVENTIONS.md` §8-10, §18, `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/M6_TRAINING_CORRECTNESS.md` | `ConvectiveNondimensionalizer`; `fit_train_standardizers` | M3.3 transform/real-HIT tests + M6 train-only scaling tests | Physical runtime and M6 train-only scaler target-validated on Calypso |
| Inference-available reference/conditioning quantities only | Project anti-leakage rule | `docs/SCIENTIFIC_SPEC.md` §9-10, `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/M5_TASK_BASELINE.md` §5 | reference validation in `ConvectiveNondimensionalizer`; conditioning validation in `NodeRegressionTask` | unavailable/snapshot reference tests + M5 unavailable-conditioning test | Physical references target-validated M3.3; conditioning selection target-validated M5 |
| Explicit resolution descriptor | Project multiresolution requirement | `docs/SCIENTIFIC_SPEC.md` §12 | geometry/preprocessing | resolution metadata tests | Planned |
| Node-renumbering equivariance | Fundamental graph-model property | `docs/SCIENTIFIC_SPEC.md` §13, §18, `docs/M5_TASK_BASELINE.md` §9, `docs/M8_SPARSE_TRANSFORMER.md` §9 | `NodeLinearBaseline`, `SparseGraphTransformer` | baseline permutation test + `tests/unit/test_sparse_transformer.py` consistent node/edge permutation test | TARGET_VALIDATED for M5 baseline; M8 implementation validation pending Calypso software gate; mandatory separately for every later graph model |
| Translation/rotation/etc. claims require explicit proof/test | Project scientific rule | `docs/SCIENTIFIC_SPEC.md` §14 | model-specific | property-specific tests | Per model; M8 makes no geometry invariance claim because it does not consume coordinates |
| Data/Geometry/Task/Model/Trainer ownership | Project software-science design | `docs/ARCHITECTURE.md` §3 | data contracts/splits, geometry transforms, M4 packing, M5 task/model separation, M6 training primitives, M7 benchmark tooling, M8 sparse model | review/tests | Frozen M0 / instantiated M1-M8 |
| Performance evidence levels | Project engineering rule | `docs/BENCHMARK_PROTOCOL.md` | `graph_attention.utils.benchmarking`, `scripts/benchmark_m7.py`, `scripts/benchmark_m8.py` | benchmark utility/CLI tests + target benchmark gates | TARGET_VALIDATED M7 protocol; M8 graph-aware performance remains ANALYTICAL until its target runs |
| Framework/null-baseline performance reference | Measured result | `docs/BENCHMARK_PROTOCOL.md`, `docs/M7_BENCHMARKS.md` §13 | `scripts/benchmark_m7.py` | Slurm job `400132`, clean `main` SHA `79b156e27842618a54a0be18a81ea76c994ac140`, NVIDIA GH200 480GB, FP32; synthetic S3 and real HIT | TARGET_VALIDATED 2026-09-04: S3 median forward/training = 0.0760/2.4206 ms; real HIT = 0.0549/1.5733 ms; real-HIT training incremental PyTorch peak allocation = 5,752,320 B; null-model/framework evidence only |
| Sparse one-hop scaled dot-product attention on supplied mesh edges | Project adaptation of established scaled dot-product and graph-neighborhood attention | `docs/SCIENTIFIC_SPEC.md` §18, `docs/M8_SPARSE_TRANSFORMER.md` §2-5 | `SparseMultiheadAttention`, `SparseGraphTransformerBlock`, `SparseGraphTransformer` | explicit-neighbor reference, edge-order tolerance, packed-vs-independent, node-renumbering, empty-edge and training-path tests in `tests/unit/test_sparse_transformer.py` | Implemented M8; scientific/software target validation pending; performance evidence ANALYTICAL |
| Stabilized sparse attention reduction with FP32 score/softmax under BF16/FP16 projections | Project numerical stability policy | `docs/NUMERICAL_CONVENTIONS.md` §19, `docs/M8_SPARSE_TRANSFORMER.md` §8 | `SparseMultiheadAttention` | explicit FP32 reference test + CPU BF16 autocast finite-output smoke in `tests/unit/test_sparse_transformer.py` | Implemented M8; CUDA BF16/FP16 target validation deferred |
| Sparse-transformer performance reference | Measured-result framework | `docs/BENCHMARK_PROTOCOL.md`, `docs/M8_SPARSE_TRANSFORMER.md` §11 | `scripts/benchmark_m8.py` | CPU CLI smoke + required synthetic S3 and real-HIT single-GPU target runs | ANALYTICAL until M8 Calypso measurements; no target graph-attention latency/memory claim yet |

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

## 5. M8 sparse-attention genealogy

```text
Concept:
    sparse graph self-attention on supplied one-hop mesh edges
Status:
    project adaptation of established scaled dot-product attention and graph-neighborhood attention
References:
    Vaswani et al., Attention Is All You Need, NeurIPS 2017, arXiv:1706.03762
    Veličković et al., Graph Attention Networks, ICLR 2018, arXiv:1710.10903
Project-specific definition:
    Transformer-style multi-head scaled dot-product scores are evaluated only on supplied directed edges;
    this is not the additive GAT scoring equation.
Repository modification:
    no implicit self-loops, topology augmentation, or geometric attention terms are added in M8.
Implementation path:
    src/graph_attention/models/sparse_transformer.py
Configuration:
    configs/model/sparse_transformer.yaml
Scientific tests:
    sparse versus explicit-neighbor reference
    edge-list reordering tolerance
    node-renumbering equivariance
    disconnected packed versus independent execution
Numerical tests:
    empty-edge finite behavior
    CPU BF16 autocast smoke
    invalid edge/configuration failure behavior
Benchmark evidence:
    ANALYTICAL until scripts/benchmark_m8.py is target-validated on Calypso
Known limitations:
    no coordinate/edge geometry, no periodic cross-boundary HIT augmentation, no fused sparse kernel,
    no CUDA low-precision or multi-node performance evidence yet
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
