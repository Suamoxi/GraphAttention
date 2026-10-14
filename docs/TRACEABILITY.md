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

## 3. M0-M13 traceability table

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
| Explicit train/validation/test sample identity | Project anti-leakage/reproducibility contract | `docs/REPRODUCIBILITY.md` §3, §7, `docs/M5_TASK_BASELINE.md` §6 | `SplitManifest` | `tests/unit/test_data_splits.py` | TARGET_VALIDATED M5 2026-09-03; generic random split generation intentionally absent |
| Task-specific named channel selection | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §6, §16, `docs/M5_TASK_BASELINE.md` §2-3 | `NodeRegressionTask`, `NodeRegressionBatch` | `tests/unit/test_task_regression.py`, config integration | TARGET_VALIDATED M5 2026-09-03 |
| Per-graph physical preprocessing composed into task representation | Project scientific/numerical integration | `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/M5_TASK_BASELINE.md` §4 | `NodeRegressionTask._preprocess_physical` using `ConvectiveNondimensionalizer` | M5 two-case packed nondimensionalization test | TARGET_VALIDATED M5 2026-09-03; source batch remains dimensional |
| Node-local affine baseline `y_i = W[x_i,c_g] + b` | Null geometric baseline | `docs/M5_TASK_BASELINE.md` §7-9, `docs/SCIENTIFIC_SPEC.md` §16 | `NodeLinearBaseline` | baseline shape, conditioning, packed-vs-independent, and permutation tests | TARGET_VALIDATED M5 2026-09-03; no CFD performance claim |
| Sample-balanced train-only statistical scaling | Project numerical convention for variable meshes | `docs/NUMERICAL_CONVENTIONS.md` §18, `docs/M6_TRAINING_CORRECTNESS.md` §2-4 | `ChannelStandardizer`, `TaskStandardizers`, `fit_train_standardizers` | `tests/unit/test_training_scaling.py` + full 122-test Calypso gate | TARGET_VALIDATED M6 2026-09-03; performance evidence ANALYTICAL |
| Statistical batch independent of microbatch composition | Project numerical requirement | `docs/NUMERICAL_CONVENTIONS.md` §5, §18, `docs/M6_TRAINING_CORRECTNESS.md` §6 | `train_equal_sample_optimizer_step` | `tests/unit/test_training_step.py` gradient/update partition-equivalence test | TARGET_VALIDATED M6 2026-09-03 |
| Per-sample MSE with optional spatial weights | Project baseline regression objective + established weighted integration form | `docs/SCIENTIFIC_SPEC.md` §17, `docs/NUMERICAL_CONVENTIONS.md` §6, §18 | `sample_reduced_mse` | `tests/unit/test_training_losses.py` + full M6 Calypso gate | TARGET_VALIDATED M6 2026-09-03; AVBP physical node-quadrature semantics remain unresolved |
| DDP global sample-weight consistency | Project distributed-training requirement | `docs/NUMERICAL_CONVENTIONS.md` §7, §18, `docs/M6_TRAINING_CORRECTNESS.md` §7-10 | `equal_sample_ddp_backward_scale`, `train_equal_sample_optimizer_step` | analytical unit test + `scripts/validate_m6_ddp.py` two-rank global-update comparison | TARGET_VALIDATED correctness on Calypso CPU/Gloo 2026-09-03; GPU/NCCL performance unclaimed |
| Autocast-compatible regression loss | Project numerical training convention using native PyTorch autocast | `docs/NUMERICAL_CONVENTIONS.md` §14, §18, `docs/M6_TRAINING_CORRECTNESS.md` §9 | `sample_reduced_mse`, `train_equal_sample_optimizer_step` | CPU BF16 unit smoke test | TARGET_VALIDATED for tested CPU path in M6; CUDA BF16/FP16 target validation deferred |
| Field catalogue with semantic roles | Project data contract | `docs/SCIENTIFIC_SPEC.md` §5-7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec`, `FieldCatalog`, `AVBP_FIELD_CATALOG` | field-contract and AVBP-reader tests | Implemented M2/M3.2 |
| Stored vs derived field provenance | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §7, `docs/M2_DATA_CONTRACTS.md` | `FieldSpec.stored`, `FieldSpec.provenance` | field-contract tests | Implemented M2 contract |
| Explicit case-level reference semantics | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §8-11, `docs/M2_DATA_CONTRACTS.md`, `docs/M3_3_NONDIMENSIONALIZATION.md` | `ReferenceScale`, `ReferenceScope`, `ReferenceScales.scheme`, `CaseDefinition` | reference-contract, case-definition, nondimensionalization tests + real HIT case | M3.3 runtime implemented and target-validated 2026-09-03 |
| Physical nondimensionalization before statistical scaling | Project scientific/numerical convention | `docs/SCIENTIFIC_SPEC.md` §8-12, §17, `docs/NUMERICAL_CONVENTIONS.md` §8-10, §18 | `ConvectiveNondimensionalizer`; `fit_train_standardizers` | M3.3 transform/real-HIT tests + M6 train-only scaling tests | Physical runtime and M6 train-only scaler target-validated on Calypso |
| Inference-available reference/conditioning quantities only | Project anti-leakage rule | `docs/SCIENTIFIC_SPEC.md` §9-10, `docs/M3_3_NONDIMENSIONALIZATION.md`, `docs/M5_TASK_BASELINE.md` §5 | reference validation in `ConvectiveNondimensionalizer`; conditioning validation in `NodeRegressionTask` | unavailable/snapshot reference tests + M5 unavailable-conditioning test | Physical references target-validated M3.3; conditioning selection target-validated M5 |
| Explicit resolution descriptor | Project multiresolution requirement | `docs/SCIENTIFIC_SPEC.md` §12 | geometry/preprocessing | resolution metadata tests | Planned |
| Node-renumbering equivariance | Fundamental graph-model property | `docs/SCIENTIFIC_SPEC.md` §13, §18-19, `docs/M5_TASK_BASELINE.md` §9, `docs/M8_SPARSE_TRANSFORMER.md` §9, `docs/M9_GEOMETRIC_ATTENTION.md` §9 | `NodeLinearBaseline`, `SparseGraphTransformer`, `GeometricSparseGraphTransformer` | baseline permutation test + M8 and M9 consistent node/edge/coordinate permutation tests | TARGET_VALIDATED for M5/M8/M9 software/scientific scope |
| Translation-invariant relative edge displacement | Project geometry/scientific convention | `docs/SCIENTIFIC_SPEC.md` §14, §19, `docs/M9_GEOMETRIC_ATTENTION.md` §2-4 | `graph_attention.geometry.edge_relative_displacement` | `tests/unit/test_geometry_relative.py`, M9 model translation test | TARGET_VALIDATED M9 software/scientific scope; GH200 target benchmark job `403939` |
| Translation/rotation/etc. claims require explicit proof/test | Project scientific rule | `docs/SCIENTIFIC_SPEC.md` §14 | model-specific | property-specific tests | M9 claims translation invariance only; no rotation/reflection/scale claim |
| Data/Geometry/Task/Model/Trainer ownership | Project software-science design | `docs/ARCHITECTURE.md` §3 | data contracts/splits, geometry transforms, M4 packing, M5 task/model separation, M6 training primitives, M7 benchmark tooling, M8 sparse model, M9 relative geometry + geometric model, M10 slice adapter/topology/runner, M11 flow-matching task/runner | review/tests | Frozen M0 / instantiated M1-M11 |
| Performance evidence levels | Project engineering rule | `docs/BENCHMARK_PROTOCOL.md` | `graph_attention.utils.benchmarking`, `scripts/benchmark_m7.py`, `scripts/benchmark_m8.py`, `scripts/benchmark_m9.py` | benchmark utility/CLI tests + target benchmark gates | TARGET_VALIDATED M7/M8/M9 reference paths |
| Framework/null-baseline performance reference | Measured result | `docs/BENCHMARK_PROTOCOL.md`, `docs/M7_BENCHMARKS.md` §13 | `scripts/benchmark_m7.py` | Slurm job `400132`, clean SHA `79b156e27842618a54a0be18a81ea76c994ac140`, NVIDIA GH200 480GB, FP32 | TARGET_VALIDATED 2026-09-04: S3 forward/training 0.0760/2.4206 ms; real HIT 0.0549/1.5733 ms; null-model/framework evidence only |
| Sparse one-hop scaled dot-product attention on supplied mesh edges | Project adaptation of established scaled dot-product and graph-neighborhood attention | `docs/SCIENTIFIC_SPEC.md` §18, `docs/M8_SPARSE_TRANSFORMER.md` §2-5 | `SparseMultiheadAttention`, `SparseGraphTransformerBlock`, `SparseGraphTransformer` | explicit-neighbor reference, edge-order tolerance, packed-vs-independent, node-renumbering, empty-edge and training-path tests | TARGET_VALIDATED M8 software/scientific scope; single-GH200 performance target-validated in job `400187` |
| Stabilized sparse attention reduction with FP32 score/softmax under BF16/FP16 projections | Project numerical stability policy | `docs/NUMERICAL_CONVENTIONS.md` §19, `docs/M8_SPARSE_TRANSFORMER.md` §8 | `SparseMultiheadAttention` | explicit FP32 reference test + CPU BF16 autocast finite-output smoke | FP32 target path validated M8; CUDA BF16/FP16 target validation deferred |
| Sparse-transformer performance reference | Measured result | `docs/BENCHMARK_PROTOCOL.md`, `docs/M8_SPARSE_TRANSFORMER.md` §17 | `scripts/benchmark_m8.py` | jobs `400187` and `400194`, one NVIDIA GH200 480GB, FP32, clean SHA `dea85e8645d614992a34003b1998d4f1a7e58261` | TARGET_VALIDATED: S3 forward/training 3.0808/90.3210 ms; real HIT 5.3406/17.7296 ms; degree stress shows >10x training penalty when an extreme hub is introduced at nearly fixed N/E |
| Relative-displacement geometric attention bias | Project adaptation | `docs/SCIENTIFIC_SPEC.md` §19, `docs/M9_GEOMETRIC_ATTENTION.md` §1-6 | `GeometricSparseMultiheadAttention`, `GeometricSparseGraphTransformerBlock`, `GeometricSparseGraphTransformer` | explicit geometric-attention reference, geometry sensitivity, translation, packed-vs-independent, node-renumbering, training integration tests | TARGET_VALIDATED M9 software/scientific and FP32 single-GH200 performance scope; job `403939` |
| M9 geometry uses displacement only, no explicit distance | Project scientific/ablation decision | `docs/M9_GEOMETRIC_ATTENTION.md` §2, `docs/NUMERICAL_CONVENTIONS.md` §20 | `edge_relative_displacement`, `GeometricSparseMultiheadAttention.geometry_mlp` | relative-geometry tests + M9 explicit reference | TARGET_VALIDATED for frozen M9 reference; explicit-distance inductive bias remains a separate deferred ablation |
| Geometric sparse-transformer performance reference | Measured result | `docs/M9_GEOMETRIC_ATTENTION.md` §10-11 | `scripts/benchmark_m9.py` | job `403939`, one NVIDIA GH200 480GB, FP32, clean SHA `83e160846badb151eef0a09c2f1e2234da22bc24` | TARGET_VALIDATED 2026-09-07: S3 forward/training 3.4138/90.9458 ms; real HIT forward/training medians 5.7989/18.7044 ms; modest overhead versus M8 and no new performance pathology observed |
| diffusion4avbp fixed-slice artifact adapter | Project data/provenance adaptation | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §2 | `PrecomputedSlicePTDataset` | `tests/unit/test_slice_pt.py` + real-artifact preflight | TARGET_VALIDATED M10 2026-09-07 on 1,488 real slices |
| Grouped split by source 3-D snapshot | Project anti-leakage convention reproduced from diffusion4avbp | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §3 | `make_grouped_split_manifest`, `PrecomputedSlicePTDataset.group_id` | `tests/unit/test_grouped_splits.py` + real split-overlap check | TARGET_VALIDATED M10: 260/55/57 source groups with zero overlap |
| Shared canonical 2-D slice coordinates preserved as model geometry | Project geometry convention | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §4 | `PrecomputedSlicePTDataset` | exact shared-coordinate and x/y/z extraction-orientation independence tests + real preflight | TARGET_VALIDATED M10: real coordinates `(1089,2)`, grid `33x33` |
| Slice extraction orientation excluded from first model input | Project ablation-control decision | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §4 | slice metadata retained as provenance only | `tests/unit/test_slice_pt.py` verifies axis/slice-coordinate changes do not change `Mesh.coords` | Implemented M10 correction; explicit orientation conditioning deferred |
| Non-periodic bidirectional Cartesian 4-neighbour slice topology | Project geometry hypothesis for first controlled ablation | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §5 | `cartesian_4_neighbor_edge_index` | exact 2x3 reference, edge-case tests, real preflight | TARGET_VALIDATED M10: 4,224 directed edges per 33x33 slice; periodic wrap deliberately deferred |
| HIT slice task `[rhou,rhov,rhow,rhoE] -> rho` | Project scientific ablation task | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §1, §6 | `configs/task/hit_density_regression.yaml`, `NodeRegressionTask` | config/task integration + GH200 training smoke | Training path TARGET_VALIDATED in job `404059`; learning evidence pending |
| M8-vs-M9 held-out learning comparison | Project hypothesis | `docs/SCIENTIFIC_SPEC.md` §20, `docs/M10_HIT_SLICE_ABLATION.md` §7-9 | `scripts/train_slice_ablation.py` | matched one-epoch smoke, then frozen full runs; multiple seeds if difference is small | GH200 job `404059` passed; no learning-quality claim yet |
| Full-state straight Gaussian flow matching | Project adaptation of established flow matching | `docs/SCIENTIFIC_SPEC.md` §21, `docs/M11_FLOW_MATCHING.md` §1-3 | `FlowMatchingTask` | `tests/unit/test_flow_matching_task.py` + real GPU smoke required | Implemented M11; software/target validation pending |
| Gaussian source sampled after train-only standardization | Project numerical/scientific convention | `docs/SCIENTIFIC_SPEC.md` §21, `docs/M11_FLOW_MATCHING.md` §2 | `fit_train_standardizers` + `FlowMatchingTask.make_training_problem` | flow-path/scaling tests + target smoke required | Implemented M11; validation pending |
| One raw scalar flow time per physical graph | Project minimal conditioning baseline | `docs/SCIENTIFIC_SPEC.md` §21, `docs/M11_FLOW_MATCHING.md` §4 | `FlowMatchingTask` + existing M8/M9 graph conditioning | exact graph-time/path unit test | Implemented M11; richer time embeddings deferred |
| Deterministic sample-ID-keyed flow validation and generation source | Project reproducibility convention | `docs/M11_FLOW_MATCHING.md` §7-8 | `FlowMatchingTask` | batch-order invariance and sampler tests | Implemented M11; target validation pending |
| Euler/Heun flow ODE sampling | Established explicit integration + project adaptation | `docs/SCIENTIFIC_SPEC.md` §21, `docs/M11_FLOW_MATCHING.md` §8 | `FlowMatchingTask.sample_standardized` | analytical Heun test + target generation smoke required | Implemented M11; adaptive solvers deferred |
| HIT-slice flow-matching training/generation runner | Project experiment integration | `docs/M11_FLOW_MATCHING.md` §9-10 | `scripts/train_slice_flow_matching.py`, M11 configs | config tests + real GPU smoke required | Implemented M11; target evidence pending |
| Per-channel flattened empirical Wasserstein-1 generation diagnostic | Project diagnostic | `docs/M11_FLOW_MATCHING.md` §10 | `_marginal_generation_metrics` | exact empirical W1 unit test | Implemented M11; explicitly not spatial/joint quality evidence |
| Full-state discrete cosine diffusion with epsilon prediction | Project adaptation of DDPM + improved cosine schedule | `docs/SCIENTIFIC_SPEC.md` §22, `docs/M13_DIFFUSION.md` | `DiffusionDenoisingTask.make_training_problem` | `tests/unit/test_diffusion_task.py`; full software gate + GPU smoke required | Implemented M13; software/target validation pending |
| One normalized discrete diffusion time per physical graph | Project controlled-comparison convention | `docs/SCIENTIFIC_SPEC.md` §22, `docs/NUMERICAL_CONVENTIONS.md` §22 | `DiffusionDenoisingTask` + existing graph conditioning path | exact forward-process/time-conditioning test | Implemented M13; richer time embeddings intentionally deferred |
| Generalized DDIM eta with ancestral-DDPM full-grid limit | Project adaptation of DDIM | `docs/SCIENTIFIC_SPEC.md` §22, `docs/NUMERICAL_CONVENTIONS.md` §22 | `DiffusionDenoisingTask.sample_standardized`, `sampler_name` | sampler-label and clean-state-independence tests + GPU generation smoke required | Implemented M13; target validation pending |
| Standalone diffusion training/generation split | Project reproducibility/experiment design | `docs/M13_DIFFUSION.md`, `docs/REPRODUCIBILITY.md` §12 | `scripts/train_slice_diffusion.py`, `scripts/generate_slice_diffusion.py` | saved-standardizer reload test + smoke required | Implemented M13; generation runtime pending target validation |
| Explicit unpaired generated/reference identifiers | Project generative-evaluation contract | `docs/SCIENTIFIC_SPEC.md` §22, `docs/M13_DIFFUSION.md` | diffusion generation artifact + generation benchmark compatibility | diffusion-generation tests + legacy benchmark compatibility test | Implemented M13; old flow artifacts remain supported through legacy `sample_ids` |

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
    TARGET_VALIDATED on one NVIDIA GH200 in jobs 400187 and 400194;
    see docs/M8_SPARSE_TRANSFORMER.md §17
Known limitations:
    no coordinate/edge geometry, no periodic cross-boundary HIT augmentation, no fused sparse kernel,
    no CUDA low-precision or multi-node performance evidence yet
```

## 6. M9 relative-geometry genealogy

```text
Concept:
    learned additive attention-score bias from source-minus-target relative displacement
Status:
    project adaptation
References:
    Vaswani et al., Attention Is All You Need, NeurIPS 2017, arXiv:1706.03762
    Pfaff et al., Learning Mesh-Based Simulation with Graph Networks, ICLR 2021, arXiv:2010.03409
Project-specific definition:
    delta_r_ij = r_j - r_i for directed source j -> target/query i;
    b_ij = MLP(delta_r_ij) gives one additive scalar per attention head;
    s_ij^h = q_i^T k_j / sqrt(d_h) + b_ij^h.
Repository modification:
    relative displacement only; no explicit distance, absolute position, normalized direction,
    value-message geometry, self-loop/topology augmentation, or rotation-equivariant tensor machinery.
Implementation path:
    src/graph_attention/geometry/relative.py
    src/graph_attention/models/geometric_transformer.py
Configuration:
    configs/model/geometric_sparse_transformer.yaml
Scientific tests:
    displacement sign/reverse-edge relation
    translation invariance
    explicit geometric-attention reference
    geometry sensitivity
    node-renumbering equivariance
    disconnected packed versus independent execution
Numerical tests:
    invalid shape/dtype/index/non-finite coordinate failures
    inherited M8 stable sparse softmax
Benchmark evidence:
    TARGET_VALIDATED on one NVIDIA GH200 in Slurm job 403939, FP32,
    clean SHA 83e160846badb151eef0a09c2f1e2234da22bc24;
    S3 forward/training medians 3.4138/90.9458 ms;
    real HIT forward/training medians 5.7989/18.7044 ms.
Known limitations:
    no rotation/reflection/scale equivariance claim, no periodic cross-boundary HIT augmentation,
    no explicit-distance ablation, no fused sparse-geometric kernel, no CUDA low-precision/multi-node evidence
```

## 7. M10 HIT-slice learning-ablation genealogy

```text
Concept:
    controlled learning comparison of topology-only M8 versus relative-geometry M9
Status:
    project hypothesis
Source artifact:
    diffusion4avbp precomputed fixed Cartesian 2-D slice dataset
Scientific definition:
    task [rhou, rhov, rhow, rhoE] -> rho;
    split groups are authoritative metadata.source_stem values;
    stored shared 2-D coordinates are preserved directly as model geometry;
    extraction axis/slice-coordinate metadata are provenance only, not model inputs;
    M9 uses canonical 2-D relative displacement with spatial_dim=2;
    topology is non-periodic bidirectional Cartesian 4-neighbour connectivity.
Repository modification:
    adds the data adapter, deterministic slice geometry, grouped split generation,
    experiment configuration, and dedicated first-ablation training runner;
    M8/M9 equations and M6 loss/scaling are reused unchanged.
Implementation path:
    src/graph_attention/data/slice_pt.py
    src/graph_attention/data/splits.py
    src/graph_attention/geometry/cartesian.py
    scripts/train_slice_ablation.py
Configuration:
    configs/data/hit_slice_pt.yaml
    configs/task/hit_density_regression.yaml
    configs/ablation/hit_slice.yaml
Scientific tests:
    metadata-based grouping
    exact shared 2-D coordinate preservation
    extraction-orientation independence of model coordinates
    grouped split zero-overlap
    exact Cartesian topology reference
Experiment evidence:
    real artifact preflight passed on 1,488 slices;
    grouped split 260/55/57 source groups with zero overlap;
    one-epoch matched M8/M9 GH200 training smoke passed in job 404059.
Known limitations:
    canonical 2-D geometry is not a global 3-D directional frame;
    periodic wrap edges and explicit slice-orientation conditioning are deferred;
    first runner is single-process FP32;
    one seed is insufficient for a robustness claim if M8/M9 differences are small
```

## 8. M11 HIT-slice flow-matching genealogy

```text
Concept:
    unconditional straight-path flow matching of the full five-channel HIT slice state
Status:
    project adaptation of established flow matching and straight-flow ideas
References:
    Lipman et al., Flow Matching for Generative Modeling, ICLR 2023, arXiv:2210.02747
    Liu et al., Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow,
    ICLR 2023, arXiv:2209.03003
Project-specific definition:
    physical nondimensionalization -> train-only sample-balanced standardization gives x1;
    x0 ~ N(0,I), t_g ~ U(0,1);
    x_t = (1-t_g)x0 + t_g*x1;
    v* = x1 - x0;
    raw scalar t_g is appended as one graph-level conditioning channel.
Repository modification:
    reuses NodeRegressionTask preparation, M6 scaling/loss/optimizer semantics,
    M10 grouped split/topology, and frozen M8/M9 backbones;
    adds flow-path construction, deterministic validation/sampling RNG,
    Euler/Heun ODE sampling, generation artifacts, and a dedicated runner.
Implementation path:
    src/graph_attention/tasks/flow_matching.py
    scripts/train_slice_flow_matching.py
Configuration:
    configs/task/hit_flow_matching.yaml
    configs/generative/hit_slice_flow_matching.yaml
Scientific tests:
    exact straight interpolation/velocity target
    one graph time broadcast to all its nodes
    sample-ID deterministic validation independent of batch order
    no clean-state leakage into deterministic sampling source
Numerical tests:
    analytical Heun integration for v=t
    exact empirical marginal Wasserstein-1 diagnostic
Experiment evidence:
    pending full software gate and real M9 GPU smoke
Known limitations:
    raw scalar time only; no AdaLN/Fourier time embedding;
    no nonlinear/OT coupling, diffusion task, periodic topology, DDP, low precision,
    adaptive ODE solver, or spatial/spectral/joint generation-quality evidence yet
```

## 9. M13 HIT-slice diffusion genealogy

```text
Concept:
    unconditional discrete diffusion of the full five-channel HIT slice state
Status:
    project adaptation of established DDPM/DDIM methods
References:
    Ho et al., Denoising Diffusion Probabilistic Models, NeurIPS 2020, arXiv:2006.11239
    Nichol and Dhariwal, Improved Denoising Diffusion Probabilistic Models,
    ICML 2021, arXiv:2102.09672
    Song et al., Denoising Diffusion Implicit Models, ICLR 2021, arXiv:2010.02502
Project-specific definition:
    physical nondimensionalization -> train-only sample-balanced standardization gives x0;
    T=1000, cosine schedule with s=0.008;
    one integer t_g in [1,T] per graph and epsilon ~ N(0,I);
    x_t = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*epsilon;
    epsilon prediction with equal-sample MSE;
    normalized scalar tau=t/T is appended through the existing graph conditioning path.
Repository modification:
    adapts the diffusion process to packed disconnected graphs without modifying M9/M12;
    adds deterministic sample-keyed validation noise, generalized DDIM eta sampling,
    standalone train/generate workflows, saved-standardizer reload, exact test-split replay,
    and explicit unpaired generated/reference IDs.
Implementation path:
    src/graph_attention/tasks/diffusion.py
    scripts/train_slice_diffusion.py
    scripts/generate_slice_diffusion.py
    src/graph_attention/evaluation/generation_benchmark.py
Configuration:
    configs/task/hit_diffusion.yaml
    configs/generative/hit_slice_diffusion.yaml
    configs/generate_slice_diffusion.yaml
Scientific tests:
    exact forward noising equation and one normalized graph time
    deterministic validation independent of batch order
    generation independence from clean held-out state values
    explicit generated/reference population-ID semantics
Numerical tests:
    eta/sampler-limit classification
    persisted standardizer reload without refitting
    legacy generation-benchmark artifact compatibility
Experiment evidence:
    software gate and Calypso/GH200 training/generation smoke pending
Known limitations:
    no SNR weighting, v/x0 prediction, learned variance, clipping/thresholding,
    richer time embedding, DDP, low-precision target validation, or conditional generation yet;
    no learning-quality claim before a controlled trained diffusion run is benchmarked
```

## 9.1. Full-volume 3-D HIT generation benchmark genealogy

```text
Concept:
    unpaired full-volume 3-D distribution and spectral evaluation for generated HIT
Status:
    project scientific/numerical benchmark convention
Project-specific definition:
    generated and held-out test populations are compared in nondimensional conservative state;
    the endpoint-inclusive 33^3 periodic HIT storage grid is reduced to the unique 32^3
    periodic grid only for FFT analysis by dropping the maximum-coordinate plane on x/y/z;
    3-D orthonormal FFT radial shells retain 0 < k <= min-axis Nyquist;
    velocity energy uses u=rhou/rho, v=rhov/rho, w=rhow/rho with per-snapshot means removed.
Implementation path:
    src/graph_attention/evaluation/spectra_3d.py
    src/graph_attention/evaluation/generation_benchmark_3d.py
    scripts/benchmark_generation_3d.py
Configuration:
    configs/benchmark_generation_3d.yaml
Scientific/numerical tests:
    permuted-node structured 3-D grid reconstruction with periodic endpoint removal;
    resolved periodic velocity mode recovers the expected integrated specific TKE.
Evidence status:
    IMPLEMENTED; target-cluster end-to-end generation/benchmark validation pending.
Known limitations:
    requires a complete uniform Cartesian 3-D mesh for FFT diagnostics;
    periodic endpoint removal is a post-processing convention only and does not repair
    the currently deferred periodic cross-boundary graph topology;
    true full-volume ray-cast qualitative field plots are now enabled; see docs/M40_3D_VOLUME_BENCHMARK.md.
```

## 9.2. Full-volume 3-D HIT spatial-statistics genealogy

```text
Concept:
    real-space validation of generated 3-D HIT beyond one-point PDFs and spectra
Status:
    project scientific benchmark convention
Project-specific definition:
    velocity is reconstructed from conservative variables;
    periodic axis-aligned separations on the unique 32^3 HIT grid are averaged across x/y/z;
    report longitudinal/transverse/vector correlations;
    report S2 longitudinal, S2 vector, S3 longitudinal, S4 longitudinal, and longitudinal flatness;
    report population mean and 10-90% snapshot bands;
    qualitative inference examples show central x/y/z slices for a few unpaired generated/test volumes.
Implementation path:
    src/graph_attention/evaluation/spatial_statistics_3d.py
    src/graph_attention/evaluation/plotting_3d.py
    src/graph_attention/evaluation/generation_benchmark_3d.py
Configuration:
    configs/benchmark_generation_3d.yaml
Validation:
    analytic periodic sinusoid test for longitudinal correlation, S2, S3, and normalized separation;
    benchmark Slurm gate checks produced CSVs and plots.
Evidence status:
    IMPLEMENTED; target-cluster benchmark rerun pending.
Known limitations:
    spatial statistics use Cartesian-axis separations rather than a full radial all-pairs estimator;
    the current 32^3 grid offers only limited inertial-range scale separation;
    snapshot panels are explicitly unpaired qualitative examples.
Detailed equation-level physics and implementation-specific limits:
    docs/3D_SPATIAL_STATISTICS_PHYSICS.md
```

## DiNAT-DiT self-attention ablation

```text
Concept:
    add the center/self token to both local and exact-two-hop DiNAT-DiT attention windows
Implementation:
    src/graph_attention/models/dinat_dit.py
    configs/model/dinat_dit.yaml
    src/graph_attention/training/model_factory.py
Semantics:
    local attention = self union one-hop
    dilated attention = self union exact-two-hop
    raw exact-two-hop geometry remains self-free
Backward compatibility:
    constructor default is false; historical resolved configs therefore keep no-self behavior
Default:
    configs/model/dinat_dit.yaml keeps include_self_attention=false during the ablation stage
M39 override:
    model.include_self_attention=true
Validation:
    unit test checks exactly one self edge per node in both layer types and zero self displacement
2-D ablation:
    M39 uses the M33/M34 2-D HIT Flow Matching setup with h128 L10 H4, batch 128, seed 42,
    scatter backend matched to M33, 1000 epochs, and self attention enabled
```

## 10. Change procedure

When code changes a scientifically meaningful mechanism:

1. update the scientific definition first or in the same change;
2. update this traceability table;
3. update or add scientific tests;
4. record the evidence status;
5. do not mark a performance or scientific hypothesis as established merely because tests pass.

## 11. Citation policy

When a mechanism comes from literature, preserve enough citation detail to identify the exact source and idea used.

Do not attribute a project-specific synthesis to one paper unless that paper actually contains the claimed formulation.

When several ideas are combined, record the genealogy explicitly.


## M40: 3-D self plus local, exact-two-hop unchanged

- Control: M37 3-D DiNAT-DiT Flow Matching (h128, L10, H4, B16, seed 42,
  1000 epochs, torch_sparse).
- Intervention: local one-hop attention includes exactly one (i,i) self-edge
  per node in even layers; exact-two-hop dilated layers **exclude** (i,i).
- New config override: `+model.include_local_self_attention=true` and
  `model.include_self_attention=false`. Former M39 include_self_attention
  behaviour is preserved.
- GPU preflight: `scripts/probe_3d_max_batch.py --model dinat
  --include-local-self-attention --max-batch 16`; requires passing B16.
- Training: `scripts/slurm/m40_3d_dinat_local_self_flow_train.slurm`
- Full 3-D generation: `scripts/slurm/m40_3d_dinat_local_self_flow_generate.slurm`
- M37/M40 3-D quality benchmark:
  `scripts/slurm/m37_m40_3d_local_self_quality_compare.slurm`
- Unit tests assert 3-D local self-loops, zero self-relative displacement,
  preserved self-free exact-two-hop topology, init metadata, and backward after
  sparse cached inference.


## 9.3. True 3-D full-volume rendering genealogy

- Purpose: qualitative visual comparison of 3-D HIT conservative scalar fields
  from unpaired DNS, M37 and M40 populations. No paired reconstruction claim.
- Scientific definition: trilinear emission/absorption ray casting through all
  32^3 unique periodic voxels, **not** central slices or three intersecting planes.
- Colormap: RdBu_r, matching 2-D and existing 3-D slice diagnostics.
- Cross-population control: one jointly computed 1%-99% scalar range per
  variable and one identical opacity transfer, camera, and ray integration
  scheme for DNS, M37, M40.
- Implementation:
  src/graph_attention/evaluation/volume_rendering_3d.py;
  scripts/plot_m37_m40_3d_volumes.py;
  src/graph_attention/evaluation/generation_benchmark_3d.py.
- Configuration: configs/benchmark_generation_3d.yaml, plots.volume_3d and
  plots.volume_resolution, plots.volume_steps, plots.volume_optical_depth.
- Validation: tests/unit/test_volume_rendering_3d.py checks responsiveness
  to an off-plane interior voxel and matched-population plot creation.
- Launch: scripts/slurm/m37_m40_3d_local_self_quality_compare.slurm.
- Documentation: docs/M40_3D_VOLUME_BENCHMARK.md.
- Status: implemented in main; target-cluster rendering/test execution pending.

### M41.1 validation hardening

Software-only executor/cache hardening and benchmark instrumentation; no change
to attention equations, training objectives or sampling conventions.
`tests/unit/test_partitioned_dinat.py` covers directed/empty neighborhoods, CSR,
inference caches, spatial-bin compaction and actual 50-step Heun parity.
`scripts/benchmark_m41_partitioned.py` gates the pretrained checkpoint at every
layer and after generation. **TARGET_VALIDATED** on Calypso GH200, 2026-10-12,
Slurm job `461023`, clean commit `2c28a89`: 47 selected tests passed, one
optional DGL test skipped; all five 1/2/4/8/16-block layouts passed against
real M37 `best.pt` at FP32, `atol=rtol=2e-4`, with preserved 209,088 local and
608,058 exact-two-hop edges, no tolerance violations, and identical initial
Heun noise / 100 calls. Largest across-layout per-layer/output/Heun absolute
errors: 5.943e-5 / 5.317e-5 / 1.907e-6. At 16 partitions the median forward
was 232.88 ms versus 31.50 ms unpartitioned (+639%); peak allocated GPU memory
was 0.216 GiB versus 0.440 GiB. The full repo suite was not part of this job.
See `docs/M41_1_HALO_PARTITIONED_DINAT.md` for full data and
`docs/M41_1_HALO_PARTITIONING.md` for the numerical contract.

### M41.2 owned-query single-GPU halo optimization (GH200 target-validated)

- Scientific contract: retain M37's original global local and exact-two-hop
  source/target edges and per-target softmax. Each layer computes nodewise
  normalization, QKV, attention output projection and MLP exactly once over
  the global resident graph; only the sparse aggregation operates on owner
  query nodes with full read/halo key/value neighborhoods.
- Design: original M41.1 ownership plan is retained, while a rectangular
  `|owned| x |owned+halo|` CSR pattern replaces the unused square cached
  topology. No M37 training, checkpoint, objective or sampler modifications.
- Implementation: `src/graph_attention/models/owned_query_dinat.py`;
  opt-in `--executor owned_query` in
  `scripts/benchmark_m41_partitioned.py`. The default remains
  `layerwise_full` for validated M41.1.
- Tests: `tests/unit/test_owned_query_dinat.py` covers forward and
  intermediate-layer equivalence, all self-loop modes, directed/empty
  rows, inference-cache tensors, immutable weights, CUDA CSR and 50-step
  Heun. Global nodewise modules are checked for exactly one execution
  per layer. `tests/unit/test_m41_2_summary.py` checks A/B pairing.
- Experiment: `scripts/slurm/m41_2_owned_query_m37.slurm` compares both
  executors, each against the original M37 checkpoint, on matched held-out
  35,937-node graphs and 1/2/4/8/16 spatial partitions. Output includes
  per-layer/Heun errors and paired runtime/peak-memory CSV.
- Current evidence: **TARGET_VALIDATED numerical equivalence and measured
  single-GPU performance**, 2026-10-12, Slurm job `461158`, clean
  commit `f3d9e5bf`, 68 tests passed (one optional DGL test skipped),
  all five M41.1/M41.2 layout pairs passing the real M37 checkpoint,
  50-step Heun and ten intermediate-layer comparisons under FP32
  `atol=rtol=2e-4`. Maximum M41.2 layer/output/Heun absolute errors:
  `5.889e-5 / 4.444e-5 / 1.907e-6`. At 16 partitions, owned-query
  forward is 126.44 ms versus 238.88 ms for M41.1 (1.889x speedup),
  while peak allocated memory rises 0.216 -> 0.377 GiB (+74%);
  unpartitioned M37 forward was 30.76 ms. Correctness does not
  imply efficient high-partition execution. The validated M41.1
  baseline is job `461023` (commit `2c28a89`).
- Follow-up profiling: **TARGET-EXECUTED** on GH200, Slurm job
  `461247`, 2026-10-12, clean main `eb1935f2`; 68 selected tests
  passed/one optional DGL skipped, 1 and 16 partition layouts both
  passed checkpoint and Heun equivalence. At 16 partitions, M41.2
  forward 199.56 ms vs M41.1 307.82 ms (1.542x speedup), but
  M37 42.16 ms. Operator profile: 640 sampled-CSR matmuls
  (10 layers x 16 partitions x 4 heads), 160 scatter reductions,
  3,841 copies and 6,512 `empty` allocations. M41.2 sampled-CSR
  reports ~12.96 ms PyTorch self CPU time vs ~2.82 ms sampled-CSR
  GPU time, supporting launch/dispatch fragmentation as a primary
  target. Operator CPU and GPU event times must not be summed.
  The previous run `461158` had a faster M37 reference, so
  cross-run comparisons should not be treated as clock-normalized.
  Full evidence and proposed experimental CUDA Graph replay:
  `docs/M41_2_OWNED_QUERY_ATTENTION.md`.

### M41.3 single-GPU CPU-host-streamed layerwise DiNAT (GH200 verified, large mesh pending)

- Purpose: the original M41.1/M41.2 single-GPU partitioned executors
  still retain global transformer hidden states on GPU. M41.3 keeps
  **two complete hidden-state buffers on CPU** and evaluates only
  one owned-plus-complete-halo partition at a time on the GPU.
- Mathematics: trained M37 unchanged, directed incoming 1-hop and
  exact-two-hop neighborhoods with original geometric bias and
  optional self loops; at each layer, every partition reads the
  immutable previous-layer host buffer, writes **owned nodes only**
  to a distinct next-layer host buffer; the next layer begins after
  all owned updates complete.
- Code: `src/graph_attention/models/streamed_dinat.py` with CPU-only
  `StreamedDiNATPlan` metadata, original M41.2 rectangular CSR,
  partition-local QKV/attention/MLP and CPU streaming. Physical output
  returns to input device for the original Flow Matching Heun sampler.
- Tests: `tests/unit/test_streamed_dinat.py` for all self-loop modes,
  spatial/interleaved ownership, directed/empty graphs, intermediate
  and final output parity, trained-weight immutability, CPU topology
  residency and 50-step Heun parity; CUDA partition size checks.
  `tests/unit/test_m41_3_summary.py` enforces report provenance
  and does not mark failed numerical gates as performance successes.
- GH200 experiment: `scripts/slurm/m41_3_host_streamed_m37.slurm`
  compares opt-in `--executor streamed` versus validated M41.2,
  using the existing reference M37 checkpoint and 100-evaluation
  Heun run. Strict paired comparison in
  `scripts/summarize_m41_3.py`.
- Current status: **GH200 M37 numerical parity PASSED on 2026-10-12**
  (job 461341; 5 partition layouts; layer, output, 50-step Heun).
  P16 forward medians: M41.2 161.96 ms, M41.3 348.08 ms;
  peak allocated GPU M41.2 0.377 GiB, M41.3 0.119 GiB.
  P16 output max error 4.07e-5, Heun max error 9.54e-7.
  The reference benchmark retains its original graph and data on GPU;
  these allocations do not establish a full `257^3` peak.
  Large-mesh graph construction and CPU graph-memory scaling remain
  unvalidated. Evidence:
  <https://drive.google.com/drive/folders/1u3ASJVKPZ9UQ0ic0QzL9eUvSBqBM4kTa>.
  Design and benchmark: `docs/M41_3_HOST_STREAMED_DINAT.md`.
- Large-mesh dataset uploaded in Drive: `Re150/Ma0.3` mesh
  `mesh.mesh.h5` (1.68 GB) and `solut_hit_00000001.h5` (2.65 GB).
  Added `scripts/preflight_m41_3_large_hdf5.py`,
  `tests/unit/test_m41_3_large_hdf5_preflight.py`, and
  `scripts/slurm/m41_3_257cube_preflight.slurm`: CPU-only HDF5
  layout, field, index-base and memory preflight, without constructing
  full global edge or two-hop tensors. End-to-end 257^3 inference
  is **not yet implemented or validated**.

### Standalone global Physics-Token DiT — Transolver-style learned states (pilot pending)

- Purpose: test *global learned physical-state tokens* as a new, independent
  Flow Matching architecture before considering any hybrid with DiNAT-DiT.
  This is **not** an M37/M41 variant; existing model weights and M42/dnstoles
  remain unchanged. Source: Transolver (Wu et al., ICML 2024,
  arXiv:2402.02366), irregular-mesh Physics-Attention.
- Model: `src/graph_attention/models/physics_token_dit.py` defines
  `PhysicsTokenMultiheadAttention` and
  `PhysicsTokenDiTGraphTransformer`. Each head learns soft assignments
  of physical nodes to `num_tokens` (default 64), pools globally
  weighted state tokens, applies attention among tokens, and deslices
  back to each original node. No graph edges in attention.
- Fair controlled initialization: matched existing Full DiT
  non-attention backbone tensors under seed 42, independent
  physics-token attention initialization seed 43. Preserve existing
  AdaLN-Zero, Flow Matching time scale, physical standardization and
  original training/generation runners. Hydra configuration:
  `configs/model/physics_token_dit.yaml`.
- Optional `token_chunk_size` recomputes slice assignments in
  two passes for smaller GPU attention workspace; **not** whole-graph
  GPU hidden-state streaming. Chunked/un-chunked forward and gradient
  parity are in `tests/unit/test_physics_token_dit.py`, plus graph
  isolation, permutation equivariance, formula, 50-step Heun smoke,
  invalid configs and factory seeding.
- GH200 pilot: `scripts/slurm/physics_token_dit_3d_train.slurm`,
  identical 3-D HIT task/data/split, `H128/L10/4 heads`, global
  tokens 64, 10 epochs by default, microbatch 4 and accumulation 4
  (nominal batch 16). A 10-epoch pilot is **not** a convergence-matched
  quality comparison against 1000-epoch M37/M38; extend to 1000
  epochs after tests and memory validation.
- Later checkpoint generation (unchanged 50-step Heun):
  `scripts/slurm/physics_token_dit_3d_generate.slurm`.
- Pilot run 461354 (GH200, 2026-10-12; repository
  `79a525faba21f0c1ef6980080af98abebf5fbcfa`) stopped at its
  pretraining pytest gate: **30 passed, 2 failed**. The physics-token
  batch-isolation test incorrectly supplied noncontiguous graph labels,
  whereas the shared DiT contract requires contiguous packed graphs.
  A pre-existing config regression test incorrectly demanded exact
  equality with M37's attention-specific YAML options
  `include_self_attention` and `sparse_attention_backend`.
  Both **tests only** were corrected, and the unsupported interleaved
  layout now has a rejection regression test. Neither architecture
  nor model weights changed.
- GH200 job **462860**, started 2026-10-13 with commit
  `6b65f01aafcf55019f18693e3f7871084708b123`:
  **32 regression tests passed**. Standalone Physics-Token DiT
  completed its configured 10-epoch pilot (epochs 0–9) without
  failure. Best validation Flow Matching velocity MSE **1.42555245**
  at epoch 9, held-out test MSE **1.46680478**; 2,938,285 parameters.
  Mean epoch compute time **85.7592 s**, fit wall **859.527 s**,
  peak GPU allocated **22.1048 GiB**, reserved **22.7324 GiB**.
  Data: 35,937 nodes/sample, 1,262 samples (883/189/190
  train/validation/test), microbatch 4, gradient accumulation 4,
  64 tokens/head, 10 layers, hidden 128, four heads, FP32.
  The validation loss fell from 1.75708 to 1.42555, with slower
  improvement after epoch 3. This validates training integration,
  **not** convergence or generated-field spectral quality.
- Status: **10-EPOCH GH200 TRAINING PILOT PASSED**.
  No generative spectra or `257^3` scaling claims.
  Run directory:
  `/scratch/coop/theret/GraphAttention_runs/physics_token_3d_flow_h128_l10_h4_g64_b4_acc4`.
  Further steps: optional 50-step Heun generation smoke on this
  checkpoint and resume same configuration to 100 epochs before
  assessing a 1000-epoch matched convergence comparison.
  Full design/limitations: `docs/PHYSICS_TOKEN_DIT_BASELINE.md`.

### M41.3 257^3 real-mesh topology gate (GH200 job 464614)

- Metadata preflight run **464614** passed **10/10** tests on
  `calypso-grace04`, 2026-10-14. HDF5 mesh has 16,974,593
  float64-coordinate nodes and 16,777,216 hexahedral cells. Flattened
  uint64 connectivity has 134,217,728 entries, globally **one-based**
  with min=1 and max=16,974,593; all five conservative fields are
  float64 and node-aligned. No graph or inference was constructed.
- The strict follow-up in
  `scripts/qualify_m41_3_cartesian_mesh.py` verifies a bijective
  uniform-coordinate lattice and every cell's canonical 12 M37 local
  hex edges, processing the HDF5 in bounded slabs. Only if complete
  qualification passes does it persist the two Cartesian permutation
  maps and permit the direct Cartesian candidate.
- `src/graph_attention/geometry/cartesian_halo.py` builds owned
  plus halo one-hop/exact-two-hop directed edges per partition, using
  the proven Cartesian inverse mapping instead of global edge tensors.
  The candidate graph edges are compared to a hex-derived oracle
  with randomized node ordering in
  `tests/unit/test_m41_3_cartesian_mesh.py`.
- Next GH200 verification:
  `scripts/slurm/m41_3_257cube_qualify.slurm`, requesting Grace,
  one GPU, four CPUs and four hours with **no explicit memory cap**.
  Its CPU-only qualification must pass before integration into the
  M41.3 CPU-streamed CSR/geometry plan.
- Status: **metadata verified; 257^3 topology qualification NOT YET RUN**.
  Complete details: `docs/M41_3_257CUBE_TOPOLOGY_GATE.md`.

### M41.4 — Generic unstructured halo CSR before 257^3 inference

- Objective: retire the full-graph one-hop-plus-two-hop construction
  requirement in M41.3 without specializing the scientific framework to
  Cartesian HIT. Keep original M37 weights and exact one-hop /
  exact-two-hop definitions, owned-query softmax and layerwise
  previous-layer Jacobi refresh.
- Implemented prototype:
  src/graph_attention/geometry/external_csr.py builds a one-hop incoming
  CSR by streaming hex/tet HDF5 element connectivity and externally
  deduplicating target buckets on disk. A generic directed edge-block
  interface handles graph inputs with other supported edge sources.
  src/graph_attention/geometry/partition_csr.py extracts only per-owned
  one-hop/exact-two-hop edges with optional self-loop policy, works with
  any contiguous owner assignment. No global exact-two-hop graph.
- Adapter: src/graph_attention/models/disk_streamed_dinat.py
  constructs the original OwnedQueryPartition/CSR and StreamedDiNATPlan
  interfaces from partition files. Unchanged M41.3 forward executor;
  on-demand disk or CPU-RAM preloading.
- Regression gate: tests/unit/test_m41_4_generic_csr.py verifies
  irregular shuffled hexahedral and tetrahedral input graph
  construction, exact-two-hop exclusion/deduplication and full M37
  output parity including self-loop variants; additionally verifies
  complete numerical forward parity on a truly unstructured tet mesh.
- Grace stages:
  scripts/slurm/m41_4_generic_257_prepare.slurm runs the unstructured
  gate BEFORE touching 257^3, then builds CSR/partitions from the real
  one-based 257^3 HDF5 mesh; requires one Grace GPU, four CPUs, four
  hours, no explicit --mem. Next:
  scripts/slurm/m41_4_generic_257_infer.slurm first performs one
  full-resolution forward, then optionally a resumable 50-step
  Heun generation using 100 NFEs.
- **Status updated 2026-10-14: real 257^3 full forward PASSED
  on Grace job 464853.** The inferred generic partitions were successfully
  loaded; the uploaded report has no end-to-end CSR preparation timing.
  Peak GPU memory 2.2552 GiB allocated, 7.6797 GiB reserved,
  forward time 158.2297 s, output [16,974,593,5] finite.
  CPU RSS was not recorded in that run; later runner versions measure it.
  Original M41.3 small-graph parity remains unchanged.
  No same-size unpartitioned M37 reference exists; this demonstrates
  executable scalability, not 257^3 M37 numerical parity or physical quality.
  Details: docs/M41_4_GENERIC_UNSTRUCTURED_HALO.md.
