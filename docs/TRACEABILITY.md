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

Record:

- reference;
- relevant equation/section;
- deviations in this repository.

### Project adaptation

A known method has been changed for this repository or CFD setting.

Record:

- source method;
- exact adaptation;
- reason;
- validation.

### Project hypothesis

A new or unvalidated research idea.

Record:

- hypothesis;
- intended mechanism;
- ablation/comparison needed;
- current evidence status.

### Measured result

An empirical result obtained from a specified experiment or benchmark.

Record enough context to reproduce it.

## 3. M0 traceability table

The implementation/test paths below are placeholders until the corresponding code exists. They should be updated rather than replaced with parallel tracking systems.

| Concept | Type | Specification | Implementation | Validation | Evidence status |
|---|---|---|---|---|---|
| CFD field + mesh + task as core abstraction | Project design | `README.md`, `docs/SCIENTIFIC_SPEC.md` | Architecture-wide | Documentation review | Frozen M0 |
| Graph vertices represent CFD mesh nodes | Project design | `docs/SCIENTIFIC_SPEC.md` §3 | `src/.../data`, `src/.../geometry` | schema tests | Frozen M0 |
| Native mesh connectivity is first-class input | Project design | `docs/SCIENTIFIC_SPEC.md` §2–4 | `src/.../geometry` | connectivity tests | Frozen M0 |
| Packed disconnected variable-graph batching | Established graph batching pattern + project requirements | `docs/ARCHITECTURE.md` §5 | `src/.../data/collate.py` | variable-size packed-batch tests | Planned |
| Node/edge computational budgets | Project systems design | `docs/ARCHITECTURE.md` §6, `docs/NUMERICAL_CONVENTIONS.md` §4 | sampler/packer | boundary/oversize tests | Planned |
| Statistical batch independent of microbatch composition | Project numerical requirement | `docs/NUMERICAL_CONVENTIONS.md` §5 | trainer/task loss aggregation | gradient-equivalence tests | Planned |
| Physical quadrature-weighted per-sample loss | Established numerical integration principle | `docs/NUMERICAL_CONVENTIONS.md` §6 | task/loss code | reference integration tests | Planned |
| DDP global sample-weight consistency | Project distributed-training requirement | `docs/NUMERICAL_CONVENTIONS.md` §7 | trainer | multi-rank equivalence test | Planned |
| Field catalogue with semantic roles | Project data contract | `docs/SCIENTIFIC_SPEC.md` §5–7 | data schemas/readers | field-catalogue tests | Planned |
| Task-specific named channel selection | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §6 | task/data interface | ordering/provenance tests | Planned |
| Stored vs derived field provenance | Project scientific contract | `docs/SCIENTIFIC_SPEC.md` §7 | preprocessing | provenance tests | Planned |
| Physical nondimensionalization before statistical scaling | Established scientific-ML convention + project rule | `docs/SCIENTIFIC_SPEC.md` §8–12, `docs/NUMERICAL_CONVENTIONS.md` §8–10 | preprocessing | transform round-trip/reference tests | Planned |
| Inference-available reference quantities only | Project anti-leakage rule | `docs/SCIENTIFIC_SPEC.md` §9 | preprocessing/task validation | leakage validation tests | Planned |
| Explicit resolution descriptor | Project multiresolution requirement | `docs/SCIENTIFIC_SPEC.md` §12 | geometry/preprocessing | resolution metadata tests | Planned |
| Node-renumbering equivariance | Fundamental graph-model property | `docs/SCIENTIFIC_SPEC.md` §13 | every graph model | permutation test | Mandatory |
| Translation/rotation/etc. claims require explicit proof/test | Project scientific rule | `docs/SCIENTIFIC_SPEC.md` §14 | model-specific | property-specific tests | Per model |
| Data/Geometry/Task/Model/Trainer ownership | Project software-science design | `docs/ARCHITECTURE.md` §3 | architecture-wide | dependency/review tests | Frozen M0 |
| Performance evidence levels | Project engineering rule | `docs/BENCHMARK_PROTOCOL.md` | benchmark tooling | benchmark schema checks | Planned |

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
