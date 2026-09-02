# GraphAttention

## Purpose

GraphAttention is a scientific machine-learning framework for modelling CFD fields on meshes of varying size, resolution, and topology, with a focus on efficient sparse geometric neural architectures.

The framework is intended to support multiple learning paradigms without making any one of them the defining abstraction of the repository. Supported or planned problem classes include deterministic regression, wall-quantity prediction, super-resolution, operator learning, and conditional or unconditional generative modelling such as diffusion and flow matching.

The fundamental project abstraction is:

$$
\boxed{\text{CFD field} + \text{mesh geometry/topology} + \text{task}}
$$

The repository is CFD-specific. The model is mesh-aware but mesh-agnostic: mesh connectivity and geometry are first-class runtime inputs, while learned parameters must not depend on a particular mesh size, node numbering, or topology.

## Core design principles

1. **Scientific correctness first.** Mathematical and physical meaning must be explicit, testable, and traceable.
2. **Variable meshes are first-class.** A training run, and when computationally feasible a single packed microbatch, may contain meshes with different node counts, edge counts, resolutions, geometries, and topologies.
3. **Sparse computation is the default direction.** Padding variable graphs to common node or neighborhood sizes is not the canonical representation.
4. **Computational batching and statistical batching are distinct.** GPU packing decisions must not silently change the statistical weight of training samples.
5. **Physical nondimensionalization is separate from machine-learning scaling.** Reference definitions, field transformations, statistics, and provenance must be reproducible.
6. **Scientific roles remain explicit.** Physical state, geometry, resolution, global regime conditioning, diagnostics, and solver metadata must not be implicitly conflated.
7. **Clear ownership boundaries.** Data, geometry, task, model, and training code have distinct responsibilities.
8. **Performance is a first-class requirement.** Efficiency claims must be evidence-based and labelled according to the hardware and workload on which they were established.
9. **Node renumbering must not change physics.** Graph models must be equivariant to consistent node relabelling.
10. **No scientifically meaningful implementation exists only in code.** Important equations, assumptions, and transformations must be documented and mapped to implementation and validation.

## Initial graph representation

For the first version of the repository:

$$
\boxed{\text{one graph vertex} = \text{one CFD mesh node}}
$$

The native CFD mesh is preserved. Cells and faces may supply geometric or numerical metadata such as volumes, normals, boundary information, and quadrature weights, but they are not initially separate learned graph entities.

## Documentation

- [`docs/M0_FROZEN_SPEC.md`](docs/M0_FROZEN_SPEC.md): authoritative frozen M0 decisions.
- [`docs/M1_FOUNDATION.md`](docs/M1_FOUNDATION.md): technical repository foundation and completion gate.
- [`docs/M2_DATA_CONTRACTS.md`](docs/M2_DATA_CONTRACTS.md): core field, reference-scale, mesh, and sample runtime contracts.
- [`docs/M3_1_SYNTHETIC_DATA.md`](docs/M3_1_SYNTHETIC_DATA.md): deterministic non-physical variable-mesh data used to exercise the real data contracts.
- [`docs/M3_2_AVBP_HDF5.md`](docs/M3_2_AVBP_HDF5.md): named-field AVBP/HDF5 reading and native hexahedral connectivity handling.
- [`docs/M3_3_NONDIMENSIONALIZATION.md`](docs/M3_3_NONDIMENSIONALIZATION.md): M3.3 reference-state and physical nondimensionalization directive.
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): software boundaries, runtime representation, and dependency direction.
- [`docs/SCIENTIFIC_SPEC.md`](docs/SCIENTIFIC_SPEC.md): scientific scope, graph assumptions, field semantics, and required invariances.
- [`docs/NUMERICAL_CONVENTIONS.md`](docs/NUMERICAL_CONVENTIONS.md): nondimensionalization, normalization, loss weighting, precision, and batching conventions.
- [`docs/TRACEABILITY.md`](docs/TRACEABILITY.md): mapping from scientific concepts to code, tests, and experiments.
- [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md): run metadata and reproducibility requirements.
- [`docs/BENCHMARK_PROTOCOL.md`](docs/BENCHMARK_PROTOCOL.md): performance evidence levels and benchmark protocol.
- [`AGENTS.md`](AGENTS.md): mandatory rules for agents and contributors modifying the repository.

## Current status

M0-M2 are complete. M3.1 provides deterministic synthetic variable-mesh samples. M3.2 is complete and has been validated against a real AVBP HIT snapshot/mesh pair on Calypso. M3.3 has a frozen scientific directive for case-level reference states and physical nondimensionalization; implementation and numerical reference tests are next. Packed graph batching, tasks, and sparse attention remain later work.

Current smoke checks:

```bash
python -m pip install -e ".[dev]"
python scripts/inspect_config.py
pytest
ruff check .
ruff format --check .
```
