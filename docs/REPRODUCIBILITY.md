# Reproducibility

## 1. Purpose

A training or inference result is scientifically useful only if its data selection, preprocessing, model configuration, and runtime environment can be reconstructed.

This document defines the minimum reproducibility state that must accompany meaningful experiments.

## 2. Reproducibility principle

A checkpoint alone is not a complete experiment artifact.

A reproducible run must preserve enough information to reconstruct:

\[
\boxed{
\text{data}
+\text{field semantics}
+\text{physical preprocessing}
+\text{task}
+\text{model}
+\text{optimization}
+\text{runtime environment}
}
\]

## 3. Required run metadata

Every meaningful training run should preserve at least:

### Repository state

- git commit SHA;
- branch/tag when available;
- dirty/clean status;
- optional diff or patch for dirty runs if supported.

### Configuration

- fully resolved Hydra configuration;
- run name/identifier;
- seed configuration;
- all task/model/data/trainer/optimizer settings.

### Runtime environment

- Python version;
- PyTorch version;
- CUDA runtime/version when applicable;
- GPU model(s);
- number of GPUs/ranks;
- sparse backend and version;
- Lightning/Hydra versions;
- relevant optional kernel/library versions;
- precision mode.

### Data identity

- dataset manifest;
- source paths or stable dataset identifiers;
- sample/case IDs;
- train/validation/test split definition;
- split-generation seed if generated;
- mesh identifiers;
- file checksums or equivalent stable provenance where practical.

### Field semantics

- available/supported field catalogue version;
- ordered input fields/components;
- ordered target fields/components;
- conditioning fields/variables;
- stored-versus-derived provenance;
- units/dimensional conventions.

### Physical preprocessing

- reference-scale definitions;
- reference-value derivation rules;
- case-specific reference values where needed for exact replay;
- field nondimensionalization formulas/specification version;
- derived dimensionless regime variables;
- resolution descriptor definitions.

### Statistical preprocessing

- exact training-set scaler statistics;
- named field/component association;
- variance epsilon/policy;
- any clipping or nonlinear transforms.

### Training state

- optimizer state;
- scheduler state;
- global step/epoch;
- gradient-accumulation policy;
- effective statistical batch definition;
- microbatch node/edge budgets;
- sampler/packer configuration;
- DDP/world-size details;
- RNG states when exact continuation is required and practical.

## 4. Checkpoint metadata

Checkpoints should contain or reference the complete scientific preprocessing contract required for inference.

At minimum, checkpoint loading for inference must be able to verify compatibility of:

- ordered model input fields;
- output fields;
- field dimensional transformations;
- statistical scalers;
- reference-scale semantics;
- model architecture/configuration.

A checkpoint trained with one semantic channel order must not silently accept a different order.

## 5. Field and scaler naming

Do not persist normalization only as anonymous arrays such as:

```text
mean = [ ... ]
std  = [ ... ]
```

without a semantic mapping.

Persist a structure equivalent to:

```yaml
rho:
  transform: rho / rho_ref
  mean: ...
  std: ...

rhou:
  transform: rhou / (rho_ref * U_ref)
  mean: ...
  std: ...
```

Vector/tensor components must retain explicit ordering.

## 6. Reference-scale reproducibility

For each reference quantity, preserve both numerical value/derivation and semantic definition.

Example:

```yaml
U_ref:
  definition: bulk_velocity
  provenance: boundary_conditions
  value: ...
```

A value without its definition is insufficient.

At inference, the new case may have a different numerical value, but it must be obtained using the same documented definition rules unless a new preprocessing specification is intentionally introduced.

## 7. Split integrity

Statistical preprocessing must use training data only.

Validation and test data must not influence:

- scaler statistics;
- reference definitions fitted from data;
- model selection beyond the explicitly allowed validation process.

Any data-derived preprocessing fit must record the exact training subset used.

## 8. Randomness

Record seeds for:

- Python;
- NumPy;
- PyTorch CPU;
- PyTorch CUDA;
- data sampling/shuffling;
- topology augmentation;
- task noise such as diffusion noise where deterministic replay is required.

Exact bitwise reproducibility across different GPUs/backends is not always guaranteed. When it is not guaranteed, document the expected reproducibility level instead of claiming exact identity.

## 9. Reproducibility levels

Useful labels:

### Configuration reproducible

All scientific/configuration information is preserved, but low-level nondeterministic kernels may prevent exact numerical replay.

### Numerically reproducible

Same environment/hardware reproduces results within documented tolerance.

### Bitwise reproducible

Exact replay is demonstrated under explicitly constrained environment/settings.

Do not claim a stronger level than has been verified.

## 10. Inference reproducibility

An inference output should be traceable to:

- checkpoint identifier;
- checkpoint training metadata;
- inference resolved config;
- input case/sample identity;
- input field provenance;
- new-case reference values and derivation;
- frozen training scaler;
- model/runtime versions;
- random seed if sampling/generation is stochastic.

## 11. Benchmark reproducibility

Performance results additionally follow `BENCHMARK_PROTOCOL.md` and must include hardware, backend, precision, graph sizes, warmup/measurement protocol, and git SHA.
