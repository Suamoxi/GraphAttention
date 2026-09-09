# Reproducibility

## 1. Purpose

A training or inference result is scientifically useful only if its data selection, preprocessing, model configuration, and runtime environment can be reconstructed.

This document defines the minimum reproducibility state that must accompany meaningful experiments.

## 2. Reproducibility principle

A checkpoint alone is not a complete experiment artifact.

A reproducible run must preserve enough information to reconstruct:

$$
\boxed{
\text{data}
+\text{field semantics}
+\text{physical preprocessing}
+\text{task}
+\text{model}
+\text{optimization}
+\text{runtime environment}
}
$$

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

- authoritative case-definition document or exact preserved content/hash;
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

The M3.3 baseline uses an authoritative case-definition document. Its `value` entries are explicit numerical values; optional `derivation` entries document how those values were obtained but are not evaluated by the runtime loader.

Example:

```yaml
case_id: case_a
reference_scheme: bulk_flow_reference
references:
  U_ref:
    definition: bulk_velocity
    provenance: boundary_conditions
    units: m/s
    value: ...
    inference_available: true
    derivation: prescribed value from simulation setup
```

A value without its definition is insufficient. The case document itself, or stable content sufficient to reconstruct it exactly, must be retained with the run.

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
- authoritative new-case definition and reference values/derivation;
- frozen training scaler;
- model/runtime versions;
- random seed if sampling/generation is stochastic.

## 11. Benchmark reproducibility

Performance results additionally follow `BENCHMARK_PROTOCOL.md` and must include hardware, backend, precision, graph sizes, warmup/measurement protocol, and git SHA.

## 12. M13 standalone diffusion-generation replay

M13 deliberately separates training from reverse generation. A diffusion training run must persist enough state for generation to reconstruct the scientific model without refitting any data-dependent quantity. The required source artifacts are:

- `resolved_config.yaml` for the exact data/task/model/geometry definitions;
- `summary.json` for model/run provenance and selected epoch;
- `dataset_split_manifest.json` for the exact held-out test IDs;
- `standardizers.pt` for the frozen named train-only input/target statistics;
- the selected checkpoint, normally `best.pt`.

The standalone generation runner reloads those artifacts. It must not regenerate the train/validation/test split and must not refit statistical scaling. If a saved test ID is absent from the currently resolved dataset, generation fails rather than substituting another sample.

Every generation configuration is isolated below the source training run. Its directory name records sampler family, number of reverse model evaluations, `eta`, and sampling seed unless an explicit output name is supplied. Existing generation directories are not replaced unless `overwrite=true` is requested.

A generation directory preserves:

- `generated_test.pt` containing generated and held-out reference populations;
- `generation_config.yaml` containing reverse-process settings;
- a copy of the source `resolved_config.yaml`;
- a copy of `dataset_split_manifest.json`;
- `summary.json` identifying the source run, source checkpoint, sampler, step count, `eta`, seed, and model-evaluation count.

Generated sample IDs such as `gen_000000` are deterministic RNG keys for the generated population. Held-out test IDs are stored separately as reference identifiers. Equal list positions do **not** assert a generated-to-target pairing. The reference population is included only to support population-level post-processing with the common generation benchmark.

For stochastic reverse sampling, each generated sample receives its own deterministic generator derived from the configured sampling seed and generated ID. Reordering computational batches should therefore not redefine the intended generated sample RNG stream. As elsewhere in the repository, bitwise cross-device or cross-PyTorch-version identity is not claimed until explicitly demonstrated.
