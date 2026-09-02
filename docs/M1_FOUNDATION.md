# M1 Foundation

## Purpose

M1 establishes the minimum technical foundation required before scientific data contracts or graph models are implemented.

## Frozen choices

- Python: `>=3.11,<3.13`.
- Core framework: PyTorch.
- Configuration: Hydra/OmegaConf with one composition tree.
- Training orchestration: Lightning, kept thin and task-agnostic.
- Scientific I/O baseline: NumPy and h5py.
- Testing: pytest.
- Formatting/linting: Ruff.
- Package layout: standard `src/` layout using the `graph_attention` Python package.
- Construction policy: prefer Hydra `_target_`/`hydra.utils.instantiate` over custom registries.
- Sparse graph backend: intentionally undecided in M1. DGL, PyG, Triton, or custom kernels must not be selected before the real sparse-attention requirements and benchmark evidence exist.

## Repository structure

```text
GraphAttention/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── config.yaml
│   ├── data/synthetic.yaml
│   ├── model/baseline.yaml
│   ├── task/regression.yaml
│   ├── optimizer/adamw.yaml
│   └── trainer/default.yaml
├── docs/
├── scripts/
│   └── inspect_config.py
├── src/graph_attention/
│   ├── data/
│   ├── geometry/
│   ├── models/
│   ├── tasks/
│   ├── training/
│   └── utils/
└── tests/
    ├── unit/
    ├── scientific/
    └── integration/
```

The empty scientific/integration test directories are intentional category boundaries, not claims that those tests already exist.

## Dependency boundaries

M0 ownership remains authoritative:

- data: what exists;
- geometry: how space is related;
- task: what is learned;
- model: how the supplied representation is transformed;
- training: how optimization is executed.

M1 creates only package boundaries. It does not implement the M2 data schemas, M3 packing, M4 task/model lifecycle, or M6 sparse transformer.

## Hydra policy

The repository uses a single root composition in `configs/config.yaml`.

Experiment-specific behavior should be expressed through composition and overrides rather than duplicated top-level launch configurations.

Hydra `_target_` is used where a concrete constructible object already exists. Placeholder data/model/task configs are intentionally not given fake `_target_` values before their implementations exist.

## Reproducibility foundation

`graph_attention.utils.provenance.collect_runtime_provenance` records only information that is genuinely available in M1:

- git SHA;
- branch;
- dirty/clean state;
- Python version;
- PyTorch version;
- CUDA version/availability;
- GPU identity/count.

Dataset manifests, field semantics, physical reference scales, normalization statistics, splits, and scientific preprocessing are not represented until the milestones that define them.

## M1 gate

M1 is complete when the following succeed in a clean environment:

```bash
python -m pip install -e ".[dev]"
python -c "import graph_attention"
python scripts/inspect_config.py
pytest
ruff check .
ruff format --check .
```

In addition:

- the default Hydra configuration must compose;
- provenance collection must run without requiring a GPU;
- no graph backend may have been selected prematurely;
- no dataset/model/task implementation may have been added solely to make the repository appear complete.

Passing these checks establishes the software foundation only. It does not validate any scientific model or performance claim.
