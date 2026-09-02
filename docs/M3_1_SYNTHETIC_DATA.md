# M3.1 Synthetic Variable-Mesh Data

## Purpose

M3.1 introduces the first concrete dataset implementation without introducing CFD file I/O or scientific preprocessing.

`SyntheticMeshDataset` exists only to exercise the real `FieldCatalog`, `Mesh`, and `Sample` contracts under variable graph sizes, geometries, and topologies before AVBP/HDF5 data is added.

The generated values are **non-physical**. They must not be used for scientific evaluation, model-quality claims, or performance claims about a real CFD workload.

## Dataset contract

Each dataset element is one `Sample` containing one native node-based `Mesh`.

For sample index `i`, the node count is

$$
N_i = N_{\min} + i \bmod (N_{\max}-N_{\min}+1).
$$

The topology cycles deterministically through:

- `chain`;
- `cycle`;
- `star`.

Each undirected synthetic connection is represented by two directed entries in `edge_index`. This is a property of the synthetic fixture only; M3.1 does not establish a repository-wide directed/undirected connectivity convention.

Coordinates are generated in `[0, 1)^D` from a local CPU `torch.Generator` seeded with `seed + sample_index`. Dataset access therefore does not advance the process-global PyTorch RNG, and fetching a sample is independent of access order.

## Synthetic fields

Two named node-supported fields exercise scalar and vector semantics:

```text
rho       [N]
momentum  [N, D]
```

Their values are deterministic functions of the generated coordinates:

$$
\rho_i = 1 + \|x_i\|_2^2,
$$

$$
m_i = \rho_i (x_i - 0.5).
$$

These names are chosen to exercise CFD-like scalar/vector contracts. The values do not represent a physically consistent density or momentum field.

The corresponding `FieldSpec` entries are marked `stored=False` with explicit synthetic provenance.

## Node weights

Each sample receives uniform node weights

$$
w_i = \frac{1}{N}.
$$

These weights are a convenient normalized synthetic fixture. They are not claimed to be CFD quadrature or control-volume weights.

## Configuration

The default Hydra data group now instantiates the dataset directly:

```yaml
_target_: graph_attention.data.SyntheticMeshDataset
num_samples: 9
min_nodes: 4
max_nodes: 12
spatial_dim: 2
seed: ${seed}
```

The root run seed therefore controls the synthetic data seed by default.

## Reproducibility scope

The generator is deterministic by sample index for a fixed supported runtime and configuration. Exact cross-version random-number identity is not claimed; runtime versions remain part of provenance.

## Deliberately deferred

M3.1 does not implement:

- HDF5/AVBP readers;
- field discovery from real CFD files;
- reference-scale extraction;
- nondimensionalization;
- training-set statistical scaling;
- packed graph batching;
- node/edge microbatch budgets;
- sampling or optimizer-batch semantics;
- models or tasks.

## M3.1 gate

M3.1 is complete when:

- synthetic samples validate against their field catalogue;
- node counts, edge counts, topology, and geometry vary across the default-style dataset;
- repeated access to one index is deterministic;
- sample access does not advance the global PyTorch RNG;
- the Hydra data configuration instantiates successfully;
- `pytest`, `ruff check .`, and `ruff format --check .` pass.
