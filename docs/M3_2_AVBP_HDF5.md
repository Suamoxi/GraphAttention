# M3.2 AVBP HDF5 Data Path

## Purpose

M3.2 introduces the first real CFD file reader while preserving the ownership rules frozen in M0.

The data layer reads what exists in the AVBP HDF5 source. It does not choose learning targets, concatenate anonymous model channels, nondimensionalize fields, fit statistical scalers, or invent graph topology.

## Supported HDF5 convention

The initial reader targets the AVBP/HIT-style paths already used by this project family:

```text
GaseousPhase/rho
GaseousPhase/rhou
GaseousPhase/rhov
GaseousPhase/rhow
GaseousPhase/rhoE
Coordinates/x
Coordinates/y
Coordinates/z
Connectivity/hex->node
```

This is a project-supported file convention, not a claim that every AVBP export has the same hierarchy.

The default catalogue also declares the known optional node fields `pressure`, `temperature`, `vis_lam`, `vis_turb`, and `mpi_rank`. A field is only opened when explicitly requested by canonical name.

## Field semantics

`AVBP_FIELD_CATALOG` maps canonical field names to source paths and scientific roles. The reader accepts ordered `field_names` and loads only those datasets.

The conservative state components remain individually stored quantities because that is how this AVBP layout exposes them. Component semantics are retained explicitly for `rhou`, `rhov`, and `rhow`; later task code may group ordered components without the reader deciding a task input tensor.

No field dtype is silently converted to `float32`. Source numerical dtype is preserved when representable by PyTorch. Scalar node arrays are canonicalized from `[N, 1]` to `[N]`; other silent reshaping is rejected.

## Native mesh connectivity

The HDF5 source supplies hexahedral cell-to-node connectivity. M3.2 extends `Mesh` with optional:

```text
cell_connectivity [C, K]
```

For the AVBP reader, `K = 8`. File indexing is canonicalized to zero-based `torch.long` during format decoding.

`connectivity_indexing` accepts:

- `zero`: require zero-based source indices;
- `one`: require one-based source indices;
- `auto`: infer only when the observed range is unambiguous.

If both zero-based and one-based interpretations are valid, `auto` fails rather than silently guessing.

## Data versus geometry ownership

The AVBP reader stores native cell connectivity and returns an empty `edge_index [2, 0]`. This is deliberate.

The geometry layer owns the separate deterministic transform:

```text
hex_connectivity_to_edge_index
```

For each hexahedral cell it uses the 12 physical cell edges, removes duplicates shared by adjacent cells, introduces no self-loops, and represents every undirected mesh edge by two directed entries.

`mesh_with_hex_edge_index(mesh)` applies that transform without reopening or reinterpreting the HDF5 source.

This preserves the dependency direction:

```text
AVBP reader -> Mesh with native connectivity -> geometry transform -> graph edge_index
```

## Shared and per-sample meshes

`AVBPHDF5Dataset` supports both:

- per-sample coordinates/connectivity stored in every HDF5 snapshot;
- a separate shared `mesh_file` used by all field snapshots.

Input snapshots may be provided explicitly through `files` or discovered from `data_dir` with `file_pattern` and `recursive`.

Sample identity is currently the unique file stem. Duplicate stems are rejected rather than silently colliding. A richer dataset manifest remains part of the reproducibility work later in M3.

## Genealogy relative to diffusion4avbp

The existing `diffusion4avbp` reader established the practical project paths for `GaseousPhase/*`, `Coordinates/*`, and `Connectivity/hex->node`, together with support for separate shared mesh files and zero/one-based connectivity.

M3.2 is a project adaptation rather than a direct copy. The important changes are:

- named `FieldCatalog` semantics instead of anonymous channel-path concatenation;
- source dtype preservation instead of unconditional `float32` conversion;
- native cell connectivity retained explicitly;
- no padded `neighbor_idx` / `neighbor_mask` construction;
- graph-edge construction moved to the geometry layer;
- ambiguous connectivity indexing fails explicitly.

## Deliberate non-goals

M3.2 does not implement:

- derived physical quantities;
- species discovery under `RhoSpecies`;
- reference-scale extraction;
- physical nondimensionalization;
- training-set statistical scaling;
- task input/target selection;
- packed graph batching;
- node/edge budget packing;
- graph dilation or long-range edges;
- sparse attention.

## Validation gate

Before M3.2 is considered complete, run on the target development environment:

```bash
git pull --rebase origin main
pytest
ruff check .
ruff format --check .
python scripts/inspect_config.py
```

The unit tests create small temporary HDF5 files and validate named-field loading, shared/per-sample meshes, dtype preservation, missing-field diagnostics, file discovery, indexing ambiguity, and deterministic hex-to-edge conversion.

A final M3.2 check should also load at least one real project AVBP/HDF5 file and inspect its field, coordinate, and connectivity shapes. Passing synthetic temporary-file tests alone proves the reader contract, not compatibility with every historical solver export.
