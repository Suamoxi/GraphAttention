# M3.2 AVBP HDF5 Data Path

## Purpose

M3.2 introduces the first real CFD file reader while preserving the ownership rules frozen in M0.

The data layer reads what exists in the AVBP HDF5 source. It does not choose learning targets, concatenate anonymous model channels, nondimensionalize fields, fit statistical scalers, or invent graph topology.

## Project AVBP file relationship

For this project, an AVBP solution snapshot and its mesh are separate HDF5 files. A dataset may contain several meshes, with many snapshots sharing each mesh.

The association is therefore explicit for every physical sample:

```text
sample_id + snapshot_file + mesh_id + mesh_file
```

`AVBPSampleSpec` is the corresponding runtime contract. `AVBPHDF5Dataset` accepts either `AVBPSampleSpec` instances or mappings with exactly those four keys, which allows Hydra to supply the same information directly.

The reader does not infer mesh association from file stems, directory names, or ordering. Duplicate `sample_id` values are rejected. A `mesh_id` must refer to exactly one mesh file, and one mesh file must not silently receive several `mesh_id` values.

## Mesh caching and data efficiency

Meshes are decoded lazily and cached by canonical mesh path inside each dataset process. If 100 snapshots refer to the same mesh, that process performs 100 snapshot reads but only one mesh decode/read after the cache is warm.

The cached `Mesh` object is reused by samples that share that mesh. Current geometry transforms are functional and return new `Mesh` objects rather than mutating the cached base mesh. Callers must therefore treat dataset-owned cached mesh tensors as read-only. A future transformation that requires in-place or sample-specific mesh mutation must copy explicitly rather than modifying shared cached state.

With `DataLoader(num_workers > 0)`, workers are separate processes and may each own a mesh cache. M3.2 does not add shared-memory mesh infrastructure before a measured CPU-memory need exists.

## Supported HDF5 convention

The initial reader targets the AVBP/HIT-style paths already used by this project family:

```text
snapshot file:
  GaseousPhase/rho
  GaseousPhase/rhou
  GaseousPhase/rhov
  GaseousPhase/rhow
  GaseousPhase/rhoE

mesh file:
  Coordinates/x
  Coordinates/y
  Coordinates/z
  Connectivity/hex->node
```

This is a project-supported file convention, not a claim that every AVBP export has the same hierarchy.

The default catalogue also declares the known optional node fields `pressure`, `temperature`, `vis_lam`, `vis_turb`, and `mpi_rank`. A field is only opened when explicitly requested by canonical name.

## Field semantics

`AVBP_FIELD_CATALOG` maps canonical field names to source paths and scientific roles. The reader accepts ordered `field_names` and loads only those datasets from the snapshot file.

The conservative state components remain individually stored quantities because that is how this AVBP layout exposes them. Component semantics are retained explicitly for `rhou`, `rhov`, and `rhow`; later task code may group ordered components without the reader deciding a task input tensor.

No field dtype is silently converted to `float32`. Source numerical dtype is preserved when representable by PyTorch. Scalar node arrays are canonicalized from `[N, 1]` to `[N]`; other silent reshaping is rejected.

## Native mesh connectivity

The mesh HDF5 source supplies hexahedral cell-to-node connectivity. `Mesh` retains optional:

```text
cell_connectivity [C, K]
```

For the current AVBP reader, `K = 8`. File indexing is canonicalized to zero-based `torch.long` during format decoding.

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
AVBP sample specification
    -> snapshot fields + cached native mesh
    -> geometry transform
    -> graph edge_index
```

## Hydra configuration

The `data=avbp_hdf5` config intentionally contains an empty `samples` list. Real runs must provide explicit associations, for example:

```yaml
samples:
  - sample_id: case_a_000001
    snapshot_file: /data/case_a/snapshot_000001.h5
    mesh_id: mesh_a
    mesh_file: /data/case_a/mesh.h5
  - sample_id: case_b_000001
    snapshot_file: /data/case_b/snapshot_000001.h5
    mesh_id: mesh_b
    mesh_file: /data/case_b/mesh.h5
```

M3.2 does not add implicit directory discovery because it would reintroduce an unverified rule for deciding which mesh belongs to which snapshot. A generated manifest may be added later if a concrete dataset-layout rule can create the same explicit associations reproducibly.

## Genealogy relative to diffusion4avbp

The existing `diffusion4avbp` reader established the practical project paths for `GaseousPhase/*`, `Coordinates/*`, and `Connectivity/hex->node`, together with support for separate shared mesh files and zero/one-based connectivity.

M3.2 is a project adaptation rather than a direct copy. The important changes are:

- explicit snapshot-to-mesh association for variable-mesh datasets;
- lazy per-process mesh caching;
- named `FieldCatalog` semantics instead of anonymous channel-path concatenation;
- source dtype preservation instead of unconditional `float32` conversion;
- native cell connectivity retained explicitly;
- no padded `neighbor_idx` / `neighbor_mask` construction;
- graph-edge construction moved to the geometry layer;
- ambiguous connectivity indexing fails explicitly.

## Real-data validation

M3.2 was validated on Calypso on 2026-09-02 against a real `HIT_LES_FORCED` AVBP solution/mesh pair.

Observed real-file properties were:

```text
conservative snapshot fields: 35937 nodes
coordinates:                  [35937, 3]
raw hex connectivity:         262144 entries
canonical hex connectivity:   [32768, 8]
canonical node-index range:   0 .. 35936
```

The five conservative fields `rho`, `rhou`, `rhov`, `rhow`, and `rhoE` all matched the mesh node count and retained their source `float64` dtype. The real reader check completed successfully.

The same mesh also contains explicit AVBP `Periodicity/*` data because the HIT domain is a fully periodic cube. M3.2 does not decode that metadata into graph edges. Therefore `hex_connectivity_to_edge_index` currently represents ordinary cell-edge topology only and must not be claimed to represent the complete physical periodic topology for such a case. Periodic equivalence and translation-aware geometry are deliberately deferred to a later geometry extension.

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
- shared-memory mesh caches across DataLoader workers;
- periodic node-equivalence decoding or periodic graph edges;
- periodic translation correction for relative geometry;
- interpretation of `VertexData/volume` as a quadrature/control-volume weight;
- graph dilation or long-range edges;
- sparse attention.

## Validation gate

M3.2 is complete for its declared scope.

The final target-development checks were:

```text
pytest                 -> 34 passed
ruff check .           -> passed
ruff format --check .  -> passed
real AVBP reader check -> passed
```

The real-file validation establishes compatibility with the tested project AVBP/HDF5 convention. It is not a claim that every historical or future AVBP export uses the same paths, topology, cell type, or periodic encoding.
