# M41.3: GH200 257^3 HDF5 metadata and next graph-topology gate

## Job 464614 — completed 2026-10-14

The metadata preflight completed on `calypso-grace04` with **10/10
tests passing**. It did NOT construct a full attention graph or execute
GPU inference.

- Mesh `/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3/MESH/mesh.mesh.h5`
  is 1.5633 GiB; snapshot
  `/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3/SOLUT/solut_hit_00000001.h5`
  is 2.4662 GiB.
- `num_nodes = 16,974,593 = 257^3`,
  `num_hex_cells = 16,777,216 = 256^3`.
- Mesh coordinates `Coordinates/x,y,z` are three `float64[N]` arrays.
  Connectivity `Connectivity/hex->node` is flattened
  `uint64[134,217,728]`, **one-based**. The full connectivity scan
  found global minimum 1 and maximum `16,974,593`.
- The five conservative fields `rho, rhou, rhov, rhow, rhoE` are all
  `float64[N]`.
- Two FP32 hidden-state buffers at H=128 require **16.1882 GiB CPU RAM**.
- If the mesh is exactly a nonperiodic cubic six-neighbor graph,
  theoretical directed edge counts are **101,451,264** (one-hop) and
  **303,167,994** (exact-two-hop). These are **not** yet measured
  actual graph edge counts.

## New gate: Cartesian graph equivalence, not GPU inference

Metadata equality alone is insufficient to replace the M37 graph
with a Cartesian stencil. We must prove that *all* HDF5 nodes and
hexahedra match that construction:

- `scripts/qualify_m41_3_cartesian_mesh.py` makes bounded-size HDF5
  reads while scanning **every coordinate** and **every hexahedron**.
  It requires a bijection between HDF5 node IDs and a uniform
  `side^3` Cartesian lattice within a configurable relative
  coordinate tolerance (`0.001` of a cell width by default).
- For every cell it verifies exactly eight unit-cube corners and the
  **same 12 canonical local hexahedron edges as M37**, so a different
  cell-vertex ordering cannot silently produce diagonal graph edges.
  It also checks each physical cube appears exactly once. A failed
  check refuses the Cartesian shortcut.
- On success the qualifier writes `cartesian_qualification.json`
  and two `int32` NumPy `.npy` maps:
  `global_to_cartesian.npy` and `cartesian_to_global.npy`.
  Each map is roughly 64.75 MiB. Their content can be memory-mapped,
  including when HDF5 node IDs are arbitrarily permuted.
- `src/graph_attention/geometry/cartesian_halo.py` is a **candidate
  per-partition graph constructor**. It creates owned-node IDs, read
  IDs containing the complete one-hop/exact-two-hop halo, and directed
  incoming one-hop/exact-two-hop edges **for one partition at a time**.
  It never constructs global 100M/303M edge tensors.
- `tests/unit/test_m41_3_cartesian_mesh.py` compares the assembled
  partition edges against a mesh-derived oracle on shuffled small
  hex meshes and checks fail-closed behavior for incorrect
  connectivity ordering, duplicate cells, malformed cells,
  nonuniform coordinates and invalid one-based indices.
- The qualification does not inspect solver-side periodic pairing
  metadata; it proves equivalence to the HDF5 hexahedron-derived
  graph, not necessarily to an alternative CFD periodic adjacency.

The per-partition helper is **not yet wired into**
`StreamedDiNATPlan`, and no real `257^3` GPU inference has been
attempted. The original M41.3 numerical validation on 35,937 nodes
remains valid and unchanged.

## Run on Grace

```bash
cd /scratch/coop/theret/GraphAttention
git pull --rebase
sbatch scripts/slurm/m41_3_257cube_qualify.slurm
```

The Slurm script explicitly requests `--partition=grace`,
`--gres=gpu:1`, `--cpus-per-task=4`, `--time=04:00:00`;
there is **no explicit --mem limit**. Qualification is CPU-only,
but it reserves the known Grace resources.

Results:
`/scratch/coop/theret/GraphAttention_runs/m41_3_257_topology/<job-id>/`

If the qualification fails, examine the first failed constraint and
design a generic streamed connectivity builder instead of pretending
the mesh is a Cartesian lattice.

If it passes: integrate the validated partition-local builder into
M41.3's CPU-resident CSR/geometry plan, test exact per-layer
parity against M37 on small meshes, then run progressively larger
forward benchmarks before 50-step Heun at `257^3`.
