#!/bin/bash
#SBATCH --job-name=m41_3_257_preflight
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Reserve one GPU on Grace to request the known working cluster resources.
# The HDF5 metadata inspection is CPU-only and does not run inference.
# Do NOT call GraphTaskCollator or build the full two-hop graph.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE=${M41_3_DATA_BASE:-/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3}
MESH_FILE=${MESH_FILE:-${BASE}/MESH/mesh.mesh.h5}
SNAPSHOT_FILE=${SNAPSHOT_FILE:-${BASE}/SOLUT/solut_hit_00000001.h5}
OUT=${OUT:-/scratch/coop/theret/GraphAttention_runs/m41_3_257_preflight/${SLURM_JOB_ID:-manual}}

cd "$REPO"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --short > "$OUT/git_status.txt"

for file in "$MESH_FILE" "$SNAPSHOT_FILE"; do
    test -s "$file" || {
        echo "ERROR: missing input file $file" >&2
        echo "Set MESH_FILE and SNAPSHOT_FILE to the real Calypso locations." >&2
        exit 1
    }
done

echo "M41.3: 257^3 real mesh metadata and indexing preflight"
echo "Mesh: $MESH_FILE"
echo "Snapshot: $SNAPSHOT_FILE"
echo "Report: $OUT"
echo "Allocated host: $(hostname)"
echo "No full graph construction. No GPU computation. No trained model or sampler invoked."

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_m41_3_large_hdf5_preflight.py -q

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m scripts.preflight_m41_3_large_hdf5 \
    --mesh-file "$MESH_FILE" \
    --snapshot-file "$SNAPSHOT_FILE" \
    --expected-nodes 16974593 \
    --hidden-dim 128 \
    --scan-connectivity-extrema \
    --output "$OUT/preflight.json" > "$OUT/preflight_stdout.json"

echo "Preflight metadata saved: $OUT/preflight.json"
echo "No inference or graph generation was attempted."
