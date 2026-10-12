#!/bin/bash
#SBATCH --job-name=m41_3_257_preflight
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Metadata-only, CPU-only HDF5 inspection. This deliberately does NOT ask for a
# GPU and does NOT call GraphTaskCollator or build the full two-hop graph.
# Supply the appropriate Slurm partition with sbatch --partition=<cpu partition>
# if Calypso's default partition does not accept CPU-only jobs.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE=${M41_3_DATA_BASE:-/archive/coop/theret/HIT_DNS_COMP/Re150/Ma0.3}
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
echo "No full graph construction. No GPU. No trained model or sampler invoked."

singularity exec -B /home -B /scratch -B /archive "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_m41_3_large_hdf5_preflight.py -q

singularity exec -B /home -B /scratch -B /archive "$IMAGE" \
    "$PYTHON" -m scripts.preflight_m41_3_large_hdf5 \
    --mesh-file "$MESH_FILE" \
    --snapshot-file "$SNAPSHOT_FILE" \
    --expected-nodes 16974593 \
    --hidden-dim 128 \
    --scan-connectivity-extrema \
    --output "$OUT/preflight.json" > "$OUT/preflight_stdout.json"

echo "Preflight metadata saved: $OUT/preflight.json"
echo "No inference or graph generation was attempted."
