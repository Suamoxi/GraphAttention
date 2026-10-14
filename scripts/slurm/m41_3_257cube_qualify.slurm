#!/bin/bash
#SBATCH --job-name=m41_3_257_topology
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# A real Grace allocation (same resource contract as other M41 runs),
# with NO explicit --mem cap. CPU-only topology verification; the GPU
# is reserved but no full-graph attention or generation is attempted.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
BASE=${M41_3_DATA_BASE:-/archivenfs/phlusim/HIT_DNS_COMP/Re150/Ma0.3}
MESH_FILE=${MESH_FILE:-${BASE}/MESH/mesh.mesh.h5}
OUT=${OUT:-/scratch/coop/theret/GraphAttention_runs/m41_3_257_topology/${SLURM_JOB_ID:-manual}}

cd "$REPO"
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/git_commit.txt"
git status --short > "$OUT/git_status.txt"
test -s "$MESH_FILE" || { echo "ERROR: missing mesh $MESH_FILE" >&2; exit 1; }
echo "Qualifying entire 257^3 hexahedral topology; host $(hostname)"
echo "Source mesh: $MESH_FILE"
echo "Output: $OUT"
echo "No full graph, CUDA forward, or generation will run in this job."

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest tests/unit/test_m41_3_cartesian_mesh.py -q

singularity exec -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.qualify_m41_3_cartesian_mesh \
    --mesh-file "$MESH_FILE" \
    --output-dir "$OUT" \
    --side 257 \
    --chunk-nodes 262144 \
    --chunk-cells 65536 \
    > "$OUT/cartesian_qualification_stdout.json"

test -s "$OUT/cartesian_qualification.json"
test -s "$OUT/global_to_cartesian.npy"
test -s "$OUT/cartesian_to_global.npy"
echo "Full coordinate-bijection and per-cell topology qualification passed"
echo "Report: $OUT/cartesian_qualification.json"
