#!/bin/bash
#SBATCH --job-name=m45_two_scale_tests
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M45 small-graph qualification for BOTH static and dynamic Two-Scale DiNAT.
# Geometry / backprop / partition equivalence; no DNS data or model training.
# Runs on Grace in the existing GraphAttention Singularity + Python environment.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: GraphAttention checkout is dirty; commit/stash changes before qualification." >&2
    git status --short >&2
    exit 2
fi
test -s "$IMAGE" || { echo "Missing Singularity image: $IMAGE" >&2; exit 2; }
test -x "$PYTHON" || { echo "Missing Python environment: $PYTHON" >&2; exit 2; }
test -f tests/unit/test_two_scale_hit.py || {
    echo "Missing M45 tests: update GraphAttention first (git pull --rebase)." >&2
    exit 2
}

echo "=== M45 Two-Scale DiNAT: static and dynamic qualification ==="
echo "Host: $(hostname) | job: ${SLURM_JOB_ID:-unknown}"
echo "Revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "=== Python syntax gate ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
        src/graph_attention/geometry/relative.py \
        src/graph_attention/geometry/two_scale_hit.py \
        src/graph_attention/geometry/two_scale_partition.py \
        src/graph_attention/models/dinat_dit.py \
        src/graph_attention/models/two_scale_dinat.py \
        tests/unit/test_two_scale_hit.py

echo "=== M45 geometry, sparse attention, gradient, and partition-parity gates ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_two_scale_hit.py

echo "=== M37 / M44 regression gates for changed shared attention geometry ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
        tests/unit/test_geometry_relative.py \
        tests/unit/test_dinat_dit.py \
        tests/unit/test_m44_sr0.py \
        tests/unit/test_flow_matching_task.py

echo "PASS: M45 static/dynamic qualification and M37/M44 regression gates."
