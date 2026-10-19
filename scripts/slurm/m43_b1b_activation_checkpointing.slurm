#!/bin/bash
#SBATCH --job-name=m43_b1b_checkpoint
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M43-B1b: baseline/checkpointed training steps at 65^3 and 129^3.
# Four fresh CUDA subprocesses on the SAME Grace allocation.
# Baseline 129^3 OOM is expected; orchestrator continues to checkpoint mode.
# No new model training beyond two diagnostic optimizer updates per case.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
DATA_ROOT=${M43_B1B_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
OUTPUT=${M43_B1B_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_b1b_activation_checkpoint_gate}
STEPS=${M43_B1B_STEPS:-2}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Git checkout must be clean before training-memory diagnostics" >&2
    git status --short >&2
    exit 1
fi
test -f "$DATA_ROOT/Re54/Ma0.3/MESH/mesh.mesh.h5" || exit 1
test -f "$DATA_ROOT/Re94/Ma0.3/MESH/mesh.mesh.h5" || exit 1
echo "==== M43-B1b exact M37 checkpointing parity and memory ===="
echo "Revision: $(git rev-parse HEAD)"
echo "Host: $(hostname)"
echo "Slurm job: $SLURM_JOB_ID"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== Static syntax gate ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/models/dinat_dit.py \
    src/graph_attention/training/model_factory.py \
    scripts/m43_b1_training_step.py \
    scripts/m43_b1b_checkpoint_gate.py \
    tests/unit/test_m43_b1b_checkpointing.py

echo "==== Forward / backward / AdamW exact regression gate ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m43_b1b_checkpointing.py \
    tests/unit/test_m43_b1_training_memory.py \
    tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
    tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_cache_survives_inference_then_training

echo "==== Real 65^3 / 129^3 full-graph training: baseline vs checkpointed ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m43_b1b_checkpoint_gate \
    --root "$DATA_ROOT" \
    --output-dir "$OUTPUT" \
    --case Re54/Ma0.3:65 \
    --case Re94/Ma0.3:129 \
    --steps "$STEPS" \
    --seed 42 \
    --hidden-dim 128 \
    --layers 10 \
    --heads 4

echo "M43-B1b gate result: $OUTPUT/summary.json"
