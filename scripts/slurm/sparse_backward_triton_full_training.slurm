#!/bin/bash
#SBATCH --job-name=csr_full_train
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Paired full-model training-step performance probe on a REAL HDF5 mesh.
# The M43-B1 probe uses per-snapshot scaling for feasibility ONLY.
# Never writes checkpoints or changes frozen M47.4 training experiments.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
DATA_ROOT=${CSR_FULL_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
CASE=${CSR_FULL_CASE:-Re54/Ma0.3:65}
STEPS=${CSR_FULL_STEPS:-4}
OUTPUT_ROOT=${CSR_FULL_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/triton_full_training_v1}
OUTPUT="$OUTPUT_ROOT/job_${SLURM_JOB_ID}"
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs

echo "ARTIFACT_STDOUT=$LOG_ROOT/csr_full_train_${SLURM_JOB_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/csr_full_train_${SLURM_JOB_ID}.err"
echo "ARTIFACT_PREFLIGHT=$OUTPUT/preflight.json"
echo "ARTIFACT_SUMMARY=$OUTPUT/summary.json"
for backend in torch_sparse torch_sparse_triton; do
    echo "ARTIFACT_${backend}_REPORT=$OUTPUT/$backend/report.json"
    echo "ARTIFACT_${backend}_LOG=$OUTPUT/$backend/process.log"
done
echo "GPU full-model probe: case=$CASE steps=$STEPS root=$DATA_ROOT"
for resource in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT"; do
    test -e "$resource" || { echo "Missing: $resource" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "Output directory already exists: $OUTPUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -x "$PYTHON"
echo "Revision: $(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
# Static syntax and small checkpointed numerical gradient gate.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
    scripts/m43_b1_training_step.py scripts/benchmark_triton_full_training.py
singularity exec --nv "${BIND[@]}" "$IMAGE" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -c '
import pytest
import torch
torch.use_deterministic_algorithms(True)
raise SystemExit(pytest.main([
    "-q", "tests/unit/test_triton_sparse_backward.py::test_triton_full_dinat_with_activation_checkpointing",
]))'
# The benchmark itself uses the original unrestricted CUDA execution mode.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.benchmark_triton_full_training \
    --root "$DATA_ROOT" --case "$CASE" --steps "$STEPS" \
    --output-dir "$OUTPUT"
echo "Finished full-model probe: $OUTPUT/summary.json"
