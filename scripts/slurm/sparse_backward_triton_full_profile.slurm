#!/bin/bash
#SBATCH --job-name=csr_full_profile
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Operator attribution only: never compare CUDA-profiler wall-clock durations
# to unprofiled timings, and never touch frozen scientific training artifacts.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
DATA_ROOT=${CSR_FULL_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
CASE=${CSR_PROFILE_CASE:-Re54/Ma0.3:65}
OUTPUT_ROOT=${CSR_FULL_PROFILE_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/triton_full_profile_v1}
OUTPUT="$OUTPUT_ROOT/job_${SLURM_JOB_ID}"
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs

echo "ARTIFACT_STDOUT=$LOG_ROOT/csr_full_profile_${SLURM_JOB_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/csr_full_profile_${SLURM_JOB_ID}.err"
echo "ARTIFACT_PREFLIGHT=$OUTPUT/preflight.json"
echo "ARTIFACT_SUMMARY=$OUTPUT/summary.json"
for backend in torch_sparse torch_sparse_triton; do
    echo "ARTIFACT_${backend}_REPORT=$OUTPUT/$backend/report.json"
    echo "ARTIFACT_${backend}_OP_PROFILE=$OUTPUT/$backend/operator_profile.json"
    echo "ARTIFACT_${backend}_LOG=$OUTPUT/$backend/process.log"
done

for resource in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT"; do
    test -e "$resource" || { echo "Missing: $resource" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "Output directory exists: $OUTPUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -x "$PYTHON"
echo "Revision=$(git rev-parse HEAD), case=$CASE, host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
    scripts/m43_b1_training_step.py scripts/profile_triton_full_training.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_triton_full_profile.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.profile_triton_full_training \
    --root "$DATA_ROOT" --case "$CASE" --output-dir "$OUTPUT"
echo "Profiler diagnostic complete: $OUTPUT/summary.json"
