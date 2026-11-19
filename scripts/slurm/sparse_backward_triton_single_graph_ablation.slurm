#!/bin/bash
#SBATCH --job-name=csr_broadcast
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Computational ablation only. Four independent full M37 training probes:
# packed graph vs existing single-graph model path, for original and Triton CSR.
# The default model execution and frozen M47 training are unchanged.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
DATA_ROOT=${CSR_FULL_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
CASE=${CSR_BROADCAST_CASE:-Re54/Ma0.3:65}
STEPS=${CSR_BROADCAST_STEPS:-4}
OUTPUT_ROOT=${CSR_BROADCAST_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/triton_broadcast_ablation_v1}
OUTPUT="$OUTPUT_ROOT/job_${SLURM_JOB_ID}"
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs

echo "ARTIFACT_STDOUT=$LOG_ROOT/csr_broadcast_${SLURM_JOB_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/csr_broadcast_${SLURM_JOB_ID}.err"
echo "ARTIFACT_COMPARISON=$OUTPUT/comparison.json"
for mode in packed single_graph; do
    echo "ARTIFACT_${mode}_SUMMARY=$OUTPUT/$mode/summary.json"
    for backend in torch_sparse torch_sparse_triton; do
        echo "ARTIFACT_${mode}_${backend}_REPORT=$OUTPUT/$mode/$backend/report.json"
        echo "ARTIFACT_${mode}_${backend}_LOG=$OUTPUT/$mode/$backend/process.log"
    done
done

for resource in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT"; do
    test -e "$resource" || { echo "Missing resource: $resource" >&2; exit 2; }
done
test ! -e "$OUTPUT" || { echo "Output directory already exists: $OUTPUT" >&2; exit 2; }
test "$STEPS" -ge 3 || { echo "Need at least 3 optimizer steps" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -x "$PYTHON"
echo "Revision=$(git rev-parse HEAD); case=$CASE; steps=$STEPS; host=$(hostname)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
    scripts/m43_b1_training_step.py scripts/benchmark_triton_full_training.py \
    scripts/benchmark_triton_single_graph_ablation.py
singularity exec --nv "${BIND[@]}" "$IMAGE" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -c '
import pytest
import torch
torch.use_deterministic_algorithms(True)
raise SystemExit(pytest.main([
    "-q",
    "tests/unit/test_m47_single_graph_broadcast.py",
    "tests/unit/test_triton_full_training_benchmark.py",
]))'

# Unprofiled, unrestricted CUDA timings: existing paired backend baseline.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.benchmark_triton_single_graph_ablation \
    --root "$DATA_ROOT" --case "$CASE" --steps "$STEPS" --output-dir "$OUTPUT"
echo "Broadcast ablation complete: $OUTPUT/comparison.json"
