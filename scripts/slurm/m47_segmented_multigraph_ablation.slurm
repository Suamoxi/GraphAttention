#!/bin/bash
#SBATCH --job-name=m47_seg_batch
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# M47 compute-only diagnostic. Task 0: two distinct real 65^3 HDF5 snapshots.
# Task 1: mixed real 65^3 and 129^3 graphs (different regimes/resolutions).
# Run task 0 first; submit task 1 after reviewing its parity and memory.
# Does not modify or start scientific M47 training.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
DATA_ROOT=${M47_SEGMENTED_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
STEPS=${M47_SEGMENTED_STEPS:-4}
BACKEND=${M47_SEGMENTED_BACKEND:-torch_sparse_triton}
OUTPUT_ROOT=${M47_SEGMENTED_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_segmented_multigraph_v1}
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs
ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
OUTPUT="$OUTPUT_ROOT/job_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}"

case "$ARRAY_TASK_ID" in
    0) SCENARIO=two_snapshots_65 ;;
    1) SCENARIO=mixed_65_129 ;;
    *) echo "Unknown scenario array index $ARRAY_TASK_ID" >&2; exit 2 ;;
esac

echo "ARTIFACT_STDOUT=$LOG_ROOT/m47_seg_batch_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/m47_seg_batch_${ARRAY_JOB_ID}_${ARRAY_TASK_ID}.err"
echo "ARTIFACT_MANIFEST=$OUTPUT/manifest.json"
echo "ARTIFACT_SUMMARY=$OUTPUT/summary.json"
for mode in packed_index packed_segmented sequential_accum; do
    echo "ARTIFACT_${mode}_REPORT=$OUTPUT/$mode/report.json"
    echo "ARTIFACT_${mode}_LOG=$OUTPUT/$mode/process.log"
    echo "ARTIFACT_${mode}_AUDIT=$OUTPUT/$mode/audit.pt"
done
echo "Scenario=$SCENARIO Steps=$STEPS Backend=$BACKEND Output=$OUTPUT"
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
test "$STEPS" -ge 3 || { echo "At least 3 optimizer steps required" >&2; exit 2; }
test "$BACKEND" = "torch_sparse" || test "$BACKEND" = "torch_sparse_triton" || {
    echo "Unknown sparse backend: $BACKEND" >&2; exit 2;
}
test ! -e "$OUTPUT" || { echo "Output directory already exists: $OUTPUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -x "$PYTHON"
echo "Revision=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done

# Test both numerical implementations and the backend before GPU benchmarking.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
    scripts/benchmark_m47_segmented.py \
    scripts/benchmark_m47_segmented_worker.py
singularity exec --nv "${BIND[@]}" "$IMAGE" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -c '
import pytest
import torch
torch.use_deterministic_algorithms(True)
print("SEGMENTED_PARITY_DETERMINISTIC=", torch.are_deterministic_algorithms_enabled(), flush=True)
raise SystemExit(pytest.main([
    "-q", "-s", "tests/unit/test_m47_segmented_conditioning.py",
]))'

# Unprofiled, unrestricted CUDA pilot, in fresh processes per mode.
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.benchmark_m47_segmented \
    --root "$DATA_ROOT" --scenario "$SCENARIO" \
    --steps "$STEPS" --backend "$BACKEND" --out "$OUTPUT"
echo "ARTIFACT_FINAL_SUMMARY=$OUTPUT/summary.json"
