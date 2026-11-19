#!/bin/bash
#SBATCH --job-name=csr_bwd_triton
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# One GPU per structured graph / attention topology. Run ALL four modes:
# sbatch --array=0-3 scripts/slurm/sparse_backward_triton_benchmark.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
ROOT=${CSR_TRITON_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/sparse_backward_triton_v1}
PROFILE=${CSR_TRITON_PROFILE:-NO}
REPEATS=${CSR_TRITON_REPEATS:-3}
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) SIDE=65; TOPOLOGY=one_hop ;;
    1) SIDE=65; TOPOLOGY=two_hop ;;
    2) SIDE=129; TOPOLOGY=one_hop ;;
    3) SIDE=129; TOPOLOGY=two_hop ;;
    *) echo "submit with --array=0-3" >&2; exit 2 ;;
esac
case "$PROFILE" in YES|NO) ;; *) echo "CSR_TRITON_PROFILE must be YES or NO" >&2; exit 2 ;; esac
for f in "$REPO" "$IMAGE" "$PYTHON"; do
    test -e "$f" || { echo "Missing: $f" >&2; exit 2; }
done
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
test -x "$PYTHON"
OUTPUT="$ROOT/${SIDE}cube_${TOPOLOGY}_${SLURM_JOB_ID}"
test ! -e "$OUTPUT" || { echo "Output exists: $OUTPUT" >&2; exit 2; }
# Print exact artifact paths before any GPU tests, including failure cases.
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs
echo "ARTIFACT_STDOUT=$LOG_ROOT/csr_bwd_triton_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/csr_bwd_triton_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
echo "ARTIFACT_RESULT=$OUTPUT/results.json"
echo "ARTIFACT_PROFILE_DIR=$OUTPUT (only when CSR_TRITON_PROFILE=YES)"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "CSR Triton benchmark revision=$(git rev-parse HEAD) graph=${SIDE}cube topo=$TOPOLOGY"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -c \
    'import torch,triton;print("torch",torch.__version__,"triton",triton.__version__,"cuda",torch.cuda.is_available())'
# Use deterministic CUDA for the numerical parity gate only.
# The original chunked index_add_ forward is nondeterministic on CUDA;
# deterministic execution avoids stochastic false negatives in gradient parity.
# Run the latency benchmark separately in its original (non-deterministic) mode.
singularity exec --nv "${BIND[@]}" "$IMAGE" \
    env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -c '
import sys
import pytest
import torch

torch.use_deterministic_algorithms(True)
print("TRITON_PARITY_DETERMINISTIC=", torch.are_deterministic_algorithms_enabled(), flush=True)
raise SystemExit(pytest.main(["-q", "-s", *sys.argv[1:]]))
' tests/unit/test_triton_sparse_backward.py tests/unit/test_dinat_dit.py
ARGS=()
if [ "$PROFILE" = YES ]; then ARGS+=(--profile); fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.benchmark_triton_sparse_backward \
    --side "$SIDE" --topology "$TOPOLOGY" \
    --operation all --repeats "$REPEATS" \
    --output-dir "$OUTPUT" "${ARGS[@]}"
echo "CSR Triton benchmark complete: $OUTPUT/results.json"
