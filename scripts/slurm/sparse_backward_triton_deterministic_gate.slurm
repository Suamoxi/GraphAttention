#!/bin/bash
#SBATCH --job-name=csr_bwd_det
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Diagnostic only: repeat the high-degree geometric-attention gradient
# parity test using deterministic PyTorch CUDA reductions. No benchmark
# measurements and no changes to production forward/backward kernels.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
LOG_ROOT=/scratch/coop/theret/GraphAttention_runs
echo "ARTIFACT_STDOUT=$LOG_ROOT/csr_bwd_det_${SLURM_JOB_ID}.out"
echo "ARTIFACT_STDERR=$LOG_ROOT/csr_bwd_det_${SLURM_JOB_ID}.err"
echo "ARTIFACT_RESULTS=stdout/stderr only (no benchmark JSON)"
for path in "$REPO" "$IMAGE" "$PYTHON"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
echo "Revision=$(git rev-parse HEAD) GPU=${CUDA_VISIBLE_DEVICES:-unknown}"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
# A fresh Python process is used for each repetition. Deterministic CUDA
# index_add requires the global PyTorch flag. CuBLAS additionally needs its
# workspace setting before any CUDA initialization.
for attempt in 1 2 3 4 5; do
    echo "DETERMINISTIC_PARITY_ATTEMPT=$attempt/5"
    singularity exec --nv "${BIND[@]}" "$IMAGE" \
        env CUBLAS_WORKSPACE_CONFIG=:4096:8 "$PYTHON" -c '
import pytest
import torch
torch.use_deterministic_algorithms(True)
print("deterministic_algorithms_enabled=", torch.are_deterministic_algorithms_enabled(), flush=True)
raise SystemExit(pytest.main([
    "-q", "-s",
    "tests/unit/test_triton_sparse_backward.py::test_triton_geometric_attention_outputs_parameters_and_bias[53-True]",
]))'
done
echo "DETERMINISTIC_PARITY_RESULT=5/5 passed"
