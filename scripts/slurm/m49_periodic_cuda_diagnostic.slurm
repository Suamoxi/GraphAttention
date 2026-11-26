#!/bin/bash
#SBATCH --job-name=m49_periodic_cuda_diag
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
echo "GIT=$(git rev-parse HEAD)"
nvidia-smi
# CUBLAS_WORKSPACE_CONFIG is required for the deterministic-mode diagnostic.
# Its presence is recorded; deterministic algorithms remain disabled until
# the explicit diagnostic arm. No long training is started by this script.
singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" \
    --env M49_PERIODIC_CUDA_DIAGNOSTIC=1 --env OMP_NUM_THREADS=4 \
    --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    "$IMAGE" "$PYTHON" -m pytest -q -s -rA \
    'tests/unit/test_m49_periodic_endpoints.py::test_periodic_relative_only_equivariance_training_and_sampling[torch_sparse_triton]'
