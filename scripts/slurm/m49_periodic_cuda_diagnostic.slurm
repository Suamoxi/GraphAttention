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
# Preserve the failed run's default environment first. Optionally repeat
# with M49_DIAG_CUBLAS_WORKSPACE_CONFIG=:4096:8 to qualify deterministic
# cuBLAS execution. Never silently change the training environment.
EXTRA_ENV=()
if [[ -n "${M49_DIAG_CUBLAS_WORKSPACE_CONFIG:-}" ]]; then
    EXTRA_ENV+=(--env "CUBLAS_WORKSPACE_CONFIG=$M49_DIAG_CUBLAS_WORKSPACE_CONFIG")
fi
TEST='tests/unit/test_m49_periodic_endpoints.py::test_periodic_relative_only_equivariance_training_and_sampling[torch_sparse_triton]'
run_test() {
    singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" \
        --env M49_PERIODIC_CUDA_DIAGNOSTIC=1 --env OMP_NUM_THREADS=4 \
        "${EXTRA_ENV[@]}" "$IMAGE" "$PYTHON" "$@"
}
if [[ "${M49_DIAG_PRECISION_SWEEP:-0}" == 1 ]]; then
    # Same GPU, seed, model, data and original scientific tolerance.
    # Separate Python processes prevent residual precision/cache state.
    # Complete both arms even if the known high-precision-mode gate fails.
    HIGH_STATUS=0
    HIGHEST_STATUS=0
    for PRECISION in high highest; do
        echo "[M49 PRECISION SWEEP] precision=$PRECISION"
        STATUS=0
        run_test -c 'import sys, torch, pytest; torch.set_float32_matmul_precision(sys.argv[1]); print("matmul_precision="+torch.get_float32_matmul_precision(), flush=True); sys.exit(pytest.main(sys.argv[2:]))' \
            "$PRECISION" -q -s -rA "$TEST" || STATUS=$?
        echo "[M49 PRECISION SWEEP] precision=$PRECISION pytest_exit=$STATUS"
        if [[ "$PRECISION" == high ]]; then HIGH_STATUS=$STATUS; else HIGHEST_STATUS=$STATUS; fi
    done
    echo "[M49 PRECISION SWEEP] high=$HIGH_STATUS highest=$HIGHEST_STATUS"
    # Diagnostic success requires the full highest arm to pass, not just
    # the initial forward. This does not authorize production training.
    exit "$HIGHEST_STATUS"
else
    run_test -m pytest -q -s -rA "$TEST"
fi
