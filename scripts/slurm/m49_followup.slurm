#!/bin/bash
#SBATCH --job-name=m49_followup
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
cd "${M49_REPO:-/scratch/coop/theret/GraphAttention}"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
OUT="$ROOT/followup/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT"
run() { singularity exec --nv -B /home -B /scratch "$IMAGE" "$PYTHON" "$@"; }
git rev-parse HEAD
case "$SLURM_ARRAY_TASK_ID" in
0)
    # Run in separate processes to measure repeatability without relaxing tolerances.
    failures=0
    for i in 1 2 3 4 5; do
        run -m pytest -q -s 'tests/unit/test_triton_sparse_backward.py::test_triton_geometric_attention_outputs_parameters_and_bias[53-True]' --junitxml="$OUT/repeat_$i.xml" 2>&1 | tee "$OUT/repeat_$i.log" || failures=$((failures+1))
    done
    echo "failed_repetitions=$failures" | tee "$OUT/status.txt"
    test "$failures" -eq 0
    ;;
1) run -u -m scripts.audit_m49_hit --output "$OUT/inventory.json";;
*) exit 2;;
esac
