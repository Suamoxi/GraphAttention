#!/bin/bash
#SBATCH --job-name=m49_precision
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
case "$SLURM_ARRAY_TASK_ID" in
  0) B=1 ;;
  1) B=8 ;;
  *) exit 2 ;;
esac
OUT="$ROOT/precision/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$OUT"
run() { singularity exec --nv -B /home -B /scratch --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" "$@"; }
git rev-parse HEAD
nvidia-smi
run -m pytest -q -rs tests/unit/test_m49_full_amp.py --junitxml="$OUT/tests.xml"
run -u -m scripts.benchmark_m49_precision --batch "$B" --output "$OUT/precision.json"
