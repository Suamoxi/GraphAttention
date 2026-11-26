#!/bin/bash
#SBATCH --job-name=m49_depth_bench
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --array=0-4%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
BASE="$ROOT/training/m49_dinat_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
VARIANTS=(10x12 9x123 8x123 20x12 15x123)
VARIANT=${VARIANTS[$SLURM_ARRAY_TASK_ID]}
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
singularity exec "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -m pytest -q tests/unit/test_geometry_dilated.py tests/unit/test_m49_exact_three_hop.py tests/unit/test_m49_hop_schedule.py tests/unit/test_m49_relative_only.py
nvidia-smi
singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -u -m scripts.benchmark_m49_depth --variant "$VARIANT" --baseline "$BASE" --output "$ROOT/depth_benchmark/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"
