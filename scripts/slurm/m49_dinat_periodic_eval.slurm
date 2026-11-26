#!/bin/bash
#SBATCH --job-name=m49_periodic_eval
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --array=0-5%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
RUN="$ROOT/training/m49_dinat_periodic_relative_fp32_h128_l10_h4_b8_a2_seed42_blocked50"
EPOCHS=(25 100 250 500 750 1000)
EPOCH=${EPOCHS[$SLURM_ARRAY_TASK_ID]}
STEPS=50
cd "$REPO"
test -d .git && test -z "$(git status --porcelain)" || { echo "Use a clean standalone checkout"; exit 2; }
BIND=(-B /home -B /scratch)
test "$(singularity exec "${BIND[@]}" "$IMAGE" git -C "$REPO" rev-parse HEAD)" = "$(git rev-parse HEAD)"
singularity exec --nv "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -u -m scripts.evaluate_m49_generation --run-dir "$RUN" --epoch "$EPOCH" --steps "$STEPS" --boundary
