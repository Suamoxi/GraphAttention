#!/bin/bash
#SBATCH --job-name=m49_full_resume
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
NAME=m49_full_dit_fp32_h128_l10_h4_b8_a2_seed42_blocked50
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
exec 9>"$ROOT/training/$NAME.lock"
flock -n 9 || { echo "Run already locked"; exit 2; }
singularity exec --nv -B /home -B /scratch --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" -u -m scripts.resume_m49_full --run-dir "$ROOT/training/$NAME" --max-epochs "${M49_MAX_EPOCHS:-75}"
