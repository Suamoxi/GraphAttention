#!/bin/bash
#SBATCH --job-name=m49_restart
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --array=0-1%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
OUT="$ROOT/restart/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
case "$SLURM_ARRAY_TASK_ID" in
    0) ARCH=full ;;
    1) ARCH=dinat ;;
    *) exit 2 ;;
esac
mkdir -p "$OUT"
run() { singularity exec --nv -B /home -B /scratch -B /gpfs-calypso/scratch:/gpfs-calypso/scratch --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" "$@"; }
export M49_REPO="$REPO" M49_ARCH="$ARCH" M49_CHECKPOINT_EVERY=2
export M49_SPLIT_MANIFEST="$OUT/restart_split.json"
run -m pytest -q tests/unit/test_m49_restart_rng.py --junitxml="$OUT/rng_tests.xml"
run -m scripts.check_m49_restart prepare --output "$M49_SPLIT_MANIFEST"
# Every call launches a fresh Python process, including the resumed run.
M49_ROOT="$OUT/reference" M49_MODE=start M49_MAX_EPOCHS=4 bash scripts/slurm/m49_full_dit_train.slurm
M49_ROOT="$OUT/restarted" M49_MODE=start M49_MAX_EPOCHS=2 bash scripts/slurm/m49_full_dit_train.slurm
M49_ROOT="$OUT/restarted" M49_MODE=resume M49_MAX_EPOCHS=4 bash scripts/slurm/m49_full_dit_train.slurm
run -m scripts.check_m49_restart compare --root "$OUT" --arch "$ARCH"
