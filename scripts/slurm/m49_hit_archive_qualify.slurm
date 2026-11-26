#!/bin/bash
#SBATCH --job-name=m49_les_qualify
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%2
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err
set -euo pipefail
REPO=${M49_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M49_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M49_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
ROOT=${M49_ROOT:-/scratch/coop/theret/GraphAttention_runs/m49}
ARCHIVE=${M49_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_LES_FORCED}
CASES=(case1_forced case2_forced case3_forced)
CASE=${CASES[$SLURM_ARRAY_TASK_ID]}
OUT="$ROOT/archive_qualification/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout"; exit 2; }
mkdir -p "$OUT"
BIND=(-B /home:/home -B /scratch:/scratch -B /archivenfs:/archivenfs)
run() { singularity exec "${BIND[@]}" --env "PYTHONPATH=$REPO/src:$REPO" "$IMAGE" "$PYTHON" "$@"; }
run -m pytest -q tests/unit/test_m49_archive_qualification.py --junitxml="$OUT/tests.xml"
run -u -m scripts.qualify_m49_hit_archive --root "$ARCHIVE" --case "$CASE" --output "$OUT/qualification.json"
test -s "$OUT/qualification.json"
