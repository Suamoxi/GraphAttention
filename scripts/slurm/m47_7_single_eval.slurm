#!/bin/bash
#SBATCH --job-name=m47_7_single_eval
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# sbatch --export=ALL,M47_7_EPOCH=10 scripts/slurm/m47_7_single_eval.slurm
# sbatch --export=ALL,M47_7_EPOCH=30 scripts/slurm/m47_7_single_eval.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_CONDITION_REFS:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references}
TRAIN=${M47_7B_TRAIN_DIR:-/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training_fast_v1}
OUT_ROOT=${M47_7B_EVAL_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_evaluation_fast_v1}
EPOCH=${M47_7_EPOCH:-missing}
case "$EPOCH" in 10|30) ;; *) echo "M47_7_EPOCH must be 10 or 30" >&2; exit 2 ;; esac
OUT="$OUT_ROOT/epoch$EPOCH"
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS" \
    "$TRAIN/epoch$EPOCH.pt" "$TRAIN/standardizers.pt" \
    "$TRAIN/run_signature.json" "$TRAIN/split.json"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
mkdir -p "$OUT_ROOT"
BIND=(-B /home:/home -B /scratch:/scratch)
for path in /archivenfs /archive2; do
    if [ -d "$path" ]; then BIND+=(-B "$path:$path"); fi
done
echo "[M47.7B eval] epoch=$EPOCH revision=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_7_covariance.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_7_single_regime_eval \
    --epoch "$EPOCH" --train-dir "$TRAIN" \
    --archive-root "$DATA" --reference-dir "$REFS" \
    --output-dir "$OUT"
test -s "$OUT/summary.json"
test -s "$OUT/case_Re54_Ma0.2_covariance.json"
echo "[M47.7B eval] COMPLETE epoch=$EPOCH"
