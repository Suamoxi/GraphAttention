#!/bin/bash
#SBATCH --job-name=m47_7_single
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# Stage 1: sbatch --export=ALL,M47_7_STAGE=10 scripts/slurm/m47_7_single_regime.slurm
# Stage 2: sbatch --export=ALL,M47_7_STAGE=30 scripts/slurm/m47_7_single_regime.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_CONDITION_REFS:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references}
RUN_ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
OUT=${M47_7B_TRAIN_DIR:-/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training}
STAGE=${M47_7_STAGE:-missing}
case "$STAGE" in 10|30) ;; *) echo "Set M47_7_STAGE=10 or 30" >&2; exit 2 ;; esac
FROZEN="$RUN_ROOT/none/epoch10"
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS" "$REFS/Re54_Ma0.2.yaml" \
    "$FROZEN/last.pt" "$FROZEN/standardizers.pt" \
    "$FROZEN/run_signature.json" "$FROZEN/split.json" \
    "$RUN_ROOT/none/m47_4_source_provenance.json"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
if [ "$STAGE" = 30 ]; then
    test -s "$OUT/epoch10.pt" || {
        echo "Stage30 requires completed frozen epoch10 checkpoint" >&2
        exit 2
    }
fi
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
BIND=(-B /home:/home -B /scratch:/scratch)
for path in /archivenfs /archive2; do
    if [ -d "$path" ]; then BIND+=(-B "$path:$path"); fi
done
echo "[M47.7B] stage=$STAGE revision=$(git rev-parse HEAD) out=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_7_covariance.py

ARGS=()
if [ -d "$OUT" ]; then
    test -s "$OUT/last.pt" || { echo "Output exists without resume checkpoint" >&2; exit 2; }
    ARGS+=(--resume)
fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_7_single_regime_train \
    --output-dir "$OUT" --frozen-run-dir "$FROZEN" \
    --reference-dir "$REFS" --archive-root "$DATA" \
    --max-epochs "$STAGE" "${ARGS[@]}"
test -s "$OUT/epoch${STAGE}.pt"
test -s "$OUT/summary.json"
test -s "$OUT/history.json"
test -s "$OUT/run_signature.json"
echo "[M47.7B] COMPLETE stage=$STAGE"
