#!/bin/bash
#SBATCH --job-name=m47_7_tensor
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# sbatch --array=0 scripts/slurm/m47_7_covariance.slurm
# sbatch --array=1-2 scripts/slurm/m47_7_covariance.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_CONDITION_REFS:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references}
RUN_ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
OUT_ROOT=${M47_7A_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_7_covariance_audit}
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) MODE=none ;;
    1) MODE=mach ;;
    2) MODE=mach_re ;;
    *) echo "Use an array with task IDs 0-2" >&2; exit 2 ;;
esac
RUN="$RUN_ROOT/$MODE/epoch10"
OUT="$OUT_ROOT/$MODE/epoch10"
for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS" \
    "$RUN/last.pt" "$RUN/standardizers.pt" "$RUN/run_signature.json" \
    "$RUN/split.json" "$RUN/summary.json" \
    "$RUN_ROOT/$MODE/m47_4_source_provenance.json"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "Refusing overwrite: $OUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
mkdir -p "$OUT_ROOT/$MODE"
BIND=(-B /home:/home -B /scratch:/scratch)
for path in /archivenfs /archive2; do
    if [ -d "$path" ]; then BIND+=(-B "$path:$path"); fi
done
echo "[M47.7A] mode=$MODE revision=$(git rev-parse HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_7_covariance.py \
    tests/unit/test_m47_6_spatial_metrics.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_7_covariance_audit \
    --condition "$MODE" --run-dir "$RUN" \
    --output-dir "$OUT" --archive-root "$DATA" --reference-dir "$REFS"
for case in Re54_Ma0.05 Re54_Ma0.2 Re54_Ma0.3; do
    test -s "$OUT/case_${case}_covariance.json"
done
test -s "$OUT/summary.json"
echo "[M47.7A] COMPLETE mode=$MODE output=$OUT"
