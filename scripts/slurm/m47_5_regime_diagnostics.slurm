#!/bin/bash
#SBATCH --job-name=m47_5_regime
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# First run the unconditioned arm as a gate:
# sbatch --array=0 scripts/slurm/m47_5_regime_diagnostics.slurm
# Then the two conditioned arms:
# sbatch --array=1-2 scripts/slurm/m47_5_regime_diagnostics.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
TRAIN_ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
OUT_ROOT=${M47_5_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_5_regime_diagnostics}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
STEPS=${M47_5_STEPS:-50}
SAMPLES=${M47_5_SAMPLES:-4}
SEED=${M47_5_SEED:-20261116}
STORE_FIELDS=${M47_5_STORE_FIELDS:-NO}

if [ -n "${M47_CONDITION:-}" ]; then
    echo "Unset M47_CONDITION before launching the fixed three-arm array" >&2
    exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) CONDITION=none ;;
    1) CONDITION=mach ;;
    2) CONDITION=mach_re ;;
    *) echo "Submit with --array=0-2 (or a subset)" >&2; exit 2 ;;
esac
case "$STORE_FIELDS" in YES|NO) ;; *) echo "M47_5_STORE_FIELDS must be YES or NO" >&2; exit 2 ;; esac
for name in "$STEPS" "$SAMPLES" "$SEED"; do
    case "$name" in *[!0-9]*|'') echo "steps/samples/seed must be nonnegative integers" >&2; exit 2 ;; esac
done
test "$STEPS" -ge 1 && test "$SAMPLES" -ge 1
TRAIN="$TRAIN_ROOT/$CONDITION/epoch10"
OUT="$OUT_ROOT/$CONDITION/epoch10_h${STEPS}_n${SAMPLES}_seed${SEED}"
for file in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT" "$REFS" \
    "$TRAIN/last.pt" "$TRAIN/summary.json" "$TRAIN/standardizers.pt" \
    "$TRAIN/run_signature.json" "$TRAIN/split.json" \
    "$TRAIN_ROOT/$CONDITION/m47_4_source_provenance.json"; do
    test -e "$file" || { echo "Missing required path: $file" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }
test -x "$PYTHON" || { echo "Python not executable: $PYTHON" >&2; exit 2; }

cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Refusing dirty checkout" >&2; exit 2; }
mkdir -p "$OUT_ROOT/$CONDITION"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done

echo "[M47.5] mode=$CONDITION revision=$(git rev-parse HEAD)"
echo "[M47.5] checkpoint=$TRAIN/last.pt"
echo "[M47.5] output=$OUT, steps=$STEPS, samples=$SAMPLES, seed=$SEED"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_5_regime_diagnostics.py \
    tests/unit/test_m47_4.py tests/unit/test_m47_generation.py

EXTRA=()
if [ "$STORE_FIELDS" = YES ]; then EXTRA+=(--store-fields); fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_5_regime_diagnostics \
    --run-dir "$TRAIN" --condition "$CONDITION" \
    --reference-dir "$REFS" --archive-root "$DATA_ROOT" \
    --output-dir "$OUT" --steps "$STEPS" \
    --samples "$SAMPLES" --seed "$SEED" "${EXTRA[@]}"

test -s "$OUT/summary.json"
test -s "$OUT/controlled_conditioning.json"
for case in Re54_Ma0.05 Re54_Ma0.2 Re54_Ma0.3; do
    test -s "$OUT/case_$case.json"
done
echo "[M47.5] COMPLETE $CONDITION: $OUT"
