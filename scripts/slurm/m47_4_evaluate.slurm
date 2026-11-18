#!/bin/bash
#SBATCH --job-name=m47_4_eval
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# sbatch --array=0-2 --export=ALL,M47_4_STAGE=5 scripts/slurm/m47_4_evaluate.slurm
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
REF_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REF_ROOT/case_references}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
STAGE=${M47_4_STAGE:-}
STEPS=${M47_4_EVAL_STEPS:-50}
N=${M47_4_EVAL_SAMPLES:-4}
ALL_CASES=${M47_4_EVAL_ALL_CASES:-NO}
if [ -n "${M47_CONDITION:-}" ]; then
    echo "Unset M47_CONDITION for array execution" >&2; exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
    0) CONDITION=none ;;
    1) CONDITION=mach ;;
    2) CONDITION=mach_re ;;
    *) echo "submit with --array=0-2" >&2; exit 2 ;;
esac
case "$STAGE" in 5|10) ;; *) echo "M47_4_STAGE must be 5 or 10" >&2; exit 2 ;; esac
case "$ALL_CASES" in YES|NO) ;; *) echo "M47_4_EVAL_ALL_CASES must be YES or NO" >&2; exit 2 ;; esac
test "$STEPS" -ge 1 && test "$N" -ge 1
EPOCH=$(printf "%02d" "$STAGE")
TRAIN="$ROOT/$CONDITION/epoch$EPOCH"
OUT="$ROOT/$CONDITION/eval_epoch${EPOCH}_h${STEPS}_n${N}_seed20261116"
for f in "$REPO" "$IMAGE" "$PYTHON" "$REFS" "$DATA_ROOT" \
         "$TRAIN/last.pt" "$TRAIN/summary.json" \
         "$ROOT/$CONDITION/m47_4_source_provenance.json"; do
    test -e "$f" || { echo "Missing: $f" >&2; exit 2; }
done
test ! -e "$OUT" || { echo "Output exists: $OUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty checkout" >&2; exit 2; }
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
echo "M47.4 evaluation mode=$CONDITION epochs=$STAGE samples=$N cases=$ALL_CASES"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_4.py tests/unit/test_m47_fm_diagnostics.py \
    tests/unit/test_m47_generation.py
EXTRA=()
if [ "$ALL_CASES" = YES ]; then
    for case_name in Re54/Ma0.05 Re94/Ma0.05 Re54/Ma0.2 \
                     Re94/Ma0.2 Re54/Ma0.3 Re94/Ma0.3; do
        EXTRA+=(--case "$case_name")
    done
fi
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_4_evaluate \
    --run-dir "$TRAIN" --condition "$CONDITION" --epochs "$STAGE" \
    --archive-root "$DATA_ROOT" --reference-dir "$REFS" \
    --output-dir "$OUT" --steps "$STEPS" --samples-per-case "$N" \
    --validation-per-case 5 --seed 20261116 "${EXTRA[@]}"
echo "M47.4 evaluation result: $OUT/summary.json"
