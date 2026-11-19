#!/bin/bash
#SBATCH --job-name=m47_6_spatial
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# Gate with unconditioned array 0. Only then submit 1-2.
set -euo pipefail
REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
ENV_ROOT=${M47_ENV_ROOT:-/scratch/coop/theret/avbpML_env}
PYTHON=${M47_PYTHON:-$ENV_ROOT/bin/python3}
RUN_ROOT=${M47_4_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_4_longer_training}
OUT_ROOT=${M47_6_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_6_spatial_physics}
REFERENCE_ROOT=${M47_REF_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_CONDITION_REFS:-$REFERENCE_ROOT/case_references}
DATA_ROOT=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
SAMPLES=${M47_6_SAMPLES:-4}
COMPARE_COUNT=${M47_6_COMPARE_COUNT:-2}
SEED=${M47_6_SEED:-20261116}

if [ -n "${M47_CONDITION:-}" ]; then
  echo "unset M47_CONDITION to avoid overriding array selection" >&2
  exit 2
fi
case "${SLURM_ARRAY_TASK_ID:-missing}" in
  0) CONDITION=none ;;
  1) CONDITION=mach ;;
  2) CONDITION=mach_re ;;
  *) echo "Use --array=0-2" >&2; exit 2 ;;
esac
for integer in "$SAMPLES" "$COMPARE_COUNT" "$SEED"; do
  case "$integer" in *[!0-9]*|'') echo "integer arguments required" >&2; exit 2 ;; esac
done
test "$SAMPLES" -ge 2
test "$COMPARE_COUNT" -ge 1
test "$COMPARE_COUNT" -le "$SAMPLES"

RUN="$RUN_ROOT/$CONDITION/epoch10"
OUT="$OUT_ROOT/$CONDITION/epoch10_h50_h100_n${SAMPLES}_seed${SEED}"
for p in "$REPO" "$IMAGE" "$PYTHON" "$DATA_ROOT" "$REFS" \
  "$RUN/last.pt" "$RUN/summary.json" "$RUN/standardizers.pt" \
  "$RUN/run_signature.json" "$RUN/split.json" \
  "$RUN_ROOT/$CONDITION/m47_4_source_provenance.json"; do
  test -e "$p" || { echo "Missing required path: $p" >&2; exit 2; }
done
test -x "$PYTHON"
test ! -e "$OUT" || { echo "Refusing overwrite of: $OUT" >&2; exit 2; }
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Refusing dirty checkout" >&2; exit 2; }
mkdir -p "$OUT_ROOT/$CONDITION"
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
  if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done

echo "[M47.6] condition=$CONDITION git=$(git rev-parse HEAD) output=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
  tests/unit/test_m47_6_spatial_metrics.py \
  tests/unit/test_m47_5_regime_diagnostics.py \
  tests/unit/test_m47_4.py \
  tests/unit/test_m47_generation.py

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
  -m scripts.m47_6_spatial_diagnostics \
  --run-dir "$RUN" --condition "$CONDITION" \
  --reference-dir "$REFS" --archive-root "$DATA_ROOT" \
  --output-dir "$OUT" \
  --steps 50 --compare-steps 100 \
  --samples "$SAMPLES" --compare-count "$COMPARE_COUNT" --seed "$SEED"

for name in summary.json case_Re54_Ma0.05_divcurl.json \
  case_Re54_Ma0.2_divcurl.json case_Re54_Ma0.3_divcurl.json \
  flow_trajectory_Re54_Ma0.2.json heun_step_convergence.json; do
  test -s "$OUT/$name" || { echo "Missing result: $OUT/$name" >&2; exit 1; }
done
echo "[M47.6] COMPLETE: $OUT"
