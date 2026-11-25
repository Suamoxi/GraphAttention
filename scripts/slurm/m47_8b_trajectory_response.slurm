#!/bin/bash
#SBATCH --job-name=m47_8b_paths
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%A_%a.err

# Submit only through Slurm: sbatch --array=0-4 scripts/slurm/m47_8b_trajectory_response.slurm
# All Python, including pytest, executes in Grace + the established Singularity environment.
# No explicit Slurm memory request; cluster defaults apply.
set -euo pipefail

REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_CONDITION_REFS:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references}
TRAIN=${M47_8B_TRAIN_DIR:-/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training_fast_v1}
ROOT=${M47_8B_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_8b_trajectory_response}
EPOCH=${M47_8B_EPOCH:-30}
STEPS=${M47_8B_HEUN_STEPS:-100}
INDEX=${SLURM_ARRAY_TASK_ID:-missing}
case "$INDEX" in 0|1|2|3|4) ;; *) echo "Requires --array=0-4" >&2; exit 2 ;; esac
case "$EPOCH" in 10|30) ;; *) echo "M47_8B_EPOCH must be 10 or 30" >&2; exit 2 ;; esac
case "$STEPS" in 100|200) ;; *) echo "Use M47_8B_HEUN_STEPS=100 or 200" >&2; exit 2 ;; esac

for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS/Re54_Ma0.2.yaml" \
  "$TRAIN/epoch${EPOCH}.pt" "$TRAIN/standardizers.pt" \
  "$TRAIN/run_signature.json" "$TRAIN/split.json"; do
    test -e "$path" || { echo "Missing: $path" >&2; exit 2; }
done
test -s "$IMAGE" && test -x "$PYTHON"
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty repository checkout" >&2; exit 2; }

OUT="$ROOT/job_${SLURM_ARRAY_JOB_ID}_epoch${EPOCH}_val${INDEX}"
test ! -e "$OUT" || { echo "Refusing overwrite: $OUT" >&2; exit 2; }

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export SINGULARITYENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"

echo "[M47.8B] Grace host=$(hostname) job=${SLURM_ARRAY_JOB_ID} task=$INDEX"
echo "[M47.8B] git=$(git rev-parse HEAD) epoch=$EPOCH Heun=$STEPS checkpoint=$TRAIN output=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
  scripts/m47_8b_context.py scripts/m47_8b_trajectory_math.py \
  scripts/m47_8b_trajectory_response.py tests/unit/test_m47_8b_trajectory.py

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
  tests/unit/test_m47_8b_trajectory.py \
  tests/unit/test_m47_8_response.py \
  tests/unit/test_m47_7_covariance.py

singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
  -m scripts.m47_8b_trajectory_response \
  --train-dir "$TRAIN" --archive-root "$DATA" --reference-dir "$REFS" \
  --epoch "$EPOCH" --validation-index "$INDEX" --steps "$STEPS" \
  --times 0.25 0.5 0.75 \
  --modes 2,2,0 4,3,1 8,6,5 --amplitudes 0.04 0.08 \
  --output-dir "$OUT"

test -s "$OUT/audit.json"
echo "[M47.8B] COMPLETE $OUT/audit.json"
