#!/bin/bash
#SBATCH --job-name=m47_8_response
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention_runs/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention_runs/%x_%j.err

# All Python execution (syntax gate, pytest, Gaussian oracle, CUDA audit)
# takes place inside this Grace GPU allocation and the same Singularity /
# avbpML environment as M47.7. Submit ONLY with sbatch, not python.
#
# sbatch scripts/slurm/m47_8_response_audit.slurm
# sbatch --export=ALL,M47_8_VAL_INDEX=1 scripts/slurm/m47_8_response_audit.slurm
set -euo pipefail

REPO=${M47_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFS=${M47_CONDITION_REFS:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions/case_references}
TRAIN=${M47_8_TRAIN_DIR:-/scratch/coop/theret/GraphAttention_runs/m47_7_single_regime_training_fast_v1}
ROOT=${M47_8_OUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_8a_response_audit}
EPOCH=${M47_8_EPOCH:-30}
VAL_INDEX=${M47_8_VAL_INDEX:-0}
SEED=${M47_8_SEED:-20261116}
case "$EPOCH" in
    10|30) ;;
    *) echo "M47_8_EPOCH must be 10 or 30" >&2; exit 2 ;;
esac
case "$VAL_INDEX" in
    0|1|2|3|4) ;;
    *) echo "M47_8_VAL_INDEX must be in 0..4" >&2; exit 2 ;;
esac

for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS/Re54_Ma0.2.yaml" \
    "$TRAIN/epoch${EPOCH}.pt" "$TRAIN/standardizers.pt" \
    "$TRAIN/run_signature.json" "$TRAIN/split.json"; do
    test -e "$path" || { echo "Missing prerequisite: $path" >&2; exit 2; }
done
test -x "$PYTHON" || { echo "Python is not executable: $PYTHON" >&2; exit 2; }
test -s "$IMAGE" || { echo "Missing Singularity image: $IMAGE" >&2; exit 2; }

cd "$REPO"
test -z "$(git status --porcelain)" || {
    echo "Dirty GraphAttention checkout; commit or stash changes" >&2
    exit 2
}

OUT="$ROOT/job_${SLURM_JOB_ID}_epoch${EPOCH}_val${VAL_INDEX}"
test ! -e "$OUT" || { echo "Refusing existing output: $OUT" >&2; exit 2; }
mkdir -p "$OUT"
ORACLE="$OUT/full_field_gaussian_oracle.json"
AUDIT="$OUT/frozen_checkpoint_response"

BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then
        BIND+=(-B "$mount:$mount")
    fi
done

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export SINGULARITYENV_OMP_NUM_THREADS="$OMP_NUM_THREADS"

echo "[M47.8] GRACE job=${SLURM_JOB_ID} host=$(hostname)"
echo "[M47.8] git=$(git rev-parse HEAD) epoch=$EPOCH val=$VAL_INDEX seed=$SEED"
echo "[M47.8] image=$IMAGE python=$PYTHON"
echo "[M47.8] checkpoint=$TRAIN output=$OUT"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "[M47.8] Python syntax preflight"
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m py_compile \
    scripts/m47_8_response_math.py \
    scripts/m47_8_response_audit.py \
    scripts/m47_8_gaussian_oracle.py \
    tests/unit/test_m47_8_response.py \
    tests/unit/test_m47_8_gaussian_oracle.py

echo "[M47.8] Physics and response unit tests"
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_8_response.py \
    tests/unit/test_m47_8_gaussian_oracle.py \
    tests/unit/test_m47_7_covariance.py

echo "[M47.8] Exact Gaussian full-field positive control"
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_8_gaussian_oracle \
    --side 33 --fraction 0.018731345684936694 \
    --samples 3 --heun-steps 25 50 100 200 \
    --output "$ORACLE"
test -s "$ORACLE"

echo "[M47.8] Frozen M47.7B cross-component Fourier response"
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u \
    -m scripts.m47_8_response_audit \
    --train-dir "$TRAIN" --archive-root "$DATA" --reference-dir "$REFS" \
    --output-dir "$AUDIT" --epoch "$EPOCH" --validation-index "$VAL_INDEX" \
    --seed "$SEED" \
    --times 0.25 0.5 0.75 \
    --modes 2,2,0 4,3,1 8,6,5 \
    --amplitudes 0.04 0.08 \
    --sampler-steps 25 50 100

test -s "$AUDIT/audit.json"
echo "[M47.8] COMPLETE oracle=$ORACLE audit=$AUDIT/audit.json"
