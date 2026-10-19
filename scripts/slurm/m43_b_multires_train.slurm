#!/bin/bash
#SBATCH --job-name=m43_b_multires_train
#SBATCH --time=16:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# SCIENTIFIC TRAINING (requires audited M3.3 references and prior smoke pass).
# This launcher is deliberately opt-in, never automatically run by the gate:
#   sbatch --export=ALL,M43_B_CONFIRM_LONG_TRAIN=YES scripts/slurm/m43_b_multires_train.slurm
# To resume a time-limited run: export M43_B_RESUME=1 and increase M43_B_EPOCHS.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
DATA_ROOT=${M43_B_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REF_DIR=${M43_B_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
OUT=${M43_B_TRAIN_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m43_b_multires/train_s42}
EPOCHS=${M43_B_EPOCHS:-2}
RESUME=${M43_B_RESUME:-0}
CONFIRM=${M43_B_CONFIRM_LONG_TRAIN:-NO}

if [ "$CONFIRM" != YES ]; then
    echo "Refusing long scientific training without M43_B_CONFIRM_LONG_TRAIN=YES" >&2
    exit 2
fi
if [ "$RESUME" != 0 ] && [ "$RESUME" != 1 ]; then
    echo "M43_B_RESUME must be 0 or 1" >&2
    exit 2
fi
if [ "$EPOCHS" -lt 1 ]; then
    echo "M43_B_EPOCHS must be >=1" >&2
    exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "Refusing dirty code checkout" >&2
    git status --short >&2
    exit 2
fi
test -d "$DATA_ROOT"
for name in \
    Re54_Ma0.05 Re94_Ma0.05 \
    Re54_Ma0.2 Re94_Ma0.2 \
    Re54_Ma0.3 Re94_Ma0.3; do
    test -s "$REF_DIR/$name.yaml" || {
        echo "Missing audited M3.3 case reference: $REF_DIR/$name.yaml" >&2
        exit 2
    }
done
if [ "$RESUME" = 0 ] && [ -e "$OUT" ]; then
    echo "Output already exists, refusing accidental overwrite: $OUT" >&2
    exit 2
fi
if [ "$RESUME" = 1 ] && [ ! -f "$OUT/last.pt" ]; then
    echo "Resume requires prior last.pt: $OUT" >&2
    exit 2
fi

echo "==== M43 Experiment B six-case scientific training ===="
echo "Host $(hostname), Git $(git rev-parse HEAD), Job $SLURM_JOB_ID"
echo "Output $OUT, epochs $EPOCHS, resume $RESUME"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== Targeted correctness gates ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m43_b_multires.py \
    tests/unit/test_m43_b1b_checkpointing.py

RESUME_ARGS=()
if [ "$RESUME" = 1 ]; then RESUME_ARGS=(--resume); fi
echo "==== Train shared M37 with checkpointed 65/129^3 microbatches ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.train_m43_b_multires \
    --mode train \
    --root "$DATA_ROOT" \
    --reference-dir "$REF_DIR" \
    --run-dir "$OUT" \
    --seed 42 \
    --temporal-embargo 2 \
    --accumulation 6 \
    --max-epochs "$EPOCHS" \
    "${RESUME_ARGS[@]}"

echo "Saved last.pt best.pt history.csv summary.json in $OUT"
