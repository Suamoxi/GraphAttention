#!/bin/bash
#SBATCH --job-name=m43_b_multires_gate
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Default: read-only archive inspection, NO model training.
# To run one checkpointed six-case optimizer step after verifying
# all physical reference YAMLs, submit with:
#   sbatch --export=ALL,M43_B_MODE=smoke scripts/slurm/m43_b_multires_inspect_smoke.slurm
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
DATA_ROOT=${M43_B_DATA_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REF_DIR=${M43_B_REFS:-/scratch/coop/theret/HIT_DNS_COMP_case_references}
MODE=${M43_B_MODE:-inspect}
OUTPUT_ROOT=${M43_B_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m43_b_multires}
OUT="$OUTPUT_ROOT/${MODE}_${SLURM_JOB_ID}"
if [ "$MODE" != inspect ] && [ "$MODE" != smoke ]; then
    echo "This launcher permits only inspect or smoke, not long training" >&2
    exit 2
fi
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: Git checkout must be clean" >&2
    git status --short >&2
    exit 2
fi
echo "==== M43 Experiment B archive/one-step smoke gate ===="
echo "Host $(hostname), Git $(git rev-parse HEAD), mode $MODE"
echo "Data root $DATA_ROOT"
echo "References $REF_DIR"
test -d "$DATA_ROOT" || exit 2
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "==== Syntax and 3D scientific-data regression tests ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/data/multicase_avbp.py \
    src/graph_attention/data/multicase_split.py \
    scripts/train_m43_b_multires.py \
    tests/unit/test_m43_b_multires.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m43_b_multires.py \
    tests/unit/test_m43_b1b_checkpointing.py \
    tests/unit/test_generative_training_adapters.py

echo "==== Execute $MODE ===="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.train_m43_b_multires \
    --mode "$MODE" \
    --root "$DATA_ROOT" \
    --reference-dir "$REF_DIR" \
    --run-dir "$OUT" \
    --seed 42 \
    --temporal-embargo 2 \
    --accumulation 6

echo "==== Output ===="
if [ "$MODE" = inspect ]; then
    cat "$OUT/inventory.json"
else
    cat "$OUT/summary.json"
fi
