#!/bin/bash
#SBATCH --job-name=m45_2_two_layer
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M45.2: two actual local/exact-two-hop DiNAT layers per backbone stage.
# Compare static/dynamic using the SAME synthetic data and FM settings as 491555.
# No physical DNS/LES validation. Preserve the earlier one-layer Slurm job.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${M45_2_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_2_two_layer/job_${SLURM_JOB_ID}}
OPERATOR=${M45_2_OPERATOR:-block_average}
R=${M45_2_REFINEMENT:-2}
COARSE_SIDE=${M45_2_COARSE_SIDE:-4}
CORE_SIDE=${M45_2_CORE_SIDE:-4}
WIDTH=${M45_2_WIDTH:-16}
DEPTH=${M45_2_LAYERS:-2}
STEPS=${M45_2_STEPS:-2000}
TRAIN_SAMPLES=${M45_2_TRAIN_SAMPLES:-24}
VAL_SAMPLES=${M45_2_VAL_SAMPLES:-8}
GENERATED_SAMPLES=${M45_2_GENERATED_SAMPLES:-4}
HEUN_STEPS=${M45_2_HEUN_STEPS:-8}
SEED=${M45_2_SEED:-2026}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo 'ERROR: refusing dirty GraphAttention checkout' >&2
    git status --short >&2
    exit 2
fi
[ -s "$IMAGE" ] || { echo "Missing image: $IMAGE" >&2; exit 2; }
[ -x "$PYTHON" ] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }

printf '=== M45.2: two-layer DiNAT static/dynamic comparison ===\n'
echo "Host=$(hostname) Slurm=${SLURM_JOB_ID} Git=$(git rev-parse HEAD)"
echo "Output=$OUT operator=$OPERATOR r=$R coarse_side=$COARSE_SIDE core_side=$CORE_SIDE"
echo "width=$WIDTH all_backbone_depth=$DEPTH steps=$STEPS seed=$SEED"
echo "train/val/generated=$TRAIN_SAMPLES/$VAL_SAMPLES/$GENERATED_SAMPLES heun=$HEUN_STEPS"
nvidia-smi --query-gpu=name,memory.total --format=csv

printf '=== Syntax gate ===\n'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
        src/graph_attention/models/two_scale_dinat.py \
        src/graph_attention/geometry/two_scale_partition.py \
        scripts/m45_synthetic_pilot.py \
        tests/unit/test_two_scale_hit.py \
        tests/unit/test_m45_synthetic.py

printf '=== Real M37: depth-1/depth-2 global vs patch forward and gradient parity ===\n'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
        tests/unit/test_two_scale_hit.py \
        tests/unit/test_m45_synthetic.py

printf '=== CUDA: depth-2 synthetic Flow Matching training and Heun generation ===\n'
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_synthetic_pilot \
        --output-dir "$OUT" --device cuda \
        --operator "$OPERATOR" --source-subspace nullspace \
        --coarse-side "$COARSE_SIDE" --refinement "$R" --core-side "$CORE_SIDE" \
        --train-samples "$TRAIN_SAMPLES" --val-samples "$VAL_SAMPLES" \
        --cases local shared_mode \
        --steps "$STEPS" --heun-steps "$HEUN_STEPS" --generated-samples "$GENERATED_SAMPLES" \
        --width "$WIDTH" --heads 4 --seed "$SEED" \
        --fine-pre-layers "$DEPTH" --fine-post-layers "$DEPTH" \
        --coarse-static-layers "$DEPTH" --coarse-dynamic-layers "$DEPTH"

test -s "$OUT/summary.json"
test -s "$OUT/metrics.csv"
echo "M45.2 summary: $OUT/summary.json"
echo "M45.2 metrics: $OUT/metrics.csv"
