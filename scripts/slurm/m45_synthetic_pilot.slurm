#!/bin/bash
#SBATCH --job-name=m45_synthetic_pilot
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M45 synthetic-only static/dynamic DiNAT FM qualification.
# No real DNS, no physical dnstoles Q/B, no filtering claim.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${M45_SYNTHETIC_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_synthetic/job_${SLURM_JOB_ID}}
OPERATOR=${M45_SYNTHETIC_OPERATOR:-block_average}
R=${M45_SYNTHETIC_REFINEMENT:-2}
COARSE_SIDE=${M45_SYNTHETIC_COARSE_SIDE:-4}
CORE_SIDE=${M45_SYNTHETIC_CORE_SIDE:-4}
STEPS=${M45_SYNTHETIC_STEPS:-32}
SOURCE=${M45_SYNTHETIC_SOURCE_SUBSPACE:-nullspace}
WIDTH=${M45_SYNTHETIC_WIDTH:-16}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo 'ERROR: refusing dirty GraphAttention checkout' >&2
    git status --short >&2
    exit 2
fi
[ -s "$IMAGE" ] || { echo "Missing image: $IMAGE" >&2; exit 2; }
[ -x "$PYTHON" ] || { echo "Missing Python: $PYTHON" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }

echo '=== M45 synthetic qualification ==='
echo "Host=$(hostname) Slurm=${SLURM_JOB_ID} Git=$(git rev-parse HEAD)"
echo "Output=$OUT operator=$OPERATOR r=$R coarse_side=$COARSE_SIDE core_side=$CORE_SIDE"
echo "steps=$STEPS source=$SOURCE width=$WIDTH"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo '=== Python syntax ==='
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    src/graph_attention/data/two_scale_synthetic.py \
    scripts/m45_synthetic_pilot.py \
    tests/unit/test_m45_synthetic.py

echo '=== Existing M45 and synthetic Q/B tests ==='
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q tests/unit/test_two_scale_hit.py tests/unit/test_m45_synthetic.py

echo '=== Synthetic FM training and small-graph Heun generation on CUDA ==='
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_synthetic_pilot \
        --output-dir "$OUT" --device cuda \
        --operator "$OPERATOR" --source-subspace "$SOURCE" \
        --coarse-side "$COARSE_SIDE" --refinement "$R" --core-side "$CORE_SIDE" \
        --train-samples 24 --val-samples 8 \
        --cases local shared_mode \
        --steps "$STEPS" --heun-steps 8 --generated-samples 4 \
        --width "$WIDTH" --heads 4 --seed 2026

test -s "$OUT/summary.json"
test -s "$OUT/metrics.csv"
echo "M45 synthetic summary: $OUT/summary.json"
echo "M45 synthetic metrics: $OUT/metrics.csv"
