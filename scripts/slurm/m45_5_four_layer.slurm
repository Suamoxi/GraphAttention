#!/bin/bash
#SBATCH --job-name=m45_5_four_layer
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M45.5: separate 4-layer architecture ablation. Retain job 491707's data,
# seed, normalization, optimizer updates and FM settings for a depth comparison.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT=${M45_5_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_5_four_layer/job_${SLURM_JOB_ID}}
STEPS=${M45_5_STEPS:-2000}
WIDTH=${M45_5_WIDTH:-16}
DEPTH=${M45_5_DEPTH:-4}
GENERATED=${M45_5_GENERATED_SAMPLES:-128}
DRAWS=${M45_5_VELOCITY_DRAWS:-4}
RUN_DIAGNOSTICS=${M45_5_RUN_DIAGNOSTICS:-1}
RUN_ORACLE=${M45_5_RUN_ORACLE:-0}
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: refusing dirty GraphAttention checkout" >&2
    exit 2
fi
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || { echo "Missing image or Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
echo "=== M45.5 four-layer comparison Git=$(git rev-parse HEAD) Job=${SLURM_JOB_ID} ==="
echo "Depth=$DEPTH width=$WIDTH updates=$STEPS output=$OUT diagnostics=$RUN_DIAGNOSTICS"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile scripts/m45_synthetic_pilot.py \
    scripts/m45_checkpoint_diagnostics.py tests/unit/test_two_scale_hit.py \
    tests/unit/test_m45_synthetic.py tests/unit/test_m45_4_controls.py \
    tests/unit/test_m45_checkpoint_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_two_scale_hit.py tests/unit/test_m45_synthetic.py \
    tests/unit/test_m45_checkpoint_diagnostics.py tests/unit/test_m45_4_controls.py

echo "=== Four-layer static and dynamic FM training ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_synthetic_pilot \
    --output-dir "$OUT" --device cuda \
    --operator block_average --source-subspace nullspace \
    --coarse-side 4 --refinement 2 --core-side 4 \
    --train-samples 24 --val-samples 8 \
    --cases local shared_mode --steps "$STEPS" \
    --heun-steps 8 --generated-samples 4 \
    --width "$WIDTH" --heads 4 --seed 2026 --val-seed 7002026 \
    --training-scope patch_owned \
    --fine-pre-layers "$DEPTH" --fine-post-layers "$DEPTH" \
    --coarse-static-layers "$DEPTH" --coarse-dynamic-layers "$DEPTH"
test -s "$OUT/summary.json"
test -s "$OUT/metrics.csv"
if [ "$RUN_DIAGNOSTICS" = 1 ]; then
    ORACLE_FLAGS=()
    if [ "$RUN_ORACLE" = 1 ]; then
        ORACLE_FLAGS=(--oracle-sampling --oracle-heun-steps 8 25 50 100 200)
    fi
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
        --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" --device cuda \
        --generated-samples "$GENERATED" --velocity-noise-draws "$DRAWS" \
        --heun-steps 25 --seed 53177 "${ORACLE_FLAGS[@]}"
    test -s "$OUT/diagnostics/generation_metrics.csv"
    test -s "$OUT/diagnostics/time_summary.csv"
fi
echo "M45.5 completed: $OUT/summary.json"
