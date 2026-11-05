#!/bin/bash
#SBATCH --job-name=m45_4_controls
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M45.4: keep the TWO-layer architecture unchanged; vary ONLY training scope,
# independent training count and seed. The same held-out split is fixed across jobs.
# The optional --array=0-11 matrix covers: 2 scopes x 2 sizes x 3 seeds.set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    IDX=${SLURM_ARRAY_TASK_ID}
    if [ "$IDX" -lt 0 ] || [ "$IDX" -ge 12 ]; then
        echo "ERROR: valid array indices are 0..11" >&2
        exit 2
    fi
    SEED=$((2026 + IDX % 3))
    if [ $(((IDX / 3) % 2)) -eq 0 ]; then SAMPLES=24; else SAMPLES=256; fi
    if [ $(((IDX / 6) % 2)) -eq 0 ]; then SCOPE=patch_owned; else SCOPE=full_all; fi
else
    SEED=${M45_4_SEED:-2026}
    SAMPLES=${M45_4_TRAIN_SAMPLES:-24}
    SCOPE=${M45_4_SCOPE:-patch_owned}
fi
VAL_SEED=${M45_4_VAL_SEED:-7002026}
STEPS=${M45_4_STEPS:-2000}
GENERATED=${M45_4_GENERATED_SAMPLES:-128}
DRAWS=${M45_4_VELOCITY_DRAWS:-4}
RUN_DIAGNOSTICS=${M45_4_RUN_DIAGNOSTICS:-1}
# The analytical oracle is costly at many steps; run in the single-job
# qualification by default, but omit it for the 12-job training matrix.
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    RUN_ORACLE=${M45_4_RUN_ORACLE:-0}
else
    RUN_ORACLE=${M45_4_RUN_ORACLE:-1}
fi
OUT=${M45_4_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_4_controls/job_${SLURM_JOB_ID}}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo "ERROR: refusing dirty GraphAttention checkout" >&2
    exit 2
fi
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || { echo "Missing image or Python" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }

echo "=== M45.4 controls Git=$(git rev-parse HEAD) Slurm=${SLURM_JOB_ID} ==="
echo "Scope=$SCOPE train_samples=$SAMPLES seed=$SEED val_seed=$VAL_SEED steps=$STEPS"
echo "Output=$OUT diagnostics=$RUN_DIAGNOSTICS generated=$GENERATED draws=$DRAWS"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m45_synthetic_pilot.py scripts/m45_checkpoint_diagnostics.py \
    tests/unit/test_m45_4_controls.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_two_scale_hit.py \
    tests/unit/test_m45_synthetic.py \
    tests/unit/test_m45_checkpoint_diagnostics.py \
    tests/unit/test_m45_4_controls.py

echo "=== Matched 2-layer FM training ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_synthetic_pilot \
    --output-dir "$OUT" --device cuda \
    --operator block_average --source-subspace nullspace \
    --coarse-side 4 --refinement 2 --core-side 4 \
    --train-samples "$SAMPLES" --val-samples 8 \
    --cases local shared_mode \
    --steps "$STEPS" --heun-steps 8 --generated-samples 4 \
    --width 16 --heads 4 --seed "$SEED" --val-seed "$VAL_SEED" \
    --training-scope "$SCOPE" \
    --fine-pre-layers 2 --fine-post-layers 2 \
    --coarse-static-layers 2 --coarse-dynamic-layers 2

test -s "$OUT/summary.json"
test -s "$OUT/metrics.csv"
if [ "$RUN_DIAGNOSTICS" = 1 ]; then
    echo "=== Fixed-coarse generation and time-resolved FM error ==="
    ORACLE_FLAGS=()
    if [ "$RUN_ORACLE" = 1 ]; then
        ORACLE_FLAGS=(--oracle-sampling --oracle-heun-steps 8 25 50 100 200)
    fi
    singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
        "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
        --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" --device cuda \
        --generated-samples "$GENERATED" --velocity-noise-draws "$DRAWS" \
        --heun-steps 25 --seed 53177 "${ORACLE_FLAGS[@]}"
    for file in summary.json time_summary.csv generation_metrics.csv spatial_correlations.csv mode_amplitudes.csv; do
        test -s "$OUT/diagnostics/$file"
    done
fi
echo "M45.4 completed: $OUT/summary.json"
