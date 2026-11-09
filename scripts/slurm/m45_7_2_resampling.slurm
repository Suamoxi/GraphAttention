#!/bin/bash
#SBATCH --job-name=m45_7_resample
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%A_%a.err
set -euo pipefail

# Matched full-field synthetic study: 2 architectures x 2 targets x 2
# residual sampling policies. Python/pytest are run only by Slurm workers.
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

TASK=$SLURM_ARRAY_TASK_ID
case "$TASK" in
    0) MODE=static;  SUPERVISION=fm;     POLICY=fixed ;;
    1) MODE=static;  SUPERVISION=fm;     POLICY=resample ;;
    2) MODE=static;  SUPERVISION=oracle; POLICY=fixed ;;
    3) MODE=static;  SUPERVISION=oracle; POLICY=resample ;;
    4) MODE=dynamic; SUPERVISION=fm;     POLICY=fixed ;;
    5) MODE=dynamic; SUPERVISION=fm;     POLICY=resample ;;
    6) MODE=dynamic; SUPERVISION=oracle; POLICY=fixed ;;
    7) MODE=dynamic; SUPERVISION=oracle; POLICY=resample ;;
    *) echo 'Use Slurm array indices 0..7' >&2; exit 2 ;;
esac

SEED=$(printenv M45_7_2_SEED || echo 2028)
SAMPLES=$(printenv M45_7_2_TRAIN_SAMPLES || echo 24)
STEPS=$(printenv M45_7_2_STEPS || echo 2000)
DEPTH=$(printenv M45_7_2_DEPTH || echo 4)
GENERATED=$(printenv M45_7_2_GENERATED_SAMPLES || echo 128)
VELOCITY_DRAWS=$(printenv M45_7_2_VELOCITY_DRAWS || echo 4)
HEUN=$(printenv M45_7_2_HEUN_STEPS || echo 25)
OUT_ROOT=$(printenv M45_7_2_OUT_ROOT || echo /scratch/coop/theret/GraphAttention_runs/m45_7_2_resampling)
OUT="$OUT_ROOT/job_$SLURM_ARRAY_JOB_ID"_"$TASK"_"$MODE"_"$SUPERVISION"_"$POLICY"_"$STEPS"steps"

cd "$REPO"
[ -z "$(git status --porcelain)" ] || {
    echo 'Dirty checkout; refusing mixed-version comparison' >&2; exit 2;
}
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || {
    echo 'Missing Singularity image or Python interpreter' >&2; exit 2;
}
[ ! -e "$OUT" ] || {
    echo "Refusing existing output: $OUT" >&2; exit 2;
}

echo "M45.7.2 Git=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$TASK"
echo "mode=$MODE supervision=$SUPERVISION policy=$POLICY steps=$STEPS"
echo "seed=$SEED train_coarse_fields=$SAMPLES depth=$DEPTH output=$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m45_7_oracle_supervision.py \
    scripts/m45_checkpoint_diagnostics.py \
    tests/unit/test_m45_7_oracle_supervision.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m45_7_oracle_supervision.py \
    tests/unit/test_m45_checkpoint_diagnostics.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_7_oracle_supervision \
    --output-dir "$OUT" --device cuda \
    --case shared_mode --mode "$MODE" --supervision "$SUPERVISION" \
    --residual-policy "$POLICY" \
    --coarse-side 4 --refinement 2 --train-samples "$SAMPLES" \
    --val-samples 8 --steps "$STEPS" --depth "$DEPTH" \
    --width 16 --heads 4 --seed "$SEED" --val-seed 7002026

test -s "$OUT/summary.json"
test -s "$OUT/shared_mode/block_average/$MODE/checkpoint.pt"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
    --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" \
    --device cuda --cases shared_mode --modes "$MODE" \
    --generated-samples "$GENERATED" --velocity-noise-draws "$VELOCITY_DRAWS" \
    --heun-steps "$HEUN" --seed 53177

test -s "$OUT/diagnostics/summary.json"
test -s "$OUT/diagnostics/time_summary.csv"
test -s "$OUT/diagnostics/generation_metrics.csv"
test -s "$OUT/diagnostics/mode_amplitudes.csv"
test -s "$OUT/diagnostics/spatial_correlations.csv"
echo "M45.7.2 completed: $OUT"
