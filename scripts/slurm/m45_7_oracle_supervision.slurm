#!/bin/bash
#SBATCH --job-name=m45_7_oracle
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%A_%a.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%A_%a.err
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
TASK=$SLURM_ARRAY_TASK_ID
if [ "$TASK" -lt 0 ] || [ "$TASK" -ge 4 ]; then
    echo 'Use M45.7.1 array indices 0..3' >&2
    exit 2
fi
if [ $((TASK % 2)) -eq 0 ]; then SUPERVISION=fm; else SUPERVISION=oracle; fi
if [ "$TASK" -lt 2 ]; then MODE=static; else MODE=dynamic; fi

SEED=$(printenv M45_7_SEED || echo 2028)
SAMPLES=$(printenv M45_7_TRAIN_SAMPLES || echo 24)
STEPS=$(printenv M45_7_STEPS || echo 2000)
DRAWS=$(printenv M45_7_GENERATED_SAMPLES || echo 128)
DIAG_DRAWS=$(printenv M45_7_VELOCITY_DRAWS || echo 4)
HEUN=$(printenv M45_7_HEUN_STEPS || echo 25)
DEPTH=$(printenv M45_7_DEPTH || echo 4)
OUTPUT_ROOT=$(printenv M45_7_OUT_ROOT || echo /scratch/coop/theret/GraphAttention_runs/m45_7_oracle)
OUT="$OUTPUT_ROOT/job_"$SLURM_ARRAY_JOB_ID"_"$TASK"_"$MODE"_"$SUPERVISION
cd "$REPO"

[ -z "$(git status --porcelain)" ] || { echo 'Dirty checkout' >&2; exit 2; }
[ -s "$IMAGE" ] && [ -x "$PYTHON" ] || {
    echo 'Missing Singularity image or Python interpreter' >&2
    exit 2
}
[ ! -e "$OUT" ] || { echo "Refusing existing output: $OUT" >&2; exit 2; }
echo "M45.7.1 Git=$(git rev-parse HEAD) job=$SLURM_JOB_ID task=$TASK mode=$MODE supervision=$SUPERVISION"
echo "Training: samples=$SAMPLES steps=$STEPS depth=$DEPTH seed=$SEED"
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
    --coarse-side 4 --refinement 2 --train-samples "$SAMPLES" \
    --val-samples 8 --steps "$STEPS" --depth "$DEPTH" \
    --width 16 --heads 4 --seed "$SEED" --val-seed 7002026

test -s "$OUT/summary.json"
test -s "$OUT/shared_mode/block_average/$MODE/checkpoint.pt"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
    --checkpoint-root "$OUT" --output-dir "$OUT/diagnostics" \
    --device cuda --cases shared_mode --modes "$MODE" \
    --generated-samples "$DRAWS" --velocity-noise-draws "$DIAG_DRAWS" \
    --heun-steps "$HEUN" --seed 53177

test -s "$OUT/diagnostics/generation_metrics.csv"
test -s "$OUT/diagnostics/time_summary.csv"
test -s "$OUT/diagnostics/summary.json"
echo "M45.7.1 complete: $OUT"
