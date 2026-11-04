#!/bin/bash
#SBATCH --job-name=m45_3_checkpoint_diag
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Evaluate SAVED two-layer checkpoints; no retraining and no dnstoles changes.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
INPUT=${M45_3_INPUT:-/scratch/coop/theret/GraphAttention_runs/m45_2_two_layer/job_491707}
OUT=${M45_3_OUT:-/scratch/coop/theret/GraphAttention_runs/m45_3_diagnostics/job_${SLURM_JOB_ID}}
GENERATED=${M45_3_GENERATED_SAMPLES:-128}
DRAWS=${M45_3_VELOCITY_DRAWS:-4}
SEED=${M45_3_SEED:-53177}
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
    echo 'ERROR: refusing dirty GraphAttention checkout' >&2
    git status --short >&2
    exit 2
fi
[ -s "$IMAGE" ] || { echo "Missing Singularity image: $IMAGE" >&2; exit 2; }
[ -x "$PYTHON" ] || { echo "Missing Python interpreter: $PYTHON" >&2; exit 2; }
[ -s "$INPUT/summary.json" ] || { echo "Missing M45.2 summary: $INPUT/summary.json" >&2; exit 2; }
[ ! -e "$OUT" ] || { echo "Refusing to overwrite: $OUT" >&2; exit 2; }
echo "M45.3: $(hostname), job=${SLURM_JOB_ID}, git=$(git rev-parse HEAD)"
echo "Checkpoints=$INPUT Output=$OUT samples=$GENERATED draws=$DRAWS seed=$SEED"
nvidia-smi --query-gpu=name,memory.total --format=csv
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m py_compile \
    scripts/m45_checkpoint_diagnostics.py \
    src/graph_attention/data/two_scale_synthetic.py \
    tests/unit/test_m45_checkpoint_diagnostics.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -m pytest -q \
    tests/unit/test_m45_checkpoint_diagnostics.py \
    tests/unit/test_m45_synthetic.py \
    tests/unit/test_two_scale_hit.py
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m45_checkpoint_diagnostics \
        --checkpoint-root "$INPUT" --output-dir "$OUT" --device cuda \
        --generated-samples "$GENERATED" --velocity-noise-draws "$DRAWS" \
        --heun-steps 8 25 50 --seed "$SEED"
for f in summary.json time_resolved.csv time_summary.csv generation_metrics.csv spatial_correlations.csv mode_amplitudes.csv; do
    test -s "$OUT/$f"
done
echo "M45.3 completed: $OUT/summary.json"
