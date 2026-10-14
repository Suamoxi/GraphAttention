#!/bin/bash
#SBATCH --job-name=m41_4_257_infer
#SBATCH --time=04:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Stage 3: one measured forward on 257^3 nodes; then optionally Heun.
# No explicit --mem limit. 50-step Heun can span multiple Slurm jobs:
# progress.json + state_step_N.npy ensure resumable 2-NFE steps.
set -euo pipefail
REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
GRAPH_ROOT=${GRAPH_ROOT:-${RUN_ROOT}/m41_4_257_generic_graph}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
CHECKPOINT=${CHECKPOINT:-best.pt}
MODE=${M41_4_MODE:-forward}
HEUN_STEPS=${M41_4_HEUN_STEPS:-50}
STEPS_THIS_JOB=${M41_4_STEPS_THIS_JOB:-1}
PRELOAD=${M41_4_PRELOAD:-1}
SEED=${M41_4_SEED:-5678}
OUTPUT=${OUTPUT:-${RUN_ROOT}/m41_4_257_inference/${MODE}}
cd "$REPO"
test -s "$GRAPH_ROOT/prepare.json" || {
    echo "Generic graph not prepared. Run m41_4_generic_257_prepare.slurm first." >&2
    exit 1
}
for name in "$CHECKPOINT" resolved_config.yaml standardizers.pt; do
    test -s "$M37_RUN/$name" || { echo "Missing M37 run artifact $name" >&2; exit 1; }
done
echo "M41.4 on host $(hostname), commit $(git rev-parse HEAD)"
echo "Mode=$MODE, prepared=$GRAPH_ROOT, checkpoint=$M37_RUN/$CHECKPOINT"
nvidia-smi
ARGS=()
if [ "$PRELOAD" = "1" ]; then ARGS=(--preload-partitions); fi
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
    "$PYTHON" -u -m scripts.m41_4_generic_large infer \
    --prepared-dir "$GRAPH_ROOT" --run-dir "$M37_RUN" \
    --checkpoint "$CHECKPOINT" --mode "$MODE" \
    --output-dir "$OUTPUT" --seed "$SEED" \
    --steps "$HEUN_STEPS" --max-steps-this-run "$STEPS_THIS_JOB" \
    "${ARGS[@]}"
echo "Result: $OUTPUT"
