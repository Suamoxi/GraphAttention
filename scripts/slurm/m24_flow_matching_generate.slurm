#!/bin/bash
#SBATCH --job-name=m24_flow_generate
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3

RUN_ROOT=/scratch/coop/theret/GraphAttention_runs
RUN_NAME=m24_m12_flow_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=50
SOLVER=heun
SAMPLING_SEED=5678

GEN_NAME=flow_heun_steps50_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M24 - Flow Matching generation only"
echo "============================================================"
echo "Source run:    $RUN_DIR"
echo "Checkpoint:    $CHECKPOINT"
echo "Output:        $GEN_DIR"
echo "Sampler:       flow ODE"
echo "Steps:         $SAMPLING_STEPS"
echo "Solver:        $SOLVER"
echo "Sampling seed: $SAMPLING_SEED"
echo "Expected NFE:  $((2 * SAMPLING_STEPS))"
echo "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

echo
echo "===== Checking source run ====="
for artifact in     best.pt     summary.json     resolved_config.yaml     standardizers.pt     dataset_split_manifest.json
do
    if [ ! -f "${RUN_DIR}/${artifact}" ]; then
        echo "ERROR: missing source artifact: ${RUN_DIR}/${artifact}"
        exit 1
    fi
done

if [ -e "$GEN_DIR" ]; then
    echo "ERROR: generation output already exists: $GEN_DIR"
    echo "Refusing to overwrite it."
    exit 1
fi

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Flow Matching generation ====="
singularity exec --nv     -B /home     -B /scratch     "$IMAGE"     "$PYTHON" -u -m scripts.generate_generative         run_dir="$RUN_DIR"         checkpoint="$CHECKPOINT"         device=cuda         batch_size="$BATCH_SIZE"         num_workers="$NUM_WORKERS"         sampling.steps="$SAMPLING_STEPS"         sampling.method=flow_ode         sampling.solver="$SOLVER"         sampling.sampling_eps=0.0         sampling.final_denoise=false         sampling.seed="$SAMPLING_SEED"         output_name="$GEN_NAME"

echo
echo "===== Checking generation outputs ====="
test -f "${GEN_DIR}/generated_test.pt"
test -f "${GEN_DIR}/summary.json"
test -f "${GEN_DIR}/resolved_config.yaml"
test -f "${GEN_DIR}/generation_config.yaml"
test -f "${GEN_DIR}/dataset_split_manifest.json"

echo
echo "===== Generation summary ====="
cat "${GEN_DIR}/summary.json"

echo
echo "============================================================"
echo "M24 FLOW GENERATION COMPLETE"
echo "$GEN_DIR"
echo "============================================================"
