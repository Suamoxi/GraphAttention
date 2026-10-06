#!/bin/bash
#SBATCH --job-name=m37_m38_3d_eff
#SBATCH --time=02:00:00
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

M38_RUN=${M38_RUN:-${RUN_ROOT}/m38_3d_full_dit_flow_1000e_h128_l10_h4_b18}
M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-last.pt}
EVAL_CHECKPOINT=eval_current.pt

cd "$REPO"

for run_dir in "$M38_RUN" "$M37_RUN"; do
    for artifact in "$SOURCE_CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        test -s "${run_dir}/${artifact}" || {
            echo "ERROR: missing training artifact: ${run_dir}/${artifact}"
            exit 1
        }
    done
    cp "${run_dir}/${SOURCE_CHECKPOINT}" "${run_dir}/${EVAL_CHECKPOINT}"
done

M38_EPOCH=$(singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -c \
    "import torch; print(int(torch.load('$M38_RUN/$EVAL_CHECKPOINT', map_location='cpu', weights_only=True)['epoch']))")
M37_EPOCH=$(singularity exec -B /home -B /scratch "$IMAGE" "$PYTHON" -c \
    "import torch; print(int(torch.load('$M37_RUN/$EVAL_CHECKPOINT', map_location='cpu', weights_only=True)['epoch']))")
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/efficiency_m38e${M38_EPOCH}_vs_m37e${M37_EPOCH}_3d_flow}

echo "============================================================"
echo "M38 / M37 controlled 3-D DiT efficiency benchmark"
echo "============================================================"
echo "M38:            Full DiT / optimized SDPA"
echo "M37:            NAT-DiT / torch_sparse backend"
echo "Physical batch: 1 full 33^3-node volume"
echo "Flow generation:50-step Heun (100 model evaluations)"
echo "Timing scope:   model compute only"
echo "M38 checkpoint: $EVAL_CHECKPOINT, epoch $M38_EPOCH"
echo "M37 checkpoint: $EVAL_CHECKPOINT, epoch $M37_EPOCH"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

echo
echo "===== NAT backend correctness gate ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_matches_scatter \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_backend_supports_inference_mode_tensors \
        tests/unit/test_dinat_dit.py::test_dinat_dit_torch_sparse_cache_survives_inference_then_training \
        -q

nvidia-smi
nvidia-smi \
    --query-gpu=name,uuid,memory.total,driver_version,pstate,clocks.sm,clocks.mem \
    --format=csv

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_dit_efficiency \
        --run "full_dit=${M38_RUN}" \
        --run "dinat_torch_sparse=${M37_RUN}" \
        --attention-backend "dinat_torch_sparse=torch_sparse" \
        --output-dir "$OUTPUT_DIR" \
        --checkpoint "$EVAL_CHECKPOINT" \
        --device cuda \
        --batch-size 1 \
        --num-workers 0 \
        --forward-warmup 3 \
        --forward-steps 10 \
        --train-warmup 2 \
        --train-steps 5 \
        --generation-warmup 0 \
        --generation-repeats 1 \
        --sampling-steps 50 \
        --sampling-seed 5678 \
        --overwrite

for artifact in efficiency_summary.json efficiency_results.csv efficiency_comparison.csv; do
    test -s "${OUTPUT_DIR}/${artifact}"
done

echo
echo "===== 3-D efficiency results ====="
cat "${OUTPUT_DIR}/efficiency_results.csv"
echo
echo "===== NAT relative to Full DiT ====="
cat "${OUTPUT_DIR}/efficiency_comparison.csv"
