#!/bin/bash
#SBATCH --job-name=m37_m40_3d_eff
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

M37_RUN=${M37_RUN:-${RUN_ROOT}/m37_3d_dinat_dit_flow_1000e_h128_l10_h4_b16}
M40_RUN=${M40_RUN:-${RUN_ROOT}/m40_3d_dinat_local_self_flow_1000e_h128_l10_h4_b16}
SOURCE_CHECKPOINT=${SOURCE_CHECKPOINT:-best.pt}
EVAL_CHECKPOINT=eval_current.pt
OUTPUT_DIR=${OUTPUT_DIR:-${RUN_ROOT}/efficiency_m37_vs_m40_3d_local_self}

cd "$REPO"

git rev-parse HEAD
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for run_dir in "$M37_RUN" "$M40_RUN"; do
    for artifact in "$SOURCE_CHECKPOINT" resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
        test -s "${run_dir}/${artifact}" || {
            echo "ERROR: missing artifact ${run_dir}/${artifact}" >&2
            exit 1
        }
    done
    cp "${run_dir}/${SOURCE_CHECKPOINT}" "${run_dir}/${EVAL_CHECKPOINT}"
done

echo "============================================================"
echo "3D M37 no self versus M40 local-only self: CUDA efficiency"
echo "============================================================"
echo "Both: DiNAT-DiT h128 L10 H4 torch_sparse"
echo "Physical benchmark batch: one 33^3-node volume"
echo "Heun generation: 50 ODE steps, 100 model evaluations"
echo "Training: forwards + backward + AdamW optimizer step"
echo "Output: $OUTPUT_DIR"
echo "============================================================"

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -m pytest \
        tests/unit/test_dinat_dit.py::test_dinat_dit_3d_local_only_torch_sparse_survives_inference_to_training \
        -q

nvidia-smi

singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_dit_efficiency \
        --run "m37_no_self=$M37_RUN" \
        --run "m40_local_self=$M40_RUN" \
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

cat "${OUTPUT_DIR}/efficiency_results.csv"
cat "${OUTPUT_DIR}/efficiency_comparison.csv"
echo "3D M37/M40 CUDA EFFICIENCY COMPLETE: $OUTPUT_DIR"
