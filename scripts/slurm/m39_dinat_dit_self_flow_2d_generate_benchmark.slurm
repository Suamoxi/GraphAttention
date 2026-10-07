#!/bin/bash
#SBATCH --job-name=m39_dinat_self_gen
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

RUN_NAME=m39_dinat_dit_self_flow_tscale1000_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt
BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=50
SAMPLING_SEED=5678
GEN_NAME=flow_heun_steps50_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}
BENCH_ROOT=${RUN_ROOT}/benchmark_m39_dinat_dit_self_flow
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M39 - 2-D DiNAT-DiT self-attention generation + benchmark"
echo "============================================================"
echo "Checkpoint:        best.pt"
echo "Attention local:   self + one-hop"
echo "Attention dilated: self + exact-two-hop"
echo "Sampler:           Flow ODE / Heun"
echo "Steps:             $SAMPLING_STEPS"
echo "Seed:              $SAMPLING_SEED"
echo "Reference control: M33/M34 no-self DiNAT-DiT"
echo "============================================================"

git rev-parse HEAD
git status --short
if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

for artifact in "$CHECKPOINT" summary.json resolved_config.yaml standardizers.pt dataset_split_manifest.json; do
    test -s "${RUN_DIR}/${artifact}" || {
        echo "ERROR: missing M39 training artifact: ${RUN_DIR}/${artifact}"
        exit 1
    }
done

if [ ! -s "${GEN_DIR}/generated_test.pt" ] || [ ! -s "${GEN_DIR}/summary.json" ]; then
    overwrite=false
    if [ -e "$GEN_DIR" ]; then
        overwrite=true
    fi

    nvidia-smi
    singularity exec --nv \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.generate_generative \
            run_dir="$RUN_DIR" \
            checkpoint="$CHECKPOINT" \
            device=cuda \
            batch_size="$BATCH_SIZE" \
            num_workers="$NUM_WORKERS" \
            overwrite="$overwrite" \
            sampling.steps="$SAMPLING_STEPS" \
            sampling.method=flow_ode \
            sampling.solver=heun \
            sampling.sampling_eps=0.0 \
            sampling.final_denoise=false \
            sampling.seed="$SAMPLING_SEED" \
            output_name="$GEN_NAME"
else
    echo "M39 generation already exists; reusing $GEN_DIR"
fi

for artifact in generated_test.pt summary.json resolved_config.yaml generation_config.yaml dataset_split_manifest.json; do
    test -s "${GEN_DIR}/${artifact}"
done

singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$GEN_DIR" \
        output_root="$BENCH_ROOT" \
        overwrite=true \
        energy_spectrum.enabled=true \
        energy_spectrum.lower_quantile=0.10 \
        energy_spectrum.upper_quantile=0.90 \
        plots.energy_spectrum=true

for artifact in \
    summary.json \
    channel_metrics.csv \
    standardized_wasserstein.csv \
    spectral_bands.csv \
    physical_metrics.csv \
    energy_spectrum_per_sample.csv \
    energy_spectrum_summary.csv \
    energy_spectral_bands.csv
do
    test -s "${BENCH_DIR}/${artifact}"
done

echo
echo "===== M39 standardized Wasserstein ====="
cat "${BENCH_DIR}/standardized_wasserstein.csv"

echo
echo "===== M39 energy spectrum summary ====="
cat "${BENCH_DIR}/energy_spectrum_summary.csv"

echo
echo "M39 benchmark: $BENCH_DIR"
echo "No-self references:"
echo "  M33: ${RUN_ROOT}/benchmark_m33_dinat_dit_flow/${GEN_NAME}"
echo "  M34 optimized training: ${RUN_ROOT}/m34_dinat_dit_flow_optimized_retry_1000e_h128_l10_h4"
