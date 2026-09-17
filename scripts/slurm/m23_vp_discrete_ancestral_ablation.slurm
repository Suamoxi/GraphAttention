#!/bin/bash
#SBATCH --job-name=m23_vp_ancestral
#SBATCH --time=04:00:00
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
M23_RUN_NAME=m23_m12_vp_sde_1000e_h128_l10_h4
M23_RUN_DIR=${RUN_ROOT}/${M23_RUN_NAME}
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_STEPS=1000
SAMPLING_SEED=5678

ANCESTRAL_NAME=vp_discrete_ancestral_steps1000_seed5678
ANCESTRAL_GEN_DIR=${M23_RUN_DIR}/generations/${ANCESTRAL_NAME}
BENCH_ROOT=${RUN_ROOT}/benchmark_m23_vp_sde
ANCESTRAL_BENCH_DIR=${BENCH_ROOT}/${ANCESTRAL_NAME}

M21_BENCH_DIR=${RUN_ROOT}/benchmark_m21_linear_ddpm/linear_ddpm_ancestral_start1000_steps1000_eta1_seed5678
M23_SDE_BENCH_DIR=${BENCH_ROOT}/vp_reverse_sde_euler_maruyama_steps1000_denoise_eps0p001_seed5678

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
M21_RELATIVE_NAME=m21_vs_m23_discrete_ancestral_1000
M21_RELATIVE_DIR=${RELATIVE_ROOT}/${M21_RELATIVE_NAME}
SDE_RELATIVE_NAME=m23_reverse_sde_vs_discrete_ancestral_1000
SDE_RELATIVE_DIR=${RELATIVE_ROOT}/${SDE_RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M23 - discrete ancestral sampler ablation"
printf '%s\n' "============================================================"
printf 'Source run:       %s\n' "$M23_RUN_DIR"
printf 'Checkpoint:       %s\n' "$CHECKPOINT"
printf 'Sampler:          discrete VP ancestral / DDPM-style\n'
printf 'Sampling steps:   %s\n' "$SAMPLING_STEPS"
printf 'Sampling endpoint:t=0\n'
printf 'Sampling seed:    %s\n' "$SAMPLING_SEED"
printf 'Expected NFE:     %s\n' "$SAMPLING_STEPS"
printf '%s\n' "alpha_bar_i = alpha(i/steps)^2 from the trained continuous VP marginal"
printf '%s\n' "RNG namespace matches M23 reverse-SDE for common-random-number initialization"
printf '%s\n' "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

if [ ! -f "${M23_RUN_DIR}/${CHECKPOINT}" ]; then
    echo "ERROR: M23 checkpoint does not exist: ${M23_RUN_DIR}/${CHECKPOINT}"
    exit 1
fi

for source_dir in "$M21_BENCH_DIR" "$M23_SDE_BENCH_DIR"; do
    if [ ! -f "${source_dir}/summary.json" ]; then
        echo "ERROR: required existing benchmark is missing: ${source_dir}"
        exit 1
    fi
done

for path in \
    "$ANCESTRAL_GEN_DIR" \
    "$ANCESTRAL_BENCH_DIR" \
    "$M21_RELATIVE_DIR" \
    "$SDE_RELATIVE_DIR"
do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists: $path"
        exit 1
    fi
done

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== M23 generation: continuous VP model / discrete ancestral sampler ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_generative \
        run_dir="$M23_RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size="$BATCH_SIZE" \
        num_workers="$NUM_WORKERS" \
        sampling.steps="$SAMPLING_STEPS" \
        sampling.method=discrete_ancestral \
        sampling.solver=ddpm \
        sampling.sampling_eps=0.0 \
        sampling.final_denoise=false \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$ANCESTRAL_NAME"

echo
echo "===== M23 discrete ancestral distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$ANCESTRAL_GEN_DIR" \
        output_root="$BENCH_ROOT"

echo
echo "===== Relative comparison: M21 ancestral vs M23 discrete ancestral ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$M21_BENCH_DIR" \
        candidate_dir="$ANCESTRAL_BENCH_DIR" \
        output_root="$RELATIVE_ROOT" \
        comparison_name="$M21_RELATIVE_NAME"

echo
echo "===== Relative comparison: M23 reverse SDE vs M23 discrete ancestral ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$M23_SDE_BENCH_DIR" \
        candidate_dir="$ANCESTRAL_BENCH_DIR" \
        output_root="$RELATIVE_ROOT" \
        comparison_name="$SDE_RELATIVE_NAME"

echo
echo "===== Checking outputs ====="
for path in \
    "${ANCESTRAL_GEN_DIR}/generated_test.pt" \
    "${ANCESTRAL_GEN_DIR}/summary.json" \
    "${ANCESTRAL_BENCH_DIR}/summary.json" \
    "${ANCESTRAL_BENCH_DIR}/channel_metrics.csv" \
    "${ANCESTRAL_BENCH_DIR}/spectral_bands.csv" \
    "${ANCESTRAL_BENCH_DIR}/physical_metrics.csv" \
    "${M21_RELATIVE_DIR}/summary.json" \
    "${M21_RELATIVE_DIR}/relative_comparison.csv" \
    "${SDE_RELATIVE_DIR}/summary.json" \
    "${SDE_RELATIVE_DIR}/relative_comparison.csv"
do
    test -f "$path"
done

echo
echo "===== M23 discrete ancestral benchmark summary ====="
cat "${ANCESTRAL_BENCH_DIR}/summary.json"

echo
echo "===== M21 vs M23 discrete ancestral ====="
cat "${M21_RELATIVE_DIR}/summary.json"

echo
echo "===== M23 reverse SDE vs M23 discrete ancestral ====="
cat "${SDE_RELATIVE_DIR}/summary.json"

echo
echo "============================================================"
echo "M23 DISCRETE ANCESTRAL ABLATION COMPLETE"
echo "Generation: $ANCESTRAL_GEN_DIR"
echo "Benchmark:  $ANCESTRAL_BENCH_DIR"
echo "M21 comp:   $M21_RELATIVE_DIR"
echo "SDE comp:   $SDE_RELATIVE_DIR"
echo "============================================================"
