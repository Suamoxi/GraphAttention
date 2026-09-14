#!/bin/bash
#SBATCH --job-name=m20_improved_ddpm_endpoint
#SBATCH --time=08:00:00
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
RUN_NAME=m17_improved_ddpm_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt

BATCH_SIZE=128
NUM_WORKERS=4
SAMPLING_SEED=5678
SAMPLING_ETA=1.0

RESTART_START=999
RESTART_STEPS=999
RESTART_NAME=improved_ddpm_start999_steps999_eta1_seed5678
RESTART_GEN_DIR=${RUN_DIR}/generations/${RESTART_NAME}

CANONICAL_START=1000
CANONICAL_STEPS=1000
CANONICAL_NAME=improved_ddpm_start1000_steps1000_eta1_seed5678
CANONICAL_GEN_DIR=${RUN_DIR}/generations/${CANONICAL_NAME}

BENCH_ROOT=${RUN_ROOT}/benchmark_m20_improved_ddpm
RESTART_BENCH_DIR=${BENCH_ROOT}/${RESTART_NAME}
CANONICAL_BENCH_DIR=${BENCH_ROOT}/${CANONICAL_NAME}

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m20_improved_ddpm_start1000_vs_start999
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M20 - Improved-DDPM endpoint analysis"
printf '%s\n' "============================================================"
printf 'Source run:       %s\n' "$RUN_DIR"
printf 'Checkpoint:       %s\n' "$CHECKPOINT"
printf 'Sampling seed:    %s\n' "$SAMPLING_SEED"
printf 'Restart chain:    %s -> 0\n' "$RESTART_START"
printf 'Canonical chain:  %s -> 0\n' "$CANONICAL_START"
printf '%s\n' "============================================================"

echo
echo "===== Git ====="
git rev-parse HEAD
git status --short

if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: tracked repository files are modified."
    exit 1
fi

if [ ! -f "${RUN_DIR}/${CHECKPOINT}" ]; then
    echo "ERROR: checkpoint does not exist: ${RUN_DIR}/${CHECKPOINT}"
    exit 1
fi

for path in \
    "$RESTART_GEN_DIR" \
    "$CANONICAL_GEN_DIR" \
    "$RESTART_BENCH_DIR" \
    "$CANONICAL_BENCH_DIR" \
    "$RELATIVE_DIR"
do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists: $path"
        exit 1
    fi
done

echo
echo "===== GPU ====="
nvidia-smi

# Run the t=999 restart first.  If the canonical endpoint later fails, the
# diagnostically important restart population and benchmark are still preserved.
echo
echo "===== Improved-DDPM generation: Gaussian restart at t=999 ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_slice_improved_diffusion \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size="$BATCH_SIZE" \
        num_workers="$NUM_WORKERS" \
        sampling.start_timestep="$RESTART_START" \
        sampling.steps="$RESTART_STEPS" \
        sampling.eta="$SAMPLING_ETA" \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$RESTART_NAME"

echo
echo "===== Benchmark: Improved-DDPM start t=999 ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$RESTART_GEN_DIR" \
        output_root="$BENCH_ROOT"

test -f "${RESTART_GEN_DIR}/generated_test.pt"
test -f "${RESTART_GEN_DIR}/summary.json"
test -f "${RESTART_BENCH_DIR}/summary.json"
test -f "${RESTART_BENCH_DIR}/channel_metrics.csv"
test -f "${RESTART_BENCH_DIR}/spectral_bands.csv"
test -f "${RESTART_BENCH_DIR}/physical_metrics.csv"

echo
echo "===== Improved-DDPM generation: canonical start t=1000 ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_slice_improved_diffusion \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size="$BATCH_SIZE" \
        num_workers="$NUM_WORKERS" \
        sampling.start_timestep="$CANONICAL_START" \
        sampling.steps="$CANONICAL_STEPS" \
        sampling.eta="$SAMPLING_ETA" \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$CANONICAL_NAME"

echo
echo "===== Benchmark: Improved-DDPM start t=1000 ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$CANONICAL_GEN_DIR" \
        output_root="$BENCH_ROOT"

test -f "${CANONICAL_GEN_DIR}/generated_test.pt"
test -f "${CANONICAL_GEN_DIR}/summary.json"
test -f "${CANONICAL_BENCH_DIR}/summary.json"
test -f "${CANONICAL_BENCH_DIR}/channel_metrics.csv"
test -f "${CANONICAL_BENCH_DIR}/spectral_bands.csv"
test -f "${CANONICAL_BENCH_DIR}/physical_metrics.csv"

echo
echo "===== Relative comparison: canonical t=1000 vs restart t=999 ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.compare_generation_benchmarks \
        baseline_dir="$CANONICAL_BENCH_DIR" \
        candidate_dir="$RESTART_BENCH_DIR" \
        output_root="$RELATIVE_ROOT" \
        comparison_name="$RELATIVE_NAME"

test -f "${RELATIVE_DIR}/summary.json"
test -f "${RELATIVE_DIR}/relative_comparison.csv"

echo
echo "===== Canonical benchmark summary ====="
cat "${CANONICAL_BENCH_DIR}/summary.json"

echo
echo "===== Restart benchmark summary ====="
cat "${RESTART_BENCH_DIR}/summary.json"

echo
echo "===== Relative comparison summary ====="
cat "${RELATIVE_DIR}/summary.json"

echo
echo "============================================================"
echo "M20 COMPLETE"
echo "Canonical generation: ${CANONICAL_GEN_DIR}"
echo "Restart generation:   ${RESTART_GEN_DIR}"
echo "Canonical benchmark:  ${CANONICAL_BENCH_DIR}"
echo "Restart benchmark:    ${RESTART_BENCH_DIR}"
echo "Relative comparison:  ${RELATIVE_DIR}"
echo "============================================================"
