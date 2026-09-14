#!/bin/bash
#SBATCH --job-name=m19_m13_start999
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
RUN_NAME=m13_m12_diffusion_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}

CHECKPOINT=best.pt
START_TIMESTEP=999
SAMPLING_STEPS=999
SAMPLING_ETA=1.0
SAMPLING_SEED=5678
BATCH_SIZE=128
NUM_WORKERS=4

GEN_NAME=ddpm_ancestral_gaussian_restart_start999_steps999_eta1_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

BENCH_ROOT=${RUN_ROOT}/benchmark_m19_m13_start999
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

# Existing full-terminal ancestral benchmark, if present.  The relative
# comparison is run only when this directory exists.
BASELINE_BENCH=${RUN_ROOT}/benchmark_m13_l10_1000e_ancestral/best_ddpm_ancestral_steps1000_eta1_seed5678
RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m13_best_start1000_vs_start999
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M19 - M13 Gaussian restart generation + benchmark"
printf '%s\n' "============================================================"
printf 'Source run:      %s\n' "$RUN_DIR"
printf 'Checkpoint:      %s\n' "$CHECKPOINT"
printf 'Start timestep:  %s\n' "$START_TIMESTEP"
printf 'Sampling steps:  %s\n' "$SAMPLING_STEPS"
printf 'Eta:             %s\n' "$SAMPLING_ETA"
printf 'Sampling seed:   %s\n' "$SAMPLING_SEED"
printf 'Generation dir:  %s\n' "$GEN_DIR"
printf 'Benchmark dir:   %s\n' "$BENCH_DIR"
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

if [ -e "$GEN_DIR" ]; then
    echo "ERROR: generation directory already exists: $GEN_DIR"
    exit 1
fi

if [ -e "$BENCH_DIR" ]; then
    echo "ERROR: benchmark directory already exists: $BENCH_DIR"
    exit 1
fi

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Generation: start t=999 ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_slice_diffusion \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size="$BATCH_SIZE" \
        num_workers="$NUM_WORKERS" \
        sampling.start_timestep="$START_TIMESTEP" \
        sampling.steps="$SAMPLING_STEPS" \
        sampling.eta="$SAMPLING_ETA" \
        sampling.seed="$SAMPLING_SEED" \
        output_name="$GEN_NAME"

echo
echo "===== Checking generation outputs ====="
test -f "${GEN_DIR}/generated_test.pt"
test -f "${GEN_DIR}/summary.json"
test -f "${GEN_DIR}/resolved_config.yaml"
test -f "${GEN_DIR}/generation_config.yaml"
test -f "${GEN_DIR}/dataset_split_manifest.json"

echo
echo "===== Distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$GEN_DIR" \
        output_root="$BENCH_ROOT"

echo
echo "===== Checking benchmark outputs ====="
test -f "${BENCH_DIR}/summary.json"
test -f "${BENCH_DIR}/channel_metrics.csv"
test -f "${BENCH_DIR}/sample_statistics.csv"
test -f "${BENCH_DIR}/correlation_matrix.csv"
test -f "${BENCH_DIR}/spectra.csv"
test -f "${BENCH_DIR}/spectral_bands.csv"
test -f "${BENCH_DIR}/nearest_reference.csv"
test -f "${BENCH_DIR}/physical_metrics.csv"

echo
echo "===== Benchmark summary ====="
cat "${BENCH_DIR}/summary.json"

if [ -d "$BASELINE_BENCH" ]; then
    echo
    echo "===== Relative comparison: start 1000 vs start 999 ====="
    if [ -e "$RELATIVE_DIR" ]; then
        echo "ERROR: relative comparison directory already exists: $RELATIVE_DIR"
        exit 1
    fi
    singularity exec \
        -B /home \
        -B /scratch \
        "$IMAGE" \
        "$PYTHON" -u -m scripts.compare_generation_benchmarks \
            baseline_dir="$BASELINE_BENCH" \
            candidate_dir="$BENCH_DIR" \
            output_root="$RELATIVE_ROOT" \
            comparison_name="$RELATIVE_NAME"

    test -f "${RELATIVE_DIR}/relative_comparison.csv"
    test -f "${RELATIVE_DIR}/summary.json"
    echo
    echo "Relative comparison: ${RELATIVE_DIR}"
else
    echo
    echo "Baseline benchmark not found; skipping relative comparison:"
    echo "$BASELINE_BENCH"
fi

echo
echo "============================================================"
echo "M19 COMPLETE"
echo "Generation: ${GEN_DIR}"
echo "Benchmark:  ${BENCH_DIR}"
if [ -d "$RELATIVE_DIR" ]; then
    echo "Relative:   ${RELATIVE_DIR}"
fi
echo "============================================================"
