#!/bin/bash
#SBATCH --job-name=m21_linear_ddpm_gen
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
RUN_NAME=m21_m12_diffusion_linear_1000e_h128_l10_h4
RUN_DIR=${RUN_ROOT}/${RUN_NAME}
CHECKPOINT=best.pt

GEN_NAME=linear_ddpm_ancestral_start1000_steps1000_eta1_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}
BENCH_ROOT=${RUN_ROOT}/benchmark_m21_linear_ddpm
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

# Stable cosine baseline from M19, if present.
BASELINE_BENCH=${RUN_ROOT}/benchmark_m19_m13_start999/ddpm_ancestral_gaussian_restart_start999_steps999_eta1_seed5678
RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
RELATIVE_NAME=m21_cosine_start999_vs_linear_start1000
RELATIVE_DIR=${RELATIVE_ROOT}/${RELATIVE_NAME}

cd "$REPO"

printf '%s\n' "============================================================"
printf '%s\n' "M21 - Linear-beta DDPM full-chain generation + benchmark"
printf '%s\n' "============================================================"
printf 'Run:             %s\n' "$RUN_DIR"
printf 'Checkpoint:      %s\n' "$CHECKPOINT"
printf '%s\n' "Schedule:        linear beta"
printf '%s\n' "beta_start:      1e-4"
printf '%s\n' "beta_end:        2e-2"
printf '%s\n' "Start timestep:  1000"
printf '%s\n' "Steps:           1000"
printf '%s\n' "Eta:             1.0"
printf '%s\n' "Seed:            5678"
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

for path in "$GEN_DIR" "$BENCH_DIR"; do
    if [ -e "$path" ]; then
        echo "ERROR: output already exists: $path"
        exit 1
    fi
done

echo
echo "===== GPU ====="
nvidia-smi

echo
echo "===== Full ancestral generation ====="
singularity exec --nv \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.generate_slice_linear_diffusion \
        run_dir="$RUN_DIR" \
        checkpoint="$CHECKPOINT" \
        device=cuda \
        batch_size=128 \
        num_workers=4 \
        sampling.start_timestep=1000 \
        sampling.steps=1000 \
        sampling.eta=1.0 \
        sampling.seed=5678 \
        output_name="$GEN_NAME"

test -f "${GEN_DIR}/generated_test.pt"
test -f "${GEN_DIR}/summary.json"
test -f "${GEN_DIR}/resolved_config.yaml"
test -f "${GEN_DIR}/generation_config.yaml"

echo
echo "===== Distribution benchmark ====="
singularity exec \
    -B /home \
    -B /scratch \
    "$IMAGE" \
    "$PYTHON" -u -m scripts.benchmark_generation \
        run_dir="$GEN_DIR" \
        output_root="$BENCH_ROOT"

test -f "${BENCH_DIR}/summary.json"
test -f "${BENCH_DIR}/channel_metrics.csv"
test -f "${BENCH_DIR}/spectral_bands.csv"
test -f "${BENCH_DIR}/physical_metrics.csv"

echo
echo "===== Benchmark summary ====="
cat "${BENCH_DIR}/summary.json"

if [ -d "$BASELINE_BENCH" ]; then
    echo
    echo "===== Relative comparison: cosine start999 vs linear full chain ====="
    if [ -e "$RELATIVE_DIR" ]; then
        echo "ERROR: relative comparison output already exists: $RELATIVE_DIR"
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

    test -f "${RELATIVE_DIR}/summary.json"
    test -f "${RELATIVE_DIR}/relative_comparison.csv"
    echo
    echo "===== Relative comparison summary ====="
    cat "${RELATIVE_DIR}/summary.json"
else
    echo
    echo "Stable M19 cosine benchmark not found; skipping relative comparison:"
    echo "$BASELINE_BENCH"
fi

echo
echo "============================================================"
echo "M21 COMPLETE"
echo "Generation: ${GEN_DIR}"
echo "Benchmark:  ${BENCH_DIR}"
if [ -d "$RELATIVE_DIR" ]; then
    echo "Relative:   ${RELATIVE_DIR}"
fi
echo "============================================================"
