#!/bin/bash
#SBATCH --job-name=m24_flow_benchmark
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

# Preferred M24 generation name used by the combined generation+benchmark SLURM.
GEN_NAME=flow_heun_steps50_seed5678
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

BENCH_ROOT=${RUN_ROOT}/benchmark_m24_flow

OLD_FLOW_BENCH=${RUN_ROOT}/benchmark_300e_comparison/m12_flow_l10/m12_dinat_exact2_flow_300e_h128_l10_h4
M22_BENCH=${RUN_ROOT}/benchmark_m22_edm_karras/edm_karras_heun_steps18_sigmin0p004_sigmax160_rho7_seed5678

RELATIVE_ROOT=${RUN_ROOT}/relative_benchmark_comparisons
FLOW_RELATIVE_NAME=m11_flow300_vs_m24_flow1000
FLOW_RELATIVE_DIR=${RELATIVE_ROOT}/${FLOW_RELATIVE_NAME}
EDM_RELATIVE_NAME=m22_edm_vs_m24_flow1000
EDM_RELATIVE_DIR=${RELATIVE_ROOT}/${EDM_RELATIVE_NAME}

cd "$REPO"

echo "============================================================"
echo "M24 - benchmark existing Flow Matching generation"
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
echo "===== Locate existing generation ====="

# If the expected generation path is absent, locate an existing generated_test.pt.
# This covers runs created without output_name, whose directory is typically
# flow_ode_heun_steps50_eps0_seed5678.
if [ ! -f "${GEN_DIR}/generated_test.pt" ]; then
    echo "Expected generation not found at:"
    echo "  ${GEN_DIR}"
    echo "Searching under:"
    echo "  ${RUN_DIR}/generations"

    mapfile -t GENERATION_ARTIFACTS < <(
        find "${RUN_DIR}/generations"             -mindepth 2             -maxdepth 2             -type f             -name generated_test.pt             -print 2>/dev/null | sort
    )

    if [ "${#GENERATION_ARTIFACTS[@]}" -eq 0 ]; then
        echo "ERROR: no generated_test.pt found under ${RUN_DIR}/generations"
        echo
        echo "Current generation tree:"
        find "${RUN_DIR}/generations" -maxdepth 2 -type f -print 2>/dev/null | sort || true
        exit 1
    fi

    if [ "${#GENERATION_ARTIFACTS[@]}" -gt 1 ]; then
        echo "ERROR: multiple generation artifacts found; refusing to guess:"
        printf '  %s\n' "${GENERATION_ARTIFACTS[@]}"
        exit 1
    fi

    GEN_DIR=$(dirname "${GENERATION_ARTIFACTS[0]}")
    GEN_NAME=$(basename "$GEN_DIR")
    echo "Using discovered generation:"
    echo "  ${GEN_DIR}"
fi

BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

echo
echo "Generation: ${GEN_DIR}"
echo "Benchmark:  ${BENCH_DIR}"

echo
echo "===== Checking generation artifacts ====="
for artifact in     generated_test.pt     summary.json     resolved_config.yaml     dataset_split_manifest.json
do
    if [ ! -f "${GEN_DIR}/${artifact}" ]; then
        echo "ERROR: missing generation artifact: ${GEN_DIR}/${artifact}"
        exit 1
    fi
done

if [ -f "${BENCH_DIR}/summary.json" ]; then
    echo "ERROR: benchmark already appears complete: ${BENCH_DIR}"
    exit 1
fi

if [ -d "$BENCH_DIR" ]; then
    echo "Existing incomplete benchmark directory will be replaced:"
    echo "$BENCH_DIR"
fi

echo
echo "===== Distribution benchmark ====="
singularity exec     -B /home     -B /scratch     "$IMAGE"     "$PYTHON" -u -m scripts.benchmark_generation         run_dir="$GEN_DIR"         output_root="$BENCH_ROOT"         overwrite=true

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

if [ -d "$OLD_FLOW_BENCH" ]; then
    echo
    echo "===== Relative comparison: 300e Flow vs M24 1000e Flow ====="
    singularity exec         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.compare_generation_benchmarks             baseline_dir="$OLD_FLOW_BENCH"             candidate_dir="$BENCH_DIR"             output_root="$RELATIVE_ROOT"             comparison_name="$FLOW_RELATIVE_NAME"             overwrite=true

    test -f "${FLOW_RELATIVE_DIR}/summary.json"
    test -f "${FLOW_RELATIVE_DIR}/relative_comparison.csv"
else
    echo
    echo "Old 300e Flow benchmark not found; skipping:"
    echo "$OLD_FLOW_BENCH"
fi

if [ -d "$M22_BENCH" ]; then
    echo
    echo "===== Relative comparison: M22 EDM vs M24 Flow ====="
    singularity exec         -B /home         -B /scratch         "$IMAGE"         "$PYTHON" -u -m scripts.compare_generation_benchmarks             baseline_dir="$M22_BENCH"             candidate_dir="$BENCH_DIR"             output_root="$RELATIVE_ROOT"             comparison_name="$EDM_RELATIVE_NAME"             overwrite=true

    test -f "${EDM_RELATIVE_DIR}/summary.json"
    test -f "${EDM_RELATIVE_DIR}/relative_comparison.csv"
else
    echo
    echo "M22 EDM benchmark not found; skipping:"
    echo "$M22_BENCH"
fi

echo
echo "===== M24 benchmark summary ====="
cat "${BENCH_DIR}/summary.json"

if [ -f "${FLOW_RELATIVE_DIR}/summary.json" ]; then
    echo
    echo "===== 300e Flow vs M24 summary ====="
    cat "${FLOW_RELATIVE_DIR}/summary.json"
fi

if [ -f "${EDM_RELATIVE_DIR}/summary.json" ]; then
    echo
    echo "===== M22 EDM vs M24 summary ====="
    cat "${EDM_RELATIVE_DIR}/summary.json"
fi

echo
echo "============================================================"
echo "M24 FLOW BENCHMARK COMPLETE"
echo "Benchmark:     $BENCH_DIR"
if [ -d "$FLOW_RELATIVE_DIR" ]; then
    echo "Relative Flow: $FLOW_RELATIVE_DIR"
fi
if [ -d "$EDM_RELATIVE_DIR" ]; then
    echo "Relative EDM:  $EDM_RELATIVE_DIR"
fi
echo "============================================================"
