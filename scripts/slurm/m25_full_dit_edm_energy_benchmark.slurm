#!/bin/bash
#SBATCH --job-name=m25_edm_energy_bench
#SBATCH --time=00:30:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=4
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
RUN_ROOT=/scratch/coop/theret/GraphAttention_runs

RUN_NAME=m25_full_dit_edm_1000e_h128_l10_h4
GEN_NAME=edm_karras_heun_steps50_sigmin0p004_sigmax160_rho7_seed5678

RUN_DIR=${RUN_ROOT}/${RUN_NAME}
GEN_DIR=${RUN_DIR}/generations/${GEN_NAME}

# Reuse the canonical M25 benchmark directory and rebuild it with the current
# benchmark definition, including the velocity-based E_2D(k) population spectrum.
BENCH_ROOT=${RUN_ROOT}/benchmark_m25_full_dit_edm
BENCH_DIR=${BENCH_ROOT}/${GEN_NAME}

cd "$REPO"

echo "============================================================"
echo "M25 - rebenchmark existing Full DiT + EDM generation"
echo "============================================================"
echo "Training run:     $RUN_DIR"
echo "Generation:       $GEN_DIR"
echo "Benchmark output: $BENCH_DIR"
echo
echo "Energy spectrum:"
echo "  velocity based: u=rhou/rho, v=rhov/rho, w=rhow/rho"
echo "  one E_2D(k) spectrum per snapshot"
echo "  test + generated on the same plot"
echo "  central curves: mean and median"
echo "  population band: 10-90%"
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
echo "===== Checking existing M25 generation ====="
for artifact in     generated_test.pt     summary.json     resolved_config.yaml     dataset_split_manifest.json
do
    if [ ! -f "${GEN_DIR}/${artifact}" ]; then
        echo "ERROR: missing generation artifact: ${GEN_DIR}/${artifact}"
        exit 1
    fi
done

echo
echo "===== Rebuilding benchmark ====="
echo "Existing benchmark directory, if present, will be replaced:"
echo "  $BENCH_DIR"

singularity exec     -B /home     -B /scratch     "$IMAGE"     "$PYTHON" -u -m scripts.benchmark_generation         run_dir="$GEN_DIR"         output_root="$BENCH_ROOT"         overwrite=true         energy_spectrum.enabled=true         energy_spectrum.lower_quantile=0.10         energy_spectrum.upper_quantile=0.90         plots.energy_spectrum=true

echo
echo "===== Checking standard benchmark outputs ====="
for artifact in     summary.json     benchmark_config.yaml     channel_metrics.csv     sample_statistics.csv     correlation_matrix.csv     spectra.csv     spectral_bands.csv     nearest_reference.csv     physical_metrics.csv
do
    test -s "${BENCH_DIR}/${artifact}"
done

echo
echo "===== Checking energy-spectrum outputs ====="
for artifact in     energy_spectrum_per_sample.csv     energy_spectrum_summary.csv     energy_spectral_bands.csv
do
    test -s "${BENCH_DIR}/${artifact}"
done

test -s "${BENCH_DIR}/plots/energy_spectrum/energy_spectrum_mean.png"
test -s "${BENCH_DIR}/plots/energy_spectrum/energy_spectrum_median.png"

echo
echo "===== Energy-spectrum summary ====="
"$PYTHON" - "$BENCH_DIR/summary.json" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text())
energy = summary["energy_spectrum"]
print(json.dumps(energy, indent=2))
PY

echo
echo "============================================================"
echo "M25 ENERGY-SPECTRUM BENCHMARK COMPLETE"
echo "============================================================"
echo "Benchmark:"
echo "  $BENCH_DIR"
echo
echo "Primary plots:"
echo "  $BENCH_DIR/plots/energy_spectrum/energy_spectrum_mean.png"
echo "  $BENCH_DIR/plots/energy_spectrum/energy_spectrum_median.png"
echo
echo "Per-snapshot spectra:"
echo "  $BENCH_DIR/energy_spectrum_per_sample.csv"
echo "============================================================"
