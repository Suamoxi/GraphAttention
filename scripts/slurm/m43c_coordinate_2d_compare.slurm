#!/bin/bash
#SBATCH --job-name=m43c_pos_compare
#SBATCH --time=01:00:00
#SBATCH --partition=grace
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# CPU-only postprocessing of TWO COMPLETED M43C generative benchmarks.
# Use sbatch --dependency=afterok:ABS_JOB:REL_JOB for correct sequencing.
# Match existing CPU-only comparison jobs on the ARM64 Grace partition.
# No GPU is allocated (--gres=gpu:0), required for the local ARM environment.
set -euo pipefail

REPO=${M43C_REPO:-/scratch/coop/theret/GraphAttention_m43c_coords}
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
OUT_ROOT=${M43C_OUTPUT_ROOT:-/scratch/coop/theret/GraphAttention_runs/m43c_coordinate_2d}
EPOCHS=${M43C_EPOCHS:-100}

if ! [[ "$EPOCHS" =~ ^[0-9]+$ ]] || (( EPOCHS < 1 || EPOCHS > 1000 )); then
  echo "M43C_EPOCHS must be an integer in [1, 1000]" >&2
  exit 2
fi
test -d "$REPO" || { echo "Missing experiment repository: $REPO" >&2; exit 2; }
test -d "$OUT_ROOT" || { echo "Missing experimental output root: $OUT_ROOT" >&2; exit 2; }
test -f "$IMAGE" || { echo "Missing Singularity image: $IMAGE" >&2; exit 2; }
cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty experiment checkout" >&2
  git status --short >&2
  exit 2
fi

GEN_NAME=flow_heun_steps50_seed5678
for VARIANT in absolute relative; do
  RUN_DIR="$OUT_ROOT/m43c_${VARIANT}_s42_e${EPOCHS}"
  GEN_DIR="$RUN_DIR/generations/$GEN_NAME"
  BENCH_DIR="$OUT_ROOT/benchmark_${VARIANT}/$GEN_NAME"
  for f in best.pt history.csv resolved_config.yaml standardizers.pt dataset_split_manifest.json summary.json; do
    test -s "$RUN_DIR/$f" || { echo "Missing trained input: $RUN_DIR/$f" >&2; exit 3; }
  done
  for f in generated_test.pt summary.json generation_wall_seconds.txt; do
    test -s "$GEN_DIR/$f" || { echo "Missing generated input: $GEN_DIR/$f" >&2; exit 3; }
  done
  for f in summary.json standardized_wasserstein.csv energy_spectrum_summary.csv; do
    test -s "$BENCH_DIR/$f" || { echo "Missing completed benchmark: $BENCH_DIR/$f" >&2; exit 3; }
  done
done

RESULT_DIR="$OUT_ROOT/coordinate_comparison_e${EPOCHS}"
if [ -e "$RESULT_DIR" ]; then
  echo "Refusing to overwrite existing comparison: $RESULT_DIR" >&2
  exit 3
fi

export PYTHONPATH="$REPO/src:$REPO:${PYTHONPATH:-}"
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS=1
echo "==== M43C CPU-only absolute vs relative comparison ===="
echo "Job: ${SLURM_JOB_ID:-unknown}; host: $(hostname); source: $(git rev-parse HEAD)"
echo "Inputs: $OUT_ROOT ; epochs: $EPOCHS"
echo "Output: $RESULT_DIR"

singularity exec -B /home -B /scratch "$IMAGE" \
  "$PYTHON" -u -m scripts.compare_m43c_coordinate_2d \
  --root "$OUT_ROOT" --epochs "$EPOCHS"

test -s "$RESULT_DIR/comparison.json"
test -s "$RESULT_DIR/validation_mse_comparison.pdf"
test -s "$RESULT_DIR/energy_spectrum_comparison.pdf"
test -s "$RESULT_DIR/wasserstein_comparison.pdf"
echo "Comparison complete: $RESULT_DIR/comparison.json"
