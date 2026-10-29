#!/bin/bash
#SBATCH --job-name=m44_sr05_compare
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Read-only SR-0.5 paired JSON diagnostics: all Python and tests run inside
# Singularity on grace. The GPU is reserved for the established grace pattern
# but the JSON-only comparison does not need GPU arithmetic.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics
BASE_RUN=${M44_SR05_BASELINE_RUN:-}
DIVERSE_RUN=${M44_SR05_DIVERSE_RUN:-}
BASE_REPORT=${M44_SR05_BASELINE_REPORT:-}
DIVERSE_REPORT=${M44_SR05_DIVERSE_REPORT:-}
OUT=${M44_SR05_COMPARE_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr05_paired/job_${SLURM_JOB_ID}}

if [ -z "$BASE_RUN" ] || [ -z "$DIVERSE_RUN" ]; then
  echo "Set both M44_SR05_BASELINE_RUN and M44_SR05_DIVERSE_RUN to the completed continuation dirs" >&2
  exit 2
fi
BASE_RUN=${BASE_RUN%/}
DIVERSE_RUN=${DIVERSE_RUN%/}
for run in "$BASE_RUN" "$DIVERSE_RUN"; do
  if [ "${run#/}" = "$run" ] || [ ! -s "$run/continuation_provenance.json" ]; then
    echo "Missing or relative continuation run directory: $run" >&2
    exit 2
  fi
done

report_source_run() {
  local line
  line=$(grep -m1 '^  "source_run_dir": "' "$1") || return 1
  printf '%s\n' "$line" | cut -d '"' -f4
}

locate_diagnostic() {
  local run="$1"
  local matches=()
  local f
  for f in "$ROOT"/job_*/diagnostic_summary.json; do
    [ -s "$f" ] || continue
    if [ "$(report_source_run "$f")" = "$run" ]; then
      matches+=("$f")
    fi
  done
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Expected exactly one diagnostic for $run, found ${#matches[@]}" >&2
    echo "Use explicit M44_SR05_BASELINE_REPORT / M44_SR05_DIVERSE_REPORT" >&2
    printf '  %s\n' "${matches[@]}" >&2
    return 2
  fi
  printf '%s\n' "${matches[0]}"
}

if [ -z "$BASE_REPORT" ]; then BASE_REPORT=$(locate_diagnostic "$BASE_RUN"); fi
if [ -z "$DIVERSE_REPORT" ]; then DIVERSE_REPORT=$(locate_diagnostic "$DIVERSE_RUN"); fi

for file in "$BASE_REPORT" "$DIVERSE_REPORT"; do
  if [ "${file#/}" = "$file" ] || [ ! -s "$file" ]; then
    echo "Missing/relative diagnostic report: $file" >&2
    exit 2
  fi
done
if [ "$BASE_REPORT" = "$DIVERSE_REPORT" ]; then
  echo "The two diagnostics must be distinct" >&2
  exit 2
fi
if [ "$(report_source_run "$BASE_REPORT")" != "$BASE_RUN" ]; then
  echo "Baseline report is not from the specified baseline continuation" >&2
  exit 2
fi
if [ "$(report_source_run "$DIVERSE_REPORT")" != "$DIVERSE_RUN" ]; then
  echo "Diversified report is not from the specified diversified continuation" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite paired comparison: $OUT" >&2
  exit 2
fi

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi
echo "=== M44 SR-0.5 paired continuation on grace ==="
echo "Git=$(git rev-parse HEAD); host=$(hostname); job=${SLURM_JOB_ID}"
echo "Baseline diagnostic: $BASE_REPORT"
echo "Diverse diagnostic: $DIVERSE_REPORT"
echo "Output: $OUT"
sha256sum "$BASE_REPORT" "$DIVERSE_REPORT"

echo "=== Comparison correctness gates inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q \
  tests/unit/test_m44_sr05_comparison.py \
  tests/unit/test_m44_sr04_comparison.py

echo "=== Compare completed 150-epoch checkpoints inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.compare_m44_sr05_diagnostics \
  --baseline "$BASE_REPORT" --diverse "$DIVERSE_REPORT" --output-dir "$OUT"

test -s "$OUT/comparison_summary.json"
test -s "$OUT/comparison_summary.md"
echo "M44 SR-0.5 result: $OUT/comparison_summary.json"
