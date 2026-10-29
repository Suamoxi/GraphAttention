#!/bin/bash
#SBATCH --job-name=m44_sr04_compare
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# Read-only SR-0.3 vs SR-0.4 JSON comparison, inside Singularity on grace.
# GPU reservation follows the existing grace launchers; Python is CPU-only.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
ROOT=/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics
BASE_RUN=${M44_SR04_BASE_RUN:-/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_484619}
DIVERSE_RUN=${M44_SR04_DIVERSE_RUN:-}
BASE_REPORT=${M44_SR04_BASE_REPORT:-}
DIVERSE_REPORT=${M44_SR04_DIVERSE_REPORT:-}
OUT=${M44_SR04_COMPARE_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr04_paired_comparison/job_${SLURM_JOB_ID}}

if [ -z "$DIVERSE_RUN" ]; then
  echo "Required: M44_SR04_DIVERSE_RUN=/scratch/.../m44_sr04/job_<ID>" >&2
  exit 2
fi
BASE_RUN=${BASE_RUN%/}
DIVERSE_RUN=${DIVERSE_RUN%/}

# The first-level source_run_dir is indented by exactly two spaces in
# diagnose_m44_sr0's json.dump(indent=2). Do NOT grep the whole JSON:
# SR-0.4 also contains the SR-0.3 path in frozen_scale_provenance.
report_source_run() {
  local report="$1"
  local source_line
  source_line=$(grep -m1 '^  "source_run_dir": "' "$report") || return 1
  printf '%s\n' "$source_line" | cut -d '"' -f4
}

find_for_run() {
  local source="$1"
  local matches=()
  local file
  for file in "$ROOT"/job_*/diagnostic_summary.json; do
    [ -s "$file" ] || continue
    if [ "$(report_source_run "$file")" = "$source" ]; then
      matches+=("$file")
    fi
  done
  if [ "${#matches[@]}" -ne 1 ]; then
    echo "Need exactly one checkpoint diagnostic for $source; found ${#matches[@]}." >&2
    echo "Provide M44_SR04_BASE_REPORT or M44_SR04_DIVERSE_REPORT explicitly." >&2
    printf '  %s\n' "${matches[@]}" >&2
    return 2
  fi
  printf '%s\n' "${matches[0]}"
}

if [ -z "$BASE_REPORT" ]; then BASE_REPORT=$(find_for_run "$BASE_RUN"); fi
if [ -z "$DIVERSE_REPORT" ]; then DIVERSE_REPORT=$(find_for_run "$DIVERSE_RUN"); fi

for file in "$BASE_REPORT" "$DIVERSE_REPORT"; do
  if [ "${file#/}" = "$file" ] || [ ! -s "$file" ]; then
    echo "Missing or nonabsolute diagnostic report: $file" >&2
    exit 2
  fi
done
if [ "$BASE_REPORT" = "$DIVERSE_REPORT" ]; then
  echo "Baseline and diversified diagnostics must be different files" >&2
  exit 2
fi
if [ "$(report_source_run "$BASE_REPORT")" != "$BASE_RUN" ]; then
  echo "Baseline diagnostic does not belong to $BASE_RUN" >&2
  exit 2
fi
if [ "$(report_source_run "$DIVERSE_REPORT")" != "$DIVERSE_RUN" ]; then
  echo "Diversified diagnostic does not belong to $DIVERSE_RUN" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite comparison output: $OUT" >&2
  exit 2
fi

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty Git checkout" >&2
  git status --short >&2
  exit 2
fi
echo "=== M44 SR-0.4 paired comparison on grace ==="
echo "Git: $(git rev-parse HEAD); job: ${SLURM_JOB_ID}; host: $(hostname)"
echo "Baseline report: $BASE_REPORT"
echo "Diverse report: $DIVERSE_REPORT"
echo "Output: $OUT"
sha256sum "$BASE_REPORT" "$DIVERSE_REPORT"

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m44_sr04_comparison.py

singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.compare_m44_sr04_diagnostics \
  --baseline "$BASE_REPORT" --diverse "$DIVERSE_REPORT" --output-dir "$OUT"

test -s "$OUT/comparison_summary.json"
test -s "$OUT/comparison_summary.md"
echo "SR-0.4 comparison: $OUT/comparison_summary.json"
