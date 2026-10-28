#!/bin/bash
#SBATCH --job-name=m44_sr03c_compare
#SBATCH --partition=grace
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/coop/theret/GraphAttention/%x_%j.out
#SBATCH --error=/scratch/coop/theret/GraphAttention/%x_%j.err

# M44 SR-0.3c: read-only paired comparison of existing JSON diagnostics.
# All Python (including pytest) runs inside Singularity on a grace Slurm node.
# A GPU is reserved for consistency with the validated grace launch pattern;
# this particular JSON-only comparison does not use GPU computation.
set -euo pipefail

REPO=/scratch/coop/theret/GraphAttention
IMAGE=/softs/local_arm/singularity/images/pyg25.03.sif
PYTHON=/scratch/coop/theret/avbpML_env/bin/python3
REPORT_ROOT=/scratch/coop/theret/GraphAttention_runs/m44_sr02_diagnostics
OLD_REPORT=${M44_SR03C_OLD_REPORT:-$REPORT_ROOT/job_484628/diagnostic_summary.json}
CONSTRAINED_RUN=${M44_SR03C_CONSTRAINED_RUN:-/scratch/coop/theret/GraphAttention_runs/m44_sr01/job_484619}
CONSTRAINED_RUN=${CONSTRAINED_RUN%/}
NEW_REPORT=${M44_SR03C_NEW_REPORT:-}
OUT=${M44_SR03C_OUTPUT:-/scratch/coop/theret/GraphAttention_runs/m44_sr03c_paired_comparison/job_${SLURM_JOB_ID}}

cd "$REPO"
if [ -n "$(git status --porcelain)" ]; then
  echo "Refusing dirty GraphAttention checkout" >&2
  git status --short >&2
  exit 2
fi

# Discover only the exact source_run_dir recorded in the report, not a
# substring of the file name or an arbitrary first grep result.
if [ -z "$NEW_REPORT" ]; then
  candidates=()
  for report in "$REPORT_ROOT"/job_*/diagnostic_summary.json; do
    [ -s "$report" ] || continue
    if grep -Fq "\"source_run_dir\": \"$CONSTRAINED_RUN\"" "$report"; then
      candidates+=("$report")
    fi
  done
  if [ "${#candidates[@]}" -ne 1 ]; then
    echo "Expected exactly one constrained report for: $CONSTRAINED_RUN" >&2
    echo "Found ${#candidates[@]}; set M44_SR03C_NEW_REPORT explicitly:" >&2
    printf '  %s\n' "${candidates[@]}" >&2
    exit 2
  fi
  NEW_REPORT=${candidates[0]}
fi

for report in "$OLD_REPORT" "$NEW_REPORT"; do
  if [ "${report#/}" = "$report" ] || [ ! -s "$report" ]; then
    echo "Report must be an existing, nonempty absolute file path: $report" >&2
    exit 2
  fi
done
if [ "$OLD_REPORT" = "$NEW_REPORT" ]; then
  echo "Constrained and unconstrained report paths are identical" >&2
  exit 2
fi
if ! grep -Fq "\"source_run_dir\": \"$CONSTRAINED_RUN\"" "$NEW_REPORT"; then
  echo "Constrained report does not belong to $CONSTRAINED_RUN" >&2
  exit 2
fi
if [ -e "$OUT" ]; then
  echo "Refusing to overwrite an existing comparison output: $OUT" >&2
  exit 2
fi

echo "=== M44 SR-0.3c paired common-604-node comparison ==="
echo "Git revision: $(git rev-parse HEAD)"
echo "Slurm job: ${SLURM_JOB_ID}; host: $(hostname)"
echo "Unconstrained diagnostic: $OLD_REPORT"
echo "Constrained diagnostic:   $NEW_REPORT"
echo "Output directory:         $OUT"
echo "WARNING: point-sampled DNS proxy, not filtered LES or SGS validation"
sha256sum "$OLD_REPORT" "$NEW_REPORT"

echo "=== Run comparison correctness gates inside Singularity ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -m pytest -q tests/unit/test_m44_sr03_common_support_comparison.py

echo "=== Compare saved JSON diagnostics inside Singularity (read-only) ==="
singularity exec --nv -B /home -B /scratch -B /archivenfs:/archivenfs "$IMAGE" \
  "$PYTHON" -u -m scripts.compare_m44_sr03_diagnostics \
  --unconstrained "$OLD_REPORT" \
  --constrained "$NEW_REPORT" \
  --output-dir "$OUT"

test -s "$OUT/comparison_summary.json"
test -s "$OUT/comparison_summary.md"
echo "M44 SR-0.3c JSON: $OUT/comparison_summary.json"
echo "M44 SR-0.3c Markdown: $OUT/comparison_summary.md"
