#!/usr/bin/env bash
# Common M47.10 training body. Run through an arm-specific .slurm wrapper.
# Retains strict preflight verification and existing checkpoint/resume behavior.
set -euo pipefail
VARIANT=${1:?Usage: m47_10_train_common.sh a1|a2|a3}
case "$VARIANT" in
    a1|a2|a3) ;;
    *) echo "Expected training arm a1, a2 or a3" >&2; exit 2 ;;
esac
REPO=${M47_10_REPO:-/scratch/coop/theret/GraphAttention}
IMAGE=${M47_10_IMAGE:-/softs/local_arm/singularity/images/pyg25.03.sif}
PYTHON=${M47_10_PYTHON:-/scratch/coop/theret/avbpML_env/bin/python3}
DATA=${M47_10_ARCHIVE_ROOT:-/archivenfs/phlusim/HIT_DNS_COMP}
REFERENCE_ROOT=${M47_10_REFERENCE_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_physical_conditions}
REFS=${M47_10_REFS:-$REFERENCE_ROOT/case_references}
REF_AUDIT=${M47_10_REF_AUDIT:-$REFERENCE_ROOT/reference_audit.json}
M46_ROOT=${M47_10_M46_ROOT:-/scratch/coop/theret/GraphAttention_runs/m46_hit_similarity}
QUALIFIED=${M47_10_QUALIFIED:-$M46_ROOT/qualified_manifest.json}
INVENTORY=${M47_10_INVENTORY:-$M46_ROOT/inventory.json}
POLICY=${M47_10_POLICY:-$REPO/configs/m46_stationary_selection_reviewed_legacy.json}
ROOT=${M47_10_ROOT:-/scratch/coop/theret/GraphAttention_runs/m47_10_relative_periodic}
OUT="$ROOT/$VARIANT/train_s42"

for path in "$REPO" "$IMAGE" "$PYTHON" "$DATA" "$REFS" "$REF_AUDIT" \
    "$QUALIFIED" "$INVENTORY" "$POLICY"; do
    test -e "$path" || { echo "Missing required path: $path" >&2; exit 2; }
done
cd "$REPO"
test -z "$(git status --porcelain)" || { echo "Dirty repository" >&2; exit 2; }
mkdir -p "$ROOT/$VARIANT"
exec 9>"$ROOT/$VARIANT/train.lock"
flock -n 9 || { echo "A training job already owns $VARIANT" >&2; exit 2; }
ARGS=()
if [ -e "$OUT" ]; then
    test -s "$OUT/last.pt" || {
        echo "An existing $OUT lacks a completed last.pt; inspect it before retrying" >&2
        exit 2
    }
    ARGS+=(--resume)
fi
BIND=(-B /home:/home -B /scratch:/scratch)
for mount in /archivenfs /archive2; do
    if [ -d "$mount" ]; then BIND+=(-B "$mount:$mount"); fi
done
# A1/A2 require real-mesh qualification, independently of A3 diagnostics.
# A3 additionally requires a full-size Triton forward/backward/AdamW check;
# numerical parity experiments are not a mandatory launch gate.
VERIFY_SCOPE=mesh
if [ "$VARIANT" = a3 ]; then VERIFY_SCOPE=full; fi
singularity exec "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.m47_10_preflight \
    --stage verify --verify-scope "$VERIFY_SCOPE" --repo "$REPO" --archive-root "$DATA" --reference-dir "$REFS" \
    --reference-audit "$REF_AUDIT" --qualified-manifest "$QUALIFIED" \
    --source-inventory "$INVENTORY" --selection-policy "$POLICY" \
    --output-root "$ROOT"
printf '[M47.10] arm=%s epoch_target=100 checkpoint_every=10 source=iid mach_re=on\n' "$VARIANT"
printf '[M47.10] job=%s revision=%s run=%s resume=%s\n' "$SLURM_JOB_ID" "$(git rev-parse HEAD)" "$OUT" "${ARGS[*]:-no}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -m pytest -q \
    tests/unit/test_m47_10_variants.py \
    tests/unit/test_m47_10_periodic_forward.py \
    tests/unit/test_m49_relative_only.py
singularity exec --nv "${BIND[@]}" "$IMAGE" "$PYTHON" -u -m scripts.train_m43_b_multires \
    --mode train --root "$DATA" --reference-dir "$REFS" \
    --qualified-manifest "$QUALIFIED" --source-inventory "$INVENTORY" \
    --selection-policy "$POLICY" --reference-audit "$REF_AUDIT" \
    --conditioning-mode mach_re --campaign-variant "$VARIANT" \
    --checkpoint-every 10 --run-dir "$OUT" --seed 42 \
    --temporal-embargo 2 --accumulation 6 --max-epochs 100 \
    "${ARGS[@]}"
test -s "$OUT/epoch_0100.pt"
test -s "$OUT/summary.json"
echo "[M47.10] completed arm=$VARIANT"
