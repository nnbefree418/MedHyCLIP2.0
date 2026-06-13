#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT/external_baselines/adaclip_bmad"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x "/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python" ]]; then
    PYTHON="/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/envs/MVFA/bin/python"
  else
    PYTHON="python3"
  fi
fi

echo "This script is intentionally gated."
echo "Run and pass the BrainMRI dry-run first:"
echo "  bash external_baselines/adaclip_bmad/run_dryrun.sh"
echo
echo "After user confirmation, remove this guard by setting RUN_FULL_LOSO=1."

if [[ "${RUN_FULL_LOSO:-0}" != "1" ]]; then
  exit 2
fi

"$PYTHON" "$WORK_DIR/tools/build_prompt_mapping.py" --overwrite
"$PYTHON" "$WORK_DIR/tools/prepare_bmad_for_adaclip.py" --target all --overwrite
"$PYTHON" "$WORK_DIR/tools/validate_bmad_indices.py" \
  --target all \
  --index-dir "$WORK_DIR/indices" \
  --report "$WORK_DIR/results/index_validation.md"
"$PYTHON" "$WORK_DIR/tools/audit_adaclip_protocol.py" \
  --adaclip-root "$ROOT/external_baselines/AdaCLIP" \
  --output-json "$WORK_DIR/results/adaclip_protocol_audit.json" \
  --output-md "$WORK_DIR/results/adaclip_protocol_audit.md"

for target in his chestxray oct17 brainmri liverct resc; do
  "$PYTHON" "$WORK_DIR/adapters/adaclip_runner.py" \
    --adaclip-root "$ROOT/external_baselines/AdaCLIP" \
    --target "$target" \
    --source-train-index "$WORK_DIR/indices/$target/source_train.jsonl" \
    --source-val-index "$WORK_DIR/indices/$target/source_val.jsonl" \
    --target-test-index "$WORK_DIR/indices/$target/target_test.jsonl" \
    --prompt-mapping "$WORK_DIR/configs/prompt_mapping.yaml" \
    --output-dir "$WORK_DIR/results/predictions/$target" \
    --image-size 240 \
    --backbone ViT-L-14-336 \
    2>&1 | tee "$WORK_DIR/results/logs/$target.log"
done
