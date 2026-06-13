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
TARGET="${TARGET:-brainmri}"

mkdir -p "$WORK_DIR/results/dryrun"

"$PYTHON" "$WORK_DIR/tools/build_prompt_mapping.py" --overwrite
"$PYTHON" "$WORK_DIR/tools/prepare_bmad_for_adaclip.py" \
  --target "$TARGET" \
  --dryrun \
  --overwrite \
  --output-dir "$WORK_DIR/indices_dryrun"
"$PYTHON" "$WORK_DIR/tools/validate_bmad_indices.py" \
  --target "$TARGET" \
  --index-dir "$WORK_DIR/indices_dryrun" \
  --report "$WORK_DIR/results/dryrun/index_validation.md"

set +e
"$PYTHON" "$WORK_DIR/adapters/adaclip_runner.py" \
  --adaclip-root "$ROOT/external_baselines/AdaCLIP" \
  --target "$TARGET" \
  --source-train-index "$WORK_DIR/indices_dryrun/$TARGET/source_train.jsonl" \
  --source-val-index "$WORK_DIR/indices_dryrun/$TARGET/source_val.jsonl" \
  --target-test-index "$WORK_DIR/indices_dryrun/$TARGET/target_test.jsonl" \
  --prompt-mapping "$WORK_DIR/configs/prompt_mapping.yaml" \
  --output-dir "$WORK_DIR/results/dryrun" \
  --image-size 240 \
  --backbone ViT-L-14-336 \
  --predictions-csv "$WORK_DIR/results/dryrun/dryrun_predictions.csv" \
  --protocol-manifest "$WORK_DIR/results/dryrun/protocol_manifest.json" \
  --dryrun
RUNNER_EXIT=$?
set -e

METRICS_EXIT=0
if [[ "$RUNNER_EXIT" -eq 0 ]]; then
  set +e
  "$PYTHON" "$WORK_DIR/eval_bmad_metrics.py" \
    --dataset "$TARGET" \
    --predictions "$WORK_DIR/results/dryrun/dryrun_predictions.csv" \
    --output "$WORK_DIR/results/dryrun/dryrun_metrics.json"
  METRICS_EXIT=$?
  set -e
fi

set +e
"$PYTHON" "$WORK_DIR/tools/audit_adaclip_protocol.py" \
  --adaclip-root "$ROOT/external_baselines/AdaCLIP" \
  --protocol-manifest "$WORK_DIR/results/dryrun/protocol_manifest.json" \
  --output-json "$WORK_DIR/results/dryrun/adaclip_protocol_audit.json" \
  --output-md "$WORK_DIR/results/dryrun/adaclip_protocol_audit.md"
AUDIT_EXIT=$?
set -e

cat > "$WORK_DIR/results/dryrun/dryrun_report.md" <<REPORT
# AdaCLIP BMAD BrainMRI Dry-Run Report

- Target: $TARGET
- Index generation: completed
- Source/target leakage validation: completed
- Dry-run predictions: $WORK_DIR/results/dryrun/dryrun_predictions.csv
- Dry-run metrics: $WORK_DIR/results/dryrun/dryrun_metrics.json
- AdaCLIP protocol audit exit code: $AUDIT_EXIT
- AdaCLIP runner exit code: $RUNNER_EXIT
- Metrics exit code: $METRICS_EXIT
- Main table result generated: no

## Status

The BMAD data indexing and leakage checks completed. This dry-run is considered
passed only when runner, metrics, and protocol audit all exit with code 0.
The audit uses a wrapper protocol manifest to verify that official default data,
target validation, target checkpoint selection, target threshold selection, and
official metrics are bypassed.

If the runner is blocked because \`external_baselines/AdaCLIP\` is missing,
restore network access or provide the official AdaCLIP repository at that path,
then rerun:

\`\`\`bash
bash external_baselines/adaclip_bmad/run_dryrun.sh
\`\`\`
REPORT

if [[ "$AUDIT_EXIT" -ne 0 || "$RUNNER_EXIT" -ne 0 || "$METRICS_EXIT" -ne 0 ]]; then
  echo "Dry-run blocked. See $WORK_DIR/results/dryrun/dryrun_report.md"
  exit 2
fi

echo "Dry-run passed. See $WORK_DIR/results/dryrun/dryrun_report.md"
