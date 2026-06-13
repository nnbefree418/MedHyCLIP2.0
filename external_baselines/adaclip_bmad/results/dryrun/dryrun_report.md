# AdaCLIP BMAD BrainMRI Dry-Run Report

- Target: brainmri
- Index generation: completed
- Source/target leakage validation: completed
- Dry-run predictions: /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP/external_baselines/adaclip_bmad/results/dryrun/dryrun_predictions.csv
- Dry-run metrics: /mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP/external_baselines/adaclip_bmad/results/dryrun/dryrun_metrics.json
- AdaCLIP protocol audit exit code: 3
- AdaCLIP runner exit code: 1
- Metrics exit code: 0
- Main table result generated: no

## Status

The BMAD data indexing and leakage checks completed. This dry-run is considered
passed only when runner, metrics, and protocol audit all exit with code 0.
The audit uses a wrapper protocol manifest to verify that official default data,
target validation, target checkpoint selection, target threshold selection, and
official metrics are bypassed.

If the runner is blocked because `external_baselines/AdaCLIP` is missing,
restore network access or provide the official AdaCLIP repository at that path,
then rerun:

```bash
bash external_baselines/adaclip_bmad/run_dryrun.sh
```
