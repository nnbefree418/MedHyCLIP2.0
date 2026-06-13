# AdaCLIP Protocol Audit

- Status: review_required
- Main table allowed before manual resolution: False
- Reason: static audit completed, but manifest gate failed

## Findings

- `loss.py`: target_or_test_data_use -> `normal`
- `loss.py`: test_statistics_or_calibration -> `mean()`
- `train.sh`: default_external_training_data -> `mvtec`
- `train.sh`: target_or_test_data_use -> `testing_data`
- `train.sh`: threshold_selection -> `best`
- `train.sh`: checkpoint_selection -> `best`
- `test.py`: default_external_training_data -> `mvtec`
- `test.py`: target_or_test_data_use -> `testing_data`
- `README.md`: default_external_training_data -> `mvtec`
- `README.md`: target_or_test_data_use -> `testing_data`
- `README.md`: threshold_selection -> `best`
- `README.md`: checkpoint_selection -> `best`
- `test.sh`: default_external_training_data -> `mvtec`
- `test.sh`: target_or_test_data_use -> `testing_data`
- `test_single_image.sh`: target_or_test_data_use -> `class_name`
- `app.py`: default_external_training_data -> `mvtec`
- `train.py`: default_external_training_data -> `mvtec`
- `train.py`: target_or_test_data_use -> `testing_data`
- `train.py`: threshold_selection -> `best`
- `train.py`: checkpoint_selection -> `best`
- `tools/csv_tools.py`: target_or_test_data_use -> `class_name`
- `tools/training_tools.py`: default_external_training_data -> `training_data`
- `tools/training_tools.py`: target_or_test_data_use -> `testing_data`
- `tools/metrics.py`: target_or_test_data_use -> `normal`
- `tools/metrics.py`: threshold_selection -> `precision_recall_curve`
- `tools/metrics.py`: test_statistics_or_calibration -> `min()`
- `method/clip_model.py`: threshold_selection -> `optimal`
- `method/clip_model.py`: checkpoint_selection -> `checkpoint`
- `method/adaclip.py`: target_or_test_data_use -> `normal`
- `method/adaclip.py`: test_statistics_or_calibration -> `mean()`
- `method/custom_clip.py`: target_or_test_data_use -> `normal`
- `method/custom_clip.py`: checkpoint_selection -> `checkpoint`
- `method/custom_clip.py`: test_statistics_or_calibration -> `normalize`
- `method/simple_tokenizer.py`: target_or_test_data_use -> `normal`
- `method/transformer.py`: target_or_test_data_use -> `normal`
- `method/transformer.py`: threshold_selection -> `best`
- `method/transformer.py`: checkpoint_selection -> `checkpoint`
- `method/transformer.py`: test_statistics_or_calibration -> `normalize`
- `method/tokenizer.py`: target_or_test_data_use -> `normal`
- `data_preprocess/dagm-pre.py`: target_or_test_data_use -> `class_name`
- `data_preprocess/dagm.py`: target_or_test_data_use -> `normal`
- `data_preprocess/visa.py`: default_external_training_data -> `visa`
- `data_preprocess/visa.py`: target_or_test_data_use -> `normal`
- `data_preprocess/clinicdb.py`: default_external_training_data -> `clinicdb`
- `data_preprocess/clinicdb.py`: target_or_test_data_use -> `normal`
- `data_preprocess/headct.py`: target_or_test_data_use -> `normal`
- `data_preprocess/br35h.py`: target_or_test_data_use -> `normal`
- `data_preprocess/sdd-pre.py`: target_or_test_data_use -> `normal`
- `data_preprocess/colondb.py`: default_external_training_data -> `colondb`
- `data_preprocess/colondb.py`: target_or_test_data_use -> `normal`
- `data_preprocess/mvtec.py`: default_external_training_data -> `mvtec`
- `data_preprocess/mvtec.py`: target_or_test_data_use -> `normal`
- `data_preprocess/mpdd.py`: target_or_test_data_use -> `normal`
- `data_preprocess/brain_mri.py`: target_or_test_data_use -> `normal`
- `data_preprocess/btad.py`: target_or_test_data_use -> `normal`
- `data_preprocess/headct-pre.py`: target_or_test_data_use -> `normal`
- `data_preprocess/dtd.py`: target_or_test_data_use -> `normal`
- `data_preprocess/endo.py`: target_or_test_data_use -> `normal`
- `data_preprocess/sdd.py`: target_or_test_data_use -> `normal`
- `data_preprocess/tn3k.py`: target_or_test_data_use -> `normal`
- `data_preprocess/isic.py`: target_or_test_data_use -> `normal`
- `dataset/__init__.py`: default_external_training_data -> `mvtec`
- `dataset/visa.py`: default_external_training_data -> `visa`
- `dataset/clinicdb.py`: default_external_training_data -> `clinicdb`
- `dataset/colondb.py`: default_external_training_data -> `colondb`
- `dataset/mvtec.py`: default_external_training_data -> `mvtec`

## Required Actions

- Disable all non-BMAD default training_data/testing_data settings.
- Use only source BMAD datasets for train/adaptation/validation.
- Use held-out BMAD target only for final inference and AUROC.
- Do not use target test thresholds, statistics, or checkpoint selection.

## Protocol Manifest Gate

- Manifest: `/mnt/ecc5c6c9-7631-4983-9ba0-1ec98729589b/wxy/MedHyCLIP/external_baselines/adaclip_bmad/results/dryrun/protocol_manifest.json`
- Failures:
  - `status` expected `predictions_exported or metrics_computed`, got `blocked`