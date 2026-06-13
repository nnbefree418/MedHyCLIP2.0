# AdaCLIP BMAD Leave-One-Out Baseline

This directory contains an isolated wrapper for reproducing AdaCLIP under the
BMAD leave-one-out protocol used by MedHyCLIP. It must not modify existing
project code, data, configs, training scripts, or paper files.

## Scope

- Official AdaCLIP code location: `external_baselines/AdaCLIP/`.
- BMAD wrapper location: `external_baselines/adaclip_bmad/`.
- Target datasets are used only for final inference and AUROC computation.
- Training, prompt adaptation, checkpoint selection, threshold selection,
  hyperparameter selection, and validation must use only the five non-target
  BMAD source datasets for each held-out run.

## Dataset Mapping

The wrapper reads `prompt.py` and uses the existing `REAL_NAME` mapping:

- HIS -> `Histopathology` -> `histopathological image`
- ChestXray -> `Chest` -> `Chest X-ray film`
- OCT17 -> `Retina_OCT2017` -> `retinal OCT`
- BrainMRI -> `Brain` -> `Brain`
- LiverCT -> `Liver` -> `Liver`
- RESC -> `Retina_RESC` -> `retinal OCT`

The generated file is `configs/prompt_mapping.yaml`.

## Metrics

- AC: image-level AUROC.
- AS: pixel-level AUROC.
- AS is computed only for BrainMRI, LiverCT, and RESC.
- HIS, ChestXray, and OCT17 use `--` for AS.
- No partial AUC is computed or reported.
- AC average is over all six datasets.
- AS average is over BrainMRI, LiverCT, and RESC.

## Dry-Run

Run the BrainMRI dry-run:

```bash
bash external_baselines/adaclip_bmad/run_dryrun.sh
```

The dry-run generates BMAD indices, validates source/target leakage, audits the
official AdaCLIP code if present, and attempts to enter the AdaCLIP runner
boundary. It is considered passed only when official AdaCLIP code exists at
`external_baselines/AdaCLIP/` and the audit/runner both exit successfully.

## Full LOSO

Full LOSO is gated and should only be launched after the dry-run report is
accepted:

```bash
RUN_FULL_LOSO=1 bash external_baselines/adaclip_bmad/run_adaclip_bmad_loso.sh
```

## Outputs

- `results/dryrun/dryrun_report.md`
- `results/dryrun/index_validation.md`
- `results/dryrun/adaclip_protocol_audit.md`
- `results/logs/<dataset>.log`
- `results/predictions/<dataset>/`
- `results/adaclip_bmad_loso_summary.csv`
- `results/adaclip_table_row.tex`
- `results/run_metadata.json`
- `results/feasibility_report.md` if official AdaCLIP cannot be fairly adapted

## Current Integration Boundary

The current wrapper does not edit official AdaCLIP files. If official code
requires changes that cannot be expressed through config or wrapper calls, add
patch files under `patches/` and document why they are necessary.
