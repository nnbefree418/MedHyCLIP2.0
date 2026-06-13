# Lightweight Euclidean vs Hyperbolic Benchmark

## Summary

| Method | Total Params (M) | Trainable Params (M) | Extra Params (M) | Zero-shot Train Time / Epoch | Few-shot Train Time / Epoch | Zero-shot Infer Time / Image | Few-shot Infer Time / Image | FPS | Peak Memory |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MVFA (Euclidean) | 440.233 | 12.583 | 0.000 | 7.64 min +/- 1.000 s | 48.222 s +/- 9.070 s | 0.063 s +/- 0.001 s | 0.064 s +/- 0.006 s | zero 15.77 / few 15.61 | zero 13.22 GB / few 7.82 GB |
| Hyper-MVFA | 440.248 | 12.597 | 0.014 | 12.15 min +/- 1.644 s | 27.13 min +/- 13.96 min | 0.102 s +/- 0.003 s | 1.597 s +/- 0.827 s | zero 9.76 / few 0.63 | zero 15.68 GB / few 8.32 GB |

## Notes

- `Trainable Params` counts only `seg_adapters` and `det_adapters`, matching the optimizer target in the existing scripts.
- `Extra Params` is measured relative to MVFA (Euclidean).
- `Training Time / Epoch` is measured from wrapper wall-clock timestamps and includes the current scripts' epoch-end evaluation.
- `Inference Time / Image` is parsed from `tqdm` test throughput when available, otherwise from test wall time.
- `Peak Memory` is averaged over runs and reported as max memory used on a single monitored GPU.

## Failed Or Incomplete Runs

- None detected in the manifest.

## Detail CSV

- `benchmark_details.csv` contains per-run parsed values and log paths.
