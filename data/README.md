# `data/`

This directory contains the datasets used by the reproducibility workflow.

## Subfolders

- `reference/` — mapping tables and population denominators.
- `output_agg/` — aggregate monthly/yearly outputs used by plotting and trend scripts.
- `processed_snapshot/` — packaged processed tables used for reproducible reruns.
- `mcd/` — raw yearly mortality inputs (excluded from this GitHub package and shared separately by the authors).

## Notes

- The full `all_I=converted_with_original.csv` source can be reconstructed from `processed_snapshot/all_I_parts/`.
- Snapshot files are sufficient to reproduce the paper outputs and claim checks in this repository.
