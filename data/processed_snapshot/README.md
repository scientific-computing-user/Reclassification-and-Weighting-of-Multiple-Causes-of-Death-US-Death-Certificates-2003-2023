# `data/processed_snapshot/`

Snapshot tables required for reproducibility from this GitHub package.

## Key files

- `I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv`
- `I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv`
- `I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv`
- `month_idx_agg_all_schemes.csv`
- `matched_icd10_codes=use-last_eUCOD.csv` (used for lookup coverage audits)

## Rebuilding large source file

- Split parts live in `all_I_parts/`.
- Use: `bash scripts/reproducibility/rebuild_all_i_from_parts.sh --work-dir data/processed_snapshot`
- Output: `data/processed_snapshot/all_I=converted_with_original.csv`
