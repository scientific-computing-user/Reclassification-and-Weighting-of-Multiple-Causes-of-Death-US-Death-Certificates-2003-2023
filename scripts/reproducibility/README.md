# `scripts/reproducibility/`

Packaged reproducibility entrypoints for this GitHub repository.

## `reproduce_snapshot.sh`

End-to-end reproducibility run using packaged snapshot files.

- Input:
  - `data/processed_snapshot/*.csv`
  - `data/reference/*`
  - optionally reconstructed `data/processed_snapshot/all_I=converted_with_original.csv`
- Output:
  - `results/reproduced/tables/*`
  - `results/reproduced/figures/*`
  - `results/reproduced/supplement/*`
  - `results/reproduced/validation/claim_check.md`
  - `results/reproduced/validation/claim_metrics.json`
- Usage:
  - `bash scripts/reproducibility/reproduce_snapshot.sh`
  - `bash scripts/reproducibility/reproduce_snapshot.sh <output_dir>`

## `rebuild_all_i_from_parts.sh`

Rebuilds `all_I=converted_with_original.csv` from split gzip parts.

- Input:
  - `data/processed_snapshot/all_I_parts/all_I=converted_with_original.csv.gz.part.*`
- Output:
  - `data/processed_snapshot/all_I=converted_with_original.csv`
- Usage:
  - `bash scripts/reproducibility/rebuild_all_i_from_parts.sh --work-dir data/processed_snapshot`

## `validate_claims.py`

Runs numerical claim checks against `all_I` data.

- Input:
  - `--all-i` path to `all_I=converted_with_original.csv`
  - `--lookup` path to ICD10->DL lookup
- Output:
  - Markdown claim report (`--output-md`)
  - JSON metrics/checks (`--output-json`)
- Usage:
  - `python3 scripts/reproducibility/validate_claims.py --all-i data/processed_snapshot/all_I=converted_with_original.csv --lookup data/reference/icd10_to_DL_lookup_v4.csv --external-prefix-fallback STVWXY --entity-column sel_code --output-json results/validation/claim_metrics.json --output-md results/validation/claim_check.md`

### Important option

- `--external-prefix-fallback STVWXY`:
  - Treats unmapped ICD prefixes `S/T/V/W/X/Y` as `Other External (X)` for category transition checks.
  - This is the validated configuration that aligns all reported manuscript claims in the packaged rerun.
