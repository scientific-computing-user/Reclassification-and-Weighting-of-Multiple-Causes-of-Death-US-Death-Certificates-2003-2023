# Reclassification and Weighting of Multiple Causes of Death (US 2003–2023)

This is the clean, upload-ready GitHub package for the paper.

## Structure

- `paper/` — manuscript PDF
- `data/reference/` — lookup tables and population file
- `data/mcd/` — excluded from this GitHub package (provided separately peer-to-peer due size/licensing constraints)
- `data/output_agg/` — precomputed aggregate CSVs
- `data/processed_snapshot/` — precomputed processed tables used by the main reproducibility scripts
- `data/processed_snapshot/all_I_parts/` — split gzip chunks (<25 MiB each) for rebuilding `all_I=converted_with_original.csv`
- `scripts/pipeline/` — analysis scripts used in the paper workflow
- `scripts/reproducibility/` — reproducibility + validation entrypoints
- `results/reproduced/` — generated outputs included in this package
- `results/validation/` — claim-check reports from a full validation run
- `docs/` — GitHub Pages website (`docs/index.html`)
- `manifests/` — manifests for intentionally excluded oversized files

## Quick Start

```bash
python3 -m pip install -r requirements.txt
bash scripts/reproducibility/reproduce_snapshot.sh
```

Outputs are written to:
- `results/reproduced/`

## Claim Validation

Full claim validation is supported in this package via split chunks in:
- `data/processed_snapshot/all_I_parts/`

When `bash scripts/reproducibility/reproduce_snapshot.sh` starts, it automatically rebuilds:
- `data/processed_snapshot/all_I=converted_with_original.csv`

using:
- `scripts/reproducibility/rebuild_all_i_from_parts.sh`

Included reports from a completed full run:
- `results/validation/claim_check_snapshot.md`
- `results/validation/claim_metrics.json`

To run validation directly after reconstruction:

```bash
python3 scripts/reproducibility/validate_claims.py \
  --all-i data/processed_snapshot/all_I=converted_with_original.csv \
  --lookup data/reference/icd10_to_DL_lookup_v4.csv \
  --entity-column sel_code \
  --output-json results/validation/claim_metrics.json \
  --output-md results/validation/claim_check.md
```

## Large File Policy

GitHub file-size limits used for this packaging:
- Browser upload limit: `25 MiB` per file
- Git push hard limit: `100 MiB` per file
- This package includes some files in the `25–100 MiB` range, so publish it via `git push` (not browser drag-and-drop upload).

Oversized source input artifacts (`data/mcd/*.csv.xz`) are excluded from this GitHub package and shared separately peer-to-peer by the authors. The largest required CSV is included as split `<25 MiB` chunks for local reconstruction.

See:
- `manifests/excluded_large_files.csv`
- `manifests/all_I_parts_manifest.csv`
- `manifests/excluded_input_large_files.csv`
