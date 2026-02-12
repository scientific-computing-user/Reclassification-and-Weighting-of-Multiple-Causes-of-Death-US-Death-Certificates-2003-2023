# `data/reference/`

Reference files used by analysis and validation scripts.

## Files

- `icd10_to_DL_lookup_v4.csv` — ICD-10 to broad-category lookup used by the pipeline.
- `cause_of_death_list_with_falls_v4.csv` — rule list used during ICD category conversion.
- `USA_Population5.txt` — population denominators for population-scaled trend analyses.

## Used by

- `scripts/pipeline/*.py`
- `scripts/reproducibility/validate_claims.py`
- `scripts/reproducibility/reproduce_snapshot.sh`
