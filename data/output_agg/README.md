# `data/output_agg/`

Precomputed aggregate outputs at overall, yearly, monthly, and position-specific granularity.

## Main files

- `overall_agg.csv`, `year_agg.csv`, `month_idx_agg.csv`
- `pos_overall.csv`, `pos_year.csv`, `pos_month_idx.csv`

## Typical use

These files are consumed by plotting and table-generation scripts to recreate weighted trend outputs without rerunning the full raw-data conversion pipeline.
