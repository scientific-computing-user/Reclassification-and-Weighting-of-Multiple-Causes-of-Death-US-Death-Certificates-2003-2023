# `docs/assets/`

Data assets used by the interactive website.

## Files

- `site_data.json` — compact aggregated dataset used by `docs/index.html`.

## Rebuild

From repository root:

```bash
python3 scripts/reproducibility/build_site_data.py
```

This regenerates `site_data.json` from:
- `data/output_agg/year_agg.csv`
- `data/output_agg/month_idx_agg.csv`
- `data/output_agg/overall_agg.csv`
- `results/validation/claim_metrics.json`
