# `docs/assets/`

Data assets used by the interactive website.

## Files

- `site_data.json` — compact aggregated dataset used by `docs/index.html`.
- `site_data.js` — fallback payload (`window.__SITE_DATA__`) so the site can load when opened from disk (`file://`) or if JSON fetch paths vary.

## Rebuild

From repository root:

```bash
python3 scripts/reproducibility/build_site_data.py
```

This regenerates `site_data.json` and `site_data.js` from:
- `data/output_agg/year_agg.csv`
- `data/output_agg/month_idx_agg.csv`
- `data/output_agg/overall_agg.csv`
- `results/validation/claim_metrics.json`
