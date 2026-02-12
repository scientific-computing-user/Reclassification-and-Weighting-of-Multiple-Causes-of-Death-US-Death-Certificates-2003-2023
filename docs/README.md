# `docs/`

GitHub Pages assets.

## Files

- `index.html` — interactive website with charts, transition network, and claim-validation explorer.
- `assets/site_data.json` — aggregated data backing the interactive views.

## Notes

- If GitHub Pages source is set to `/docs` on `main`, the website is available at:
  - `https://scientific-computing-user.github.io/Reclassification-and-Weighting-of-Multiple-Causes-of-Death-US-Death-Certificates-2003-2023/`
- Rebuild site data after updating validation or aggregate outputs:
  - `python3 scripts/reproducibility/build_site_data.py`
