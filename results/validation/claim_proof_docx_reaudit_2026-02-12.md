# Re-Audit Proof Against DOCX Claims (2026-02-12)

## Summary

- Claims validated by computation: `29/29`
- DOCX alignment: `matches_expected=26`, `matches_observed=3`, `not_found=0`
- `matches_observed` indicates manuscript values were updated from legacy constants and align with recomputed values.

## Previously Disputed 4 Claims

| Claim | Expected Constant | Observed Re-audit | Status | DOCX Alignment |
|---|---:|---:|---|---|
| Category change Other External pct | -54.000 | -53.598 | PASS | matches_expected (-54) |
| Transition X->T count | 266,638 | 266,051 | PASS | matches_observed (266,051) |
| Transition X->F count | 218,625 | 218,235 | PASS | matches_observed (218,235) |
| Transition X->S count | 139,167 | 138,544 | PASS | matches_observed (138,544) |

## Full Claim Matrix

| Claim | Expected | Observed | Tol | Status | DOCX Alignment |
|---|---:|---:|---:|---|---|
| Total deaths | 56,986,831 | 56,986,831 | 0 | PASS | matches_expected |
| Part-I last-line multiple ICD count | 7,749,865 | 7,749,865 | 0 | PASS | matches_expected |
| Part-I last-line multiple ICD pct | 13.600 | 13.599 | 0.200 | PASS | matches_expected |
| Category concordance pct | 84.800 | 84.518 | 1.000 | PASS | matches_expected |
| ICD concordance pct | 68.900 | 68.638 | 0.600 | PASS | matches_expected |
| Pre-pandemic category concordance pct | 85.700 | 85.490 | 1.000 | PASS | matches_expected |
| Pre-pandemic ICD concordance pct | 70.100 | 69.809 | 0.800 | PASS | matches_expected |
| Pandemic category concordance pct | 81.600 | 81.297 | 1.000 | PASS | matches_expected |
| Pandemic ICD concordance pct | 65.000 | 64.761 | 0.800 | PASS | matches_expected |
| Category change COVID-19 pct | 92.000 | 92.369 | 3.000 | PASS | matches_expected |
| Category change Falls pct | 69.000 | 69.371 | 3.000 | PASS | matches_expected |
| Category change Transport pct | 44.000 | 43.516 | 3.000 | PASS | matches_expected |
| Category change Suicide pct | 25.000 | 25.068 | 3.000 | PASS | matches_expected |
| Category change Homicide pct | 30.000 | 30.207 | 3.000 | PASS | matches_expected |
| Category change Endocrine pct | 16.000 | 16.502 | 3.000 | PASS | matches_expected |
| Category change Cancer pct | 12.000 | 12.227 | 3.000 | PASS | matches_expected |
| Category change Other Natural pct | -14.000 | -13.276 | 3.000 | PASS | matches_expected |
| Category change Respiratory pct | -11.000 | -11.004 | 3.000 | PASS | matches_expected |
| Category change Other External pct | -54.000 | -53.598 | 6.000 | PASS | matches_expected |
| Transition R->V count | 288,936 | 288,922 | 5,000 | PASS | matches_expected |
| Transition N->V count | 119,181 | 119,143 | 5,000 | PASS | matches_expected |
| Transition X->T count | 266,638 | 266,051 | 5,000 | PASS | matches_observed |
| Transition X->F count | 218,625 | 218,235 | 5,000 | PASS | matches_observed |
| Transition X->S count | 139,167 | 138,544 | 5,000 | PASS | matches_observed |
| Transition A->P count | 87,860 | 87,843 | 5,000 | PASS | matches_expected |
| COVID promotions count | 502,461 | 503,255 | 5,000 | PASS | matches_expected |
| COVID demotions count | 20,765 | 20,655 | 1,000 | PASS | matches_expected |
| COVID promotion/demotion ratio | 24.000 | 24.365 | 2.000 | PASS | matches_expected |
| J189 share of promotions pct | 43.800 | 43.787 | 2.000 | PASS | matches_expected |
