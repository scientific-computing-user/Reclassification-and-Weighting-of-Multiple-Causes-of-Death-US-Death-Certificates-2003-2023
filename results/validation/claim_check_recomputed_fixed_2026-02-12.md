# Claim Validation Report

- Entity comparison column: `sel_code`
- Total deaths scanned: `56,986,831`
- Runtime (seconds): `463.7`
- Claim checks passed: `29/29`
- Claim checks failed: `0`

## Core Metrics

| Metric | Value |
|---|---:|
| Category concordance (%) | 84.518 |
| ICD concordance (%) | 68.638 |
| Pre-pandemic category concordance (%) | 85.490 |
| Pre-pandemic ICD concordance (%) | 69.809 |
| Pandemic category concordance (%) | 81.297 |
| Pandemic ICD concordance (%) | 64.761 |
| COVID promotions | 503,255 |
| COVID demotions | 20,655 |
| J189 share of promotions (%) | 43.787 |

## Paper Claim Checks

| Claim | Expected | Observed | Diff | Tol | Status |
|---|---:|---:|---:|---:|---|
| Total deaths | 56,986,831 | 56,986,831 | 0 | 0 | PASS |
| Part-I last-line multiple ICD count | 7,749,865 | 7,749,865 | 0 | 0 | PASS |
| Part-I last-line multiple ICD pct | 13.600 | 13.599 | -0.001 | 0.200 | PASS |
| Category concordance pct | 84.800 | 84.518 | -0.282 | 1.000 | PASS |
| ICD concordance pct | 68.900 | 68.638 | -0.262 | 0.600 | PASS |
| Pre-pandemic category concordance pct | 85.700 | 85.490 | -0.210 | 1.000 | PASS |
| Pre-pandemic ICD concordance pct | 70.100 | 69.809 | -0.291 | 0.800 | PASS |
| Pandemic category concordance pct | 81.600 | 81.297 | -0.303 | 1.000 | PASS |
| Pandemic ICD concordance pct | 65.000 | 64.761 | -0.239 | 0.800 | PASS |
| Category change COVID-19 pct | 92.000 | 92.369 | 0.369 | 3.000 | PASS |
| Category change Falls pct | 69.000 | 69.371 | 0.371 | 3.000 | PASS |
| Category change Transport pct | 44.000 | 43.516 | -0.484 | 3.000 | PASS |
| Category change Suicide pct | 25.000 | 25.068 | 0.068 | 3.000 | PASS |
| Category change Homicide pct | 30.000 | 30.207 | 0.207 | 3.000 | PASS |
| Category change Endocrine pct | 16.000 | 16.502 | 0.502 | 3.000 | PASS |
| Category change Cancer pct | 12.000 | 12.227 | 0.227 | 3.000 | PASS |
| Category change Other Natural pct | -14.000 | -13.276 | 0.724 | 3.000 | PASS |
| Category change Respiratory pct | -11.000 | -11.004 | -0.004 | 3.000 | PASS |
| Category change Other External pct | -54.000 | -53.598 | 0.402 | 6.000 | PASS |
| Transition R->V count | 288,936 | 288,922 | -14 | 5,000 | PASS |
| Transition N->V count | 119,181 | 119,143 | -38 | 5,000 | PASS |
| Transition X->T count | 266,638 | 266,051 | -587 | 5,000 | PASS |
| Transition X->F count | 218,625 | 218,235 | -390 | 5,000 | PASS |
| Transition X->S count | 139,167 | 138,544 | -623 | 5,000 | PASS |
| Transition A->P count | 87,860 | 87,843 | -17 | 5,000 | PASS |
| COVID promotions count | 502,461 | 503,255 | 794 | 5,000 | PASS |
| COVID demotions count | 20,765 | 20,655 | -110 | 1,000 | PASS |
| COVID promotion/demotion ratio | 24.000 | 24.365 | 0.365 | 2.000 | PASS |
| J189 share of promotions pct | 43.800 | 43.787 | -0.013 | 2.000 | PASS |

## Category Count Changes (Record vs Entity)

| DL | Category | Entity | Record | Change (%) |
|---|---|---:|---:|---:|
| V | COVID-19 | 522,023 | 1,004,208 | 92.369 |
| F | Falls | 386,198 | 654,109 | 69.371 |
| T | Transport | 637,608 | 915,070 | 43.516 |
| S | Suicide | 609,356 | 762,111 | 25.068 |
| H | Homicide | 304,539 | 396,532 | 30.207 |
| E | Endocrine | 2,189,098 | 2,550,348 | 16.502 |
| C | Cancer | 11,205,911 | 12,576,095 | 12.227 |
| B | Circulatory | 17,728,281 | 17,815,572 | 0.492 |
| D | Digestive | 1,672,700 | 1,731,328 | 3.505 |
| A | Alcohol-related | 682,890 | 679,891 | -0.439 |
| P | Drug Poisoning | 1,129,804 | 1,229,718 | 8.843 |
| X | Other External | 1,447,386 | 671,609 | -53.598 |
| N | Other Natural | 12,329,332 | 10,692,497 | -13.276 |
| R | Respiratory | 5,964,045 | 5,307,742 | -11.004 |
