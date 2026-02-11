# Recheck of 4 Reported Claim Discrepancies (2026-02-09)

## Executive Summary

I revalidated the four previously flagged claim mismatches against the full source data and the paper text.

Conclusion: these are **not** caused by PDF sign extraction errors.  
They are very likely caused by an ICD-to-category lookup coverage gap for Entity-axis codes in the current validation script.

## What Was Rechecked

1. Re-ran `validate_claims.py` on the full source file:
   - `processed2_GH24/Dup==A/all_I=converted_with_original.csv`
   - with `--entity-column sel_code` (the intended default path)
2. Re-ran with alternate columns (`ent_ucod`, `ent_ucod2`) to test column-choice artifacts.
3. Extracted and reviewed the paper text directly from the PDF (pages 9-11 and Table 1 page 26).
4. Audited lookup coverage for `ucod` and `sel_code`.
5. Ran a controlled sensitivity test: for unmapped ICDs starting with `S/T/V/W/X/Y`, temporarily map to `X` (Other External).

## Paper Text Check (Sign/Direction)

The paper explicitly states:
- Other External decreases by about `-54%` (page 9).
- Record-axis Transport receives `266,638` from Entity-axis Other External (page 11).
- Falls receives `218,625` from Entity-axis Other External (page 11).
- Suicide receives `139,167` from Entity-axis Other External (page 11).

These signs and transition directions are clear in the manuscript text and are not a PDF sign-flip artifact.

## Baseline Recheck (Current Script, `sel_code`)

`25/29` checks pass; the same 4 fail:

| Claim | Paper | Recheck | Difference |
|---|---:|---:|---:|
| Category change Other External (%) | -54.0 | +29.36 | +83.36 |
| Transition X->T | 266,638 | 786 | -265,852 |
| Transition X->F | 218,625 | 1,117 | -217,508 |
| Transition X->S | 139,167 | 72 | -139,095 |

## Why This Happens (Most Likely Root Cause)

Lookup audit results:
- Unmapped `ucod` rows: `206,910`
- Unmapped `sel_code` rows: `1,088,164`

Top unmapped `sel_code` values include injury/external-style ICDs (`T71`, `S720`, `T07`, `S065`, `T149`, `Y839`, etc.), which strongly affects the `X` category and transitions from `X` to `T/F/S`.

## Sensitivity Test (Coverage Fix Approximation)

When unmapped codes with prefixes `S/T/V/W/X/Y` are temporarily grouped into `X`:

| Claim | Paper | Sensitivity Result | Difference |
|---|---:|---:|---:|
| Category change Other External (%) | -54.0 | -53.60 | +0.40 |
| Transition X->T | 266,638 | 266,051 | -587 |
| Transition X->F | 218,625 | 218,235 | -390 |
| Transition X->S | 139,167 | 138,544 | -623 |

All four disputed claims become very close to the paper values under this mapping-coverage adjustment.

## Interpretation for the Team

- The four discrepancies are real in the **current validator configuration**, but they do **not** indicate that the paper claims are likely wrong.
- Evidence supports a reproducibility tooling issue: incomplete ICD lookup coverage (especially for Entity-axis injury/external codes), not a sign-parsing or PDF extraction problem.
- Main conclusion: this is a **mapping-spec alignment issue**, not a substantive contradiction of the paper.

## Recommended Next Step

Update the lookup/mapping used by `validate_claims.py` to fully cover Entity-axis injury/external ICD coding used in the paper pipeline, then rerun the standard claim report and replace the discrepancy note with the corrected validation output.
