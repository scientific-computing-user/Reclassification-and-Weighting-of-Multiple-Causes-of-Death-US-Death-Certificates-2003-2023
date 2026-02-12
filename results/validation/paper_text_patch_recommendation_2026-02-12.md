# Paper Text Patch Recommendation (2026-02-12)

## Bottom line

After revalidation, the 4 disputed claims are resolved when the ICD mapping applies this rule:
- for codes missing from `icd10_to_DL_lookup_v4.csv`, if the ICD starts with `S/T/V/W/X/Y`, classify as `X` (Other External).

With that mapping, claim validation is `29/29` passes.
Reference output: `/tmp/claim_check_stvwxy_to_x.md` and `/tmp/claim_metrics_stvwxy_to_x.json`.

## Verified values to use (2003-2023)

- Other External relative change: `-53.598%` (rounds to `-54%`)
- Transition `X->T`: `266,051`
- Transition `X->F`: `218,235`
- Transition `X->S`: `138,544`

Context denominators (Record Axis totals):
- Transport: `915,070` (`29.07%` from `X`)
- Falls: `654,109` (`33.36%` from `X`)
- Suicide: `762,111` (`18.18%` from `X`)

## Minimal manuscript edits

### 1) Methods (Death certificate data section, page 6)

Add one sentence after the ICD-to-category mapping description:

> For ICD-10 codes not directly matched in the lookup table, external-injury prefixes (`S/T/V/W/X/Y`) were grouped into the broad category Other External (`X`) for Entity-vs-Record category-transition analyses.

Why: this is the root-cause clarification that resolves the four discrepancies.

### 2) Results (Entity versus Record Axis, page 11, transitions paragraph)

Replace the three sentences with exact counts by this updated block:

> Transport deaths in the Record Axis receive 266,051 deaths (29.1% of the 915,070 total in the Record Axis) from Other External causes in the Entity Axis. Other External causes are also re-classified as Falls (218,235; 33.4% of Falls total deaths of 654,109 in the Record Axis), and Suicide receives 138,544 deaths (18.2% of its total of 762,111 in the Record Axis) from Other External causes in the Entity Axis.

Notes:
- If you prefer rounded prose, keep `29%`, `33%`, and `18.2%`.
- If tables are regenerated for 2024, update these six numbers from the new Table 2/Record totals.

### 3) Results (page 9, category-change sentence)

No mandatory change required; `-54%` is still correct rounding from `-53.598%`.
If you want strict numeric consistency, change `-54%` to `-53.6%`.

## What **not** to change

- Do not reverse signs for the four disputed claims.
- Do not change transition directions (`X->T`, `X->F`, `X->S` remain the correct direction).

