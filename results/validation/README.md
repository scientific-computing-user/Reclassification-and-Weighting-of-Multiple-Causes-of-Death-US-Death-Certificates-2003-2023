# `results/validation/`

Canonical claim-validation and audit outputs for this repository.

## Primary files

- `claim_check.md` — canonical latest markdown claim report (`29/29` pass).
- `claim_metrics.json` — canonical latest machine-readable metrics/checks.
- `claim_check_reaudit_2026-02-12.md` — date-stamped re-audit markdown report (`29/29` pass).
- `claim_metrics_reaudit_2026-02-12.json` — date-stamped metrics/checks from the same re-audit.
- `claim_proof_docx_reaudit_2026-02-12.md` — proof matrix aligning DOCX claim values with re-audit outputs.
- `manuscript_provenance_reaudit_2026-03-21.md` — provenance-focused re-audit showing that a later AI discrepancy list mostly reflected wrong run-variant alignment (`Dup==Empty_first` vs `Dup==Empty_last`), with one confirmed manuscript-only issue in the Table 4 age-65+ COVID sentence.

## Supporting files

- `icd10_to_DL_lookup_v4_plus_STVWXY_to_X.csv` — lookup used for explicit fallback audit runs.
- `paper_text_patch_recommendation_2026-02-12.md` — minimal manuscript patch guidance.
- prior recheck and recomputation reports for traceability.
