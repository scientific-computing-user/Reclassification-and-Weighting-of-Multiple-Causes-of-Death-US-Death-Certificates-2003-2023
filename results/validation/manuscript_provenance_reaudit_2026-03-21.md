# Manuscript Provenance Re-Audit (2026-03-21)

## Scope

This note re-audits an earlier AI-generated "numeric discrepancy" report that claimed many mismatches between the manuscript/supplement and refreshed outputs.

The re-audit used exact manuscript locations and direct output provenance and did **not** accept a discrepancy unless the table-to-file mapping could be established with high confidence.

## Main conclusion

The earlier discrepancy report was **not reliable as evidence of real manuscript/output mismatch**.

Most of the large reported discrepancies were caused by comparing manuscript values that align with the **`Dup==Empty_first`** output family against values taken from **`Dup==Empty_last`**, without proving that the manuscript was intended to use the `last` variant.

## What the re-audit found

### Not proven as real discrepancies

- Main paper **Table 1** values and the surrounding paragraph percentages align with `Dup==Empty_first/Record_Entity_Comparison.xlsx`, not the `Dup==Empty_last` workbook used in the earlier report.
- Main paper **Table 2** values and the surrounding transition paragraph align with `Dup==Empty_first/Record_Entity_Comparison_Transition.xlsx`, not the `Dup==Empty_last` workbook used in the earlier report.
- The **concordance** paragraph and Supplement **S5A** totals align with aggregated yearly concordance files in `Dup==Empty_first`, not the `Dup==Empty_last` files cited by the earlier report.
- Supplement **S4** top COVID-promotion ICD rows align with `Dup==Empty_first/analyze_U071_ucod_v3.py=100-c.csv`, not the `Dup==Empty_last` version cited by the earlier report.
- The earlier claim that Supplement **S1** was stale was a mapping error: S1 is a **Record Axis ratio** table, not the Entity-side table the earlier report implicitly assumed.

### Still needs manual attention

- The manuscript paragraph reporting COVID promotions/demotions (`502,461` / `20,765`) does **not** map cleanly to the summary rows of the checked `analyze_U071` outputs. One of those manuscript values appears exactly in a different intermediate file. That paragraph should be checked manually against the intended generating source before any paper edit is made.

### Confirmed real manuscript issue

- The paragraph discussing **Table 4** for age `65+` COVID reductions says `64%` / `57%`.
- The actual table and generated workbook show `63%` / `56%`.
- This is the only discrepancy from that review that was confirmed with high confidence as a true manuscript mismatch rather than a file-selection or mapping problem.

## Practical interpretation for repository readers

- The canonical computational validation outputs in `claim_check.md` and `claim_metrics.json` remain the repository's main reproducibility artifacts.
- This provenance re-audit is a **separate manuscript-alignment check**. It should not be read as a replacement for the validation outputs, and the validation outputs should not be read as a direct one-to-one manuscript table extractor.
- No large data files were changed for this re-audit note.
