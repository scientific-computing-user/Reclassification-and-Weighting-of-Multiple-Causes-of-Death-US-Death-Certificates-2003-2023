# `scripts/pipeline/`

Core analysis scripts from the project pipeline.

## Main scripts (inputs, outputs, usage)

- `create_record_entity_table_v2.py`
  - Input: `I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv`
  - Output: Record vs Entity Excel summary tables.
  - Example:
    - `python3 scripts/pipeline/create_record_entity_table_v2.py <input.csv> --years 2003-2023 --label "USA 2003-2023" --output results/reproduced/tables/Record_Entity_Comparison_2003-2023.xlsx`

- `create_disease_weight_tables_v4.py`
  - Input: `I_converted_with_original-f2,3,6,9,ag=...H.csv`
  - Output: disease-weight tables for all years and pandemic years.
  - Example:
    - `python3 scripts/pipeline/create_disease_weight_tables_v4.py <input.csv> --all-years 2003-2023 --pandemic-years 2020-2023 --output results/reproduced/tables/Disease_Weights`

- `trend_analysis_Table20.py`
  - Input: `month_idx_agg_all_schemes.csv`, `USA_Population5.txt`
  - Output: `Table20_trend_analysis.xlsx`.
  - Example:
    - `python3 scripts/pipeline/trend_analysis_Table20.py data/processed_snapshot/month_idx_agg_all_schemes.csv data/reference/USA_Population5.txt`

- `mortality_plot_popscaled_v8.py`
  - Input: aggregate monthly CSV + population text.
  - Output: population-scaled area chart PNGs by scheme and age stratum.
  - Example:
    - `python3 scripts/pipeline/mortality_plot_popscaled_v8.py --agg_csv data/processed_snapshot/month_idx_agg_all_schemes.csv --pop_txt data/reference/USA_Population5.txt --out_dir results/reproduced/figures --all_scheme`

- `analyze_rADS_W2A+++_v2.py`
  - Input: converted records table.
  - Output: `rADS_W2A_analysis_*.xlsx/.csv`.
  - Example:
    - `python3 scripts/pipeline/analyze_rADS_W2A+++_v2.py <input.csv> --years 2003-2023 --output results/reproduced/supplement/rADS_W2A_analysis_2003-2023`

- `analyze_questions_Q1-Q38_v2.py`
  - Input: gzipped Q1–Q38 source table.
  - Output: TSV summary.
  - Example:
    - `python3 scripts/pipeline/analyze_questions_Q1-Q38_v2.py data/processed_snapshot/I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz results/reproduced/supplement/questions_Q1-Q38.tsv`

- `ucod_preference.py`
  - Input: `I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv`
  - Output: UCOD preference CSV to stdout.
  - Example:
    - `python3 scripts/pipeline/ucod_preference.py <input.csv> 0 > results/reproduced/supplement/ucod_preference.csv`

- `convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py`
  - Input: raw converted deaths file + mapping + lookup.
  - Output: `converted_with_original.csv`, aggregate tables, `matched_icd10_codes.csv`.
  - Example:
    - `python3 scripts/pipeline/convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py --input <big_death_input.csv> --map data/reference/cause_of_death_list_with_falls_v4.csv --lookup data/reference/icd10_to_DL_lookup_v4.csv --output converted_with_original.csv`

## Shell wrappers

`Step*.sh` scripts are historical wrapper stages used during the original end-to-end build process.
