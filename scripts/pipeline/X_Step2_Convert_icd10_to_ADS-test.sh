#!/bin/sh

# New 19Jan2026

# python3 convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py --input your_death_data.csv --map cause_of_death_list_with_falls_v2.csv --lookup icd10_to_DL_lookup_v2.csv --output converted_output.csv
# With sensitivity analysis (use last code on line 1 as UCOD)
# python3 convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py --input your_death_data.csv --map cause_of_death_list_with_falls_v2.csv --lookup icd10_to_DL_lookup_v2.csv --output converted_output.csv --use-ent-ucod2
# With debug output
# python3 convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py --input your_death_data.csv --map cause_of_death_list_with_falls_v2.csv --lookup icd10_to_DL_lookup_v2.csv --output converted_output.csv --debug


# Step 1

cd /Users/levitt/levitt/NewProjects25/USDeathCertificates; /opt/homebrew/bin/pypy3 code/convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py --use-ent-ucod2 --debug --input data/processed2/head-10000=big_death_input_v3.csv --output converted_with_original_v3-10000_use-last_eUCOD.csv  --map data/processed2/cause_of_death_list_with_falls_v2.csv --lookup data/processed2/icd10_to_DL_lookup_v2.csv >& convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py=.F-new=use-last_eUCOD.log

mv matched_icd10_codes.csv  matched_icd10_codes=use-last_eUCOD.csv
mv unmatched_lines.csv      unmatched_lines=use-last_eUCOD.csv

cd /Users/levitt/levitt/NewProjects25/USDeathCertificates; /opt/homebrew/bin/pypy3 code/convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py                         --input data/processed2/head-10000=big_death_input_v3.csv --output converted_with_original_v3-10000_use-first_eUCOD.csv --map data/processed2/cause_of_death_list_with_falls_v2.csv --lookup data/processed2/icd10_to_DL_lookup_v2.csv >& convert_icd10_to_abbreviated_causes_aggregate_v3_6P_15.py=.F-new=use-first_eUCOD.log

mv matched_icd10_codes.csv  matched_icd10_codes=use-first_eUCOD.csv
mv unmatched_lines.csv      unmatched_lines=use-first_eUCOD.csv

ls -tal /Users/levitt/levitt/NewProjects25/USDeathCertificates | head -20

ls -tal /Users/levitt/levitt/NewProjects25/USDeathCertificates/data/processed | head -10

exit
