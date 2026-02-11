#!/bin/sh

# Step 1 (Dup==A, last degenerate eUCOD) to icd10_to_DL_lookup_v4C
# Run from Dup==A directory below processed2_GH24

echo "/opt/homebrew/bin/pypy3 ../code/convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py --input ../output/big_death_input.csv --output converted_with_original_use-last_eUCOD.csv --map ../cause_of_death_list_with_falls_v4.csv --lookup ../icd10_to_DL_lookup_v4.csv --use-ent-ucod --debug >& convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py=use-last_eUCOD.log"

/opt/homebrew/bin/pypy3 ../code/convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py --input ../output/big_death_input.csv --output converted_with_original_use-last_eUCOD.csv --map ../cause_of_death_list_with_falls_v4.csv --lookup ../icd10_to_DL_lookup_v4.csv --use-ent-ucod --debug >& convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py=use-last_eUCOD.log

echo "mv matched_icd10_codes.csv  matched_icd10_codes=use-last_eUCOD.csv; mv unknown_lines.csv        unknown_lines=use-last_eUCOD.csv"
mv matched_icd10_codes.csv  matched_icd10_codes=use-last_eUCOD.csv; mv unknown_lines.csv        unknown_lines=use-last_eUCOD.csv

echo "ls -tal | head -20"
ls -tal | head -20

echo "Done: Step2_Convert_icd10_to_ADS_use-last_eUCOD.sh 

exit
