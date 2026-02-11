#!/bin/sh

# Step 1.a Get the Make the big_death_input.csv file
# Rscript code/deaths-masr-icd10.r
# Run from Dup==A directory below processed2_GH24

echo "date ; for year in $(seq 2003 2023) ; do xz -dc ../mcd/usa-deaths-masr-icd10_${year}.csv.xz ; done > ../output/big_death_input.csv ; date"

date ; for year in $(seq 2003 2023) ; do xz -dc ../mcd/usa-deaths-masr-icd10_${year}.csv.xz ; done > ../output/big_death_input.csv ; date

echo "Done: Step1_Convert_mcd_usa-deaths-masr-icd10_to_big_death_input_v3.sh"

exit

