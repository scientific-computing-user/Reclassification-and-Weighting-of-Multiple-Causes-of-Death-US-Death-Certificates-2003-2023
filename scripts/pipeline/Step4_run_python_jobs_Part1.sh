#!/bin/sh

# I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv
# I_converted_with_original-f3,5,6,9,ag=sort=uniq-c=sort-rn.H.csv
# I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv
# I_converted_with_original-f2,3,4,5,7,9,10,11=sort=uniq-c=sort-rn.H.csv

rm I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz
gzip -k I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv

# Questions with analyze_questions_Q1-Q38_v2.py
python ../code/analyze_questions_Q1-Q38_v2.py I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz I_converted_with_original-f3,6,19,24=sort=uniq-c=sort-rn.H.tsv

# analyze_uds_v14.py
python ../code/analyze_uds_v14.py I_converted_with_original-f3,5,6,9,ag=sort=uniq-c=sort-rn.H.csv --xmatrix x_uc_rUDS_ALL.csv --years 2003-2023

# analyze_rADS_W2A+++_v2.py 2003-2023
python ../code/analyze_rADS_W2A+++_v2.py I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv --years 2003-2023

# analyze_rADS_W2A+++_v2.py 2020-2023
python ../code/analyze_rADS_W2A+++_v2.py I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv --years 2020-2023

exit
