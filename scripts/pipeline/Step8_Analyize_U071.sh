#!/bin/sh

# Analyze U071 icd10
( head -1 I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv ; grep U071 I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv ) > grep=U071=I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv

grep=U071=I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv

cat grep=U071=I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv | python ../code/analyze_U071_ucod_v4.py 100 -c > analyze_U071_ucod_v3.py=100-c.csv


exit
