#!/bin/sh


# Table 1 and 2
echo "python ../create_record_entity_table_v2.py I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv --label \"USA 2002-2013\""
python ../create_record_entity_table_v2.py I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv --label "USA 2002-2013"


# Table 3
echo "python ../code/ucod_preference.py I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv 0 > ucod_preference.py=.log.csv"
python ../code/ucod_preference.py I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv 0 > ucod_preference.py=.log.csv


# Tables 4 and 5
echo "python ../code/create_disease_weight_tables_v3.py I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv --all-years 2003-2023 --pandemic-years 2020-2023"
python ../code/create_disease_weight_tables_v3.py I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv --all-years 2003-2023 --pandemic-years 2020-2023


# Trend Analysis s
echo "python ../code/trend_analysis_Table20.py month_idx_agg_all_schemes.csv USA_Population5.txt"
python ../code/trend_analysis_Table20.py month_idx_agg_all_schemes.csv USA_Population5.txt


exit 
