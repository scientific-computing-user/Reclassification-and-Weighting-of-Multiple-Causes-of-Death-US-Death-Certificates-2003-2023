#!/bin/sh

# Shaded area plots
# compute_all_W_schemes_v2.py
python ../code/compute_all_W_schemes_v2.py  I_converted_with_original-f2-7,9,10,11=sort=uniq-c=sort-rn.H.csv -o month_idx_agg_all_schemes.csv

# mortality_plot_popscaled_v8.py
python ../code/mortality_plot_popscaled_v8.py --agg_csv month_idx_agg_all_schemes.csv --pop_txt USA_Population5.txt --out_dir output_W0,W1,W2,W2A --all_scheme

exit
