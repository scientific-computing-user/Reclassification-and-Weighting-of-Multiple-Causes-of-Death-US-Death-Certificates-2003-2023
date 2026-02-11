#!/usr/bin/env python3
"""
Mortality Plot with Population Scaling - Version 6
===================================================
Generates displaced area plots of monthly deaths by cause for 4 weighting schemes.

Schemes:
  1 (W0):  UCOD only - 100% to first letter (opaque white background)
  2 (W1):  Dobson - 50% to first, 50% split among others (transparent)
  4 (W2):  Equal on unique letters (transparent)
  5 (W2A): Equal on all positions including repeats (transparent)

Features:
- Population scaling: adjusted_deaths = raw_deaths * (pop_2023 / pop_year)
- Separate scaling for each age group (ALL, GE65, LT65)
- Y-axis scales optimized for ALL ages (GE65 may overflow with W2A)
- W0 plots: opaque white background (base layer)
- W1/W2/W2A plots: transparent background (for overlay on W0)
- Excel output for all 4 schemes

Usage:
    python mortality_plot_popscaled_v6.py --agg_csv path/to/month_idx_agg.csv \
                                          --pop_txt path/to/USA_Population5.txt \
                                          --out_dir output_directory \
                                          --scheme W0 \
                                          --all_schemes
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import sys

# ============================================================================
# CONSTANTS
# ============================================================================

START_YEAR_DATA = 2003
END_YEAR_DATA = 2023
REFERENCE_YEAR = 2023

# Scheme mapping: name -> number
SCHEME_MAP = {
    'W0': 1,   # UCOD only
    'W1': 2,   # Dobson
    'W2': 4,   # Equal/unique
    'W2A': 5,  # Equal/all positions
}

SCHEME_NAMES = {
    1: 'W0 (UCOD)',
    2: 'W1 (Dobson)',
    4: 'W2 (Equal/unique)',
    5: 'W2A (Equal/all)',
}

# Which schemes get transparent backgrounds (for overlay)
TRANSPARENT_SCHEMES = {2, 4, 5}  # W1, W2, W2A
OPAQUE_SCHEMES = {1}              # W0 only

LETTER_INFO = {
    'B': ('Circulatory', '#6FB559'),
    'C': ('Cancer', '#E68722'),
    'N': ('Other natural', '#00808A'),
    'R': ('Respiratory', '#EA776F'),
    'E': ('Endocrine', '#2763B0'),
    'D': ('Digestive', '#E6B224'),
    'V': ('COVID-19', '#93C8E3'),
    'P': ('Drug poisoning', '#D3C772'),
    'T': ('Transport', '#C75B6D'),
    'S': ('Suicide', '#DA2C75'),
    'A': ('Alcohol-related', '#00A777'),
    'X': ('Other external', '#D11B8A'),
    'F': ('Falls', '#D11B8A'),
    'H': ('Homicide', '#762E97'),
}

DISEASE_NAMES_SHORT = {
    'V': 'COVID-19', 'B': 'Circulatory', 'C': 'Cancer', 'N': 'Other natural',
    'R': 'Respiratory', 'E': 'Endocrine', 'D': 'Digestive', 'P': 'Drug poisoning',
    'X': 'Other external', 'T': 'Transport', 'S': 'Suicide', 'A': 'Alcohol-related',
    'F': 'Falls', 'H': 'Homicide'
}

# Plot order (V on top, H at bottom)
DISEASE_ORDER = ['V', 'B', 'C', 'N', 'R', 'E', 'D', 'P', 'X', 'T', 'S', 'A', 'F', 'H']

# Scale groups for plotting - OPTIMIZED FOR ALL AGES
# (GE65 with W2A will overflow, which is expected)
SCALE_GROUPS_ALL = [
    {'scale': 6000,   'letters': ['H', 'F', 'A', 'S', 'T']},
    {'scale': 28000,  'letters': ['X', 'P', 'D', 'E']},
    {'scale': 130000, 'letters': ['R', 'N', 'C', 'B', 'V']},
]

SCALE_GROUPS_GE65 = [
    {'scale': 6000,   'letters': ['H', 'F', 'A', 'S', 'T']},
    {'scale': 28000,  'letters': ['X', 'P', 'D', 'E']},
    {'scale': 130000, 'letters': ['R', 'N', 'C', 'B', 'V']},
]

SCALE_GROUPS_LT65 = [
    {'scale': 6000,  'letters': ['H', 'F', 'A', 'S', 'T']},
    {'scale': 12000, 'letters': ['X', 'P', 'D', 'E']},
    {'scale': 26000, 'letters': ['R', 'N', 'C', 'B', 'V']},
]

def get_scale_groups(bucket_label):
    """Return appropriate scale groups for the age bucket."""
    if bucket_label == 'LT65':
        return SCALE_GROUPS_LT65
    elif bucket_label == 'GE65':
        return SCALE_GROUPS_GE65
    else:
        return SCALE_GROUPS_ALL


def check_overflow(mat, bucket_label, scheme_num):
    """Check for values exceeding scale bounds and report them."""
    scale_groups = get_scale_groups(bucket_label)
    overflow_found = False
    scheme_name = SCHEME_NAMES.get(scheme_num, f'Scheme{scheme_num}')
    
    for group in scale_groups:
        scale = group['scale']
        for ltr in group['letters']:
            if ltr in mat.index:
                max_val = mat.loc[ltr].max()
                if max_val > scale:
                    max_month_idx = mat.loc[ltr].idxmax()
                    year = (max_month_idx // 12) + START_YEAR_DATA
                    month = (max_month_idx % 12) + 1
                    print(f"[WARN] OVERFLOW: {scheme_name} {bucket_label} {ltr} ({DISEASE_NAMES_SHORT[ltr]}): "
                          f"max={max_val:.0f} > scale={scale} at {year}-{month:02d}", file=sys.stderr)
                    overflow_found = True
    
    return overflow_found

GAP_BETWEEN_GROUPS = 0.3
FIXED_BAND_HEIGHT = 1.0

# Map from internal names to file bucket names
# File has: GE65, LT65 (no ALL - must be computed)
FILE_BUCKET_MAP = {
    'ALL': None,      # Must compute from GE65 + LT65
    'PLUS65': 'GE65',
    'UNDER65': 'LT65',
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_death_data(agg_path, scheme_num):
    """Load and filter death data for a specific scheme."""
    print(f"[INFO] Loading death data from {agg_path} for scheme {scheme_num}", file=sys.stderr)
    df = pd.read_csv(agg_path)
    df = df[df['scheme'] == scheme_num]
    return df


def load_population_data(pop_path):
    """Load population data and compute totals by age bucket."""
    print(f"[INFO] Loading population data from {pop_path}", file=sys.stderr)
    pop_raw = pd.read_csv(pop_path, skiprows=3, sep=r'\s+', engine='python',
                          header=None, names=['Year', 'Age', 'Pop_F', 'Pop_M', 'Pop_T'], dtype=str)
    pop_raw['Year'] = pop_raw['Year'].str.extract(r'(\d{4})').astype(int)
    pop_raw['Pop_T'] = pd.to_numeric(pop_raw['Pop_T'], errors='coerce')

    results = []
    for year in pop_raw['Year'].unique():
        yr_data = pop_raw[pop_raw['Year'] == year]

        total_pop = yr_data['Pop_T'].sum()
        results.append({'Year': year, 'bucket': 'ALL', 'pop': total_pop})

        plus65_pop = 0
        under65_pop = 0
        for _, row in yr_data.iterrows():
            age_str = row['Age']
            pop_val = row['Pop_T']
            if pd.isna(pop_val):
                continue
            if age_str == '100+':
                plus65_pop += pop_val
            elif '-' in age_str:
                low = int(age_str.split('-')[0])
                if low >= 65:
                    plus65_pop += pop_val
                else:
                    under65_pop += pop_val
            else:
                try:
                    age = int(age_str)
                    if age >= 65:
                        plus65_pop += pop_val
                    else:
                        under65_pop += pop_val
                except:
                    pass

        results.append({'Year': year, 'bucket': 'PLUS65', 'pop': plus65_pop})
        results.append({'Year': year, 'bucket': 'UNDER65', 'pop': under65_pop})

    return pd.DataFrame(results)


def compute_scaling_factors(pop_df, ref_year=REFERENCE_YEAR):
    """Compute pop_ref / pop_year for each bucket and year."""
    factors = {}
    for bucket in ['ALL', 'PLUS65', 'UNDER65']:
        ref_pop = pop_df[(pop_df['Year'] == ref_year) & (pop_df['bucket'] == bucket)]['pop'].values[0]
        factors[bucket] = {}
        for year in pop_df['Year'].unique():
            year_pop = pop_df[(pop_df['Year'] == year) & (pop_df['bucket'] == bucket)]['pop'].values
            if len(year_pop) > 0:
                factors[bucket][year] = ref_pop / year_pop[0]
    return factors


def prepare_monthly_matrix(death_df, bucket, scaling_factors, normalize=True):
    """Prepare monthly death matrix for a bucket, optionally scaled by population."""
    
    # Handle bucket mapping - file has GE65/LT65, we need to compute ALL
    if bucket == 'ALL':
        # Combine GE65 and LT65 data
        bucket_df = death_df[death_df['bucket'].isin(['GE65', 'LT65'])]
    elif bucket == 'PLUS65':
        bucket_df = death_df[death_df['bucket'] == 'GE65']
    elif bucket == 'UNDER65':
        bucket_df = death_df[death_df['bucket'] == 'LT65']
    else:
        bucket_df = death_df[death_df['bucket'] == bucket]

    # Pivot: letter x month_idx
    pivot = bucket_df.groupby(['letter', 'month_idx'])['weight'].sum().reset_index()
    mat = pivot.pivot(index='letter', columns='month_idx', values='weight').fillna(0)

    # Apply scaling factors only if normalizing
    if normalize:
        n_months = 252
        for m_idx in range(n_months):
            year = (m_idx // 12) + START_YEAR_DATA
            if year in scaling_factors[bucket]:
                factor = scaling_factors[bucket][year]
                if m_idx in mat.columns:
                    mat[m_idx] = mat[m_idx] * factor

    return mat


# ============================================================================
# PLOTTING
# ============================================================================

def draw_plot(mat, scheme_num, bucket_label, out_dir, start_month=0, end_month=251, normalize=True):
    """Draw the displaced area plot."""
    # Get appropriate scale groups for this age bucket
    scale_groups = get_scale_groups(bucket_label)
    
    ordered_letters = []
    for group in scale_groups:
        for ltr in group['letters']:
            if ltr in mat.index:
                ordered_letters.append(ltr)

    mat = mat.reindex(ordered_letters)

    # Filter to month range
    cols_in_range = [c for c in mat.columns if start_month <= c <= end_month]
    mat_filtered = mat[cols_in_range]

    num_gaps = len(scale_groups) - 1
    gap_height = GAP_BETWEEN_GROUPS * FIXED_BAND_HEIGHT
    overall_y_max = FIXED_BAND_HEIGHT * len(ordered_letters) + num_gaps * gap_height

    # Determine if transparent or opaque
    is_transparent = scheme_num in TRANSPARENT_SCHEMES
    
    if is_transparent:
        fig, ax = plt.subplots(figsize=(18, 14), facecolor='none')
        ax.set_facecolor('none')
    else:
        fig, ax = plt.subplots(figsize=(18, 14))

    current_y = 0
    group_boundaries = [0]

    for group_idx, group in enumerate(scale_groups):
        scale = group['scale']

        for ltr in group['letters']:
            if ltr not in mat_filtered.index:
                current_y += FIXED_BAND_HEIGHT
                continue

            disease_data = mat_filtered.loc[ltr].values
            month_indices = mat_filtered.columns.values

            # Clip to scale (cap overflow)
            disease_data_clipped = np.clip(disease_data, 0, scale)
            scaled_data = (disease_data_clipped / scale) * FIXED_BAND_HEIGHT
            offset_data = scaled_data + current_y

            color = LETTER_INFO[ltr][1]
            # Transparent schemes use black lines; opaque uses colored lines
            line_color = 'black' if is_transparent else color

            ax.plot(month_indices, offset_data, color=line_color, linewidth=1.5)
            ax.fill_between(month_indices, current_y, offset_data, color=color, alpha=0.3)

            # Transparent schemes get baseline markers
            if is_transparent:
                ax.axhline(y=current_y, color='black', linestyle='-', linewidth=0.8)

            label_text = LETTER_INFO[ltr][0]
            ax.text(-0.005, current_y + 0.5 * FIXED_BAND_HEIGHT, label_text,
                    transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

            current_y += FIXED_BAND_HEIGHT

        group_boundaries.append(current_y)
        if group_idx < len(scale_groups) - 1:
            current_y += gap_height

    # Thick demarcation lines at group boundaries
    for boundary_y in group_boundaries:
        ax.axhline(y=boundary_y, color='black', linestyle='-', linewidth=2.5)

    ax.set_ylim(0, overall_y_max)
    ax.set_xlim(start_month, end_month)
    ax.set_yticks([])
    ax.set_xlabel('Month index')

    # Year labels and vertical lines
    start_year_plot = (start_month // 12) + START_YEAR_DATA
    end_year_plot = (end_month // 12) + START_YEAR_DATA

    tick_positions = [m for m in range(start_month, end_month + 1) if m % 12 == 0]
    ax.set_xticks(tick_positions)
    ax.grid(axis='x', linestyle='--', linewidth=0.5)

    for year in range(start_year_plot, end_year_plot + 1):
        month_idx_jan = (year - START_YEAR_DATA) * 12
        if start_month <= month_idx_jan <= end_month:
            ax.text(month_idx_jan, 1.01, str(year), transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=9)
            ax.axvline(x=month_idx_jan, color='black', linestyle='--', linewidth=0.5, alpha=0.7)

    # Label in top-left corner
    # W0: "ALL", "GE65", "LT65"
    # W1/W2/W2A: "ALLW1", "GE65W2", "LT65W2A", etc.
    scheme_name = {1: '', 2: 'W1', 4: 'W2', 5: 'W2A'}.get(scheme_num, f'S{scheme_num}')
    nn_suffix = '' if normalize else 'nn'
    label = f"{bucket_label}{scheme_name}{nn_suffix}"
    ax.text(0.01, 0.99, label, transform=ax.transAxes, fontsize=14, fontweight='bold',
            ha='left', va='top', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

    fig.tight_layout(rect=[0.12, 0.05, 1, 0.95], pad=1.0)

    # Filename (always include scheme name)
    scheme_name_file = {1: 'W0', 2: 'W1', 4: 'W2', 5: 'W2A'}.get(scheme_num, f'S{scheme_num}')
    nn_file_suffix = '' if normalize else '_nn'
    if start_month == 0 and end_month == 251:
        fname = f'area_popscaled_{scheme_name_file}_{bucket_label}{nn_file_suffix}.png'
    else:
        start_yr = (start_month // 12) + START_YEAR_DATA
        end_yr = (end_month // 12) + START_YEAR_DATA
        fname = f'area_popscaled_{start_yr}-{end_yr}_{scheme_name_file}_{bucket_label}{nn_file_suffix}.png'

    # Save with appropriate background
    if is_transparent:
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight', transparent=True)
    else:
        fig.savefig(out_dir / fname, dpi=150, bbox_inches='tight', facecolor='white')

    plt.close(fig)
    print(f"[INFO] Saved {fname}", file=sys.stderr)


# ============================================================================
# EXCEL OUTPUT
# ============================================================================

def write_excel(all_mats, pop_df, scaling_factors, out_path, scheme_num, normalize=True):
    """Write scaled death data to Excel in the user's format."""

    n_months = 252
    month_labels = [f"{(m // 12) + START_YEAR_DATA}-{(m % 12) + 1:02d}" for m in range(n_months)]

    rows = []

    for bucket, bucket_label in [('ALL', 'ALL'), ('PLUS65', 'GE65'), ('UNDER65', 'LT65')]:
        mat = all_mats[bucket]

        # Section header row
        header_row = {'Label': f'=== {bucket_label} ==='}
        for ml in month_labels:
            header_row[ml] = np.nan
        rows.append(header_row)

        # Disease rows
        totals = np.zeros(n_months)

        for ltr in DISEASE_ORDER:
            row_data = {'Label': f"{ltr} ({DISEASE_NAMES_SHORT[ltr]})"}

            max_scaled = 0
            max_month_idx = 0

            for m_idx in range(n_months):
                ml = month_labels[m_idx]
                if ltr in mat.index and m_idx in mat.columns:
                    val = mat.loc[ltr, m_idx]
                else:
                    val = 0.0
                row_data[ml] = val
                totals[m_idx] += val

                if val > max_scaled:
                    max_scaled = val
                    max_month_idx = m_idx

            # Store max info in row
            year_of_max = (max_month_idx // 12) + START_YEAR_DATA
            factor = scaling_factors[bucket].get(year_of_max, 1.0)
            max_raw = max_scaled / factor if factor > 0 else max_scaled

            row_data['Row Max Sc'] = max_scaled
            row_data['Row Max'] = max_raw
            row_data['Year Max'] = month_labels[max_month_idx]

            rows.append(row_data)

        # TOTAL row
        total_row = {'Label': 'TOTAL'}
        max_total = 0
        max_total_idx = 0
        for m_idx in range(n_months):
            ml = month_labels[m_idx]
            total_row[ml] = totals[m_idx]
            if totals[m_idx] > max_total:
                max_total = totals[m_idx]
                max_total_idx = m_idx

        year_of_max = (max_total_idx // 12) + START_YEAR_DATA
        factor = scaling_factors[bucket].get(year_of_max, 1.0)
        total_row['Row Max Sc'] = max_total
        total_row['Row Max'] = max_total / factor if factor > 0 else max_total
        total_row['Year Max'] = month_labels[max_total_idx]
        rows.append(total_row)

        # Population row
        pop_row = {'Label': 'Population'}
        for m_idx in range(n_months):
            ml = month_labels[m_idx]
            year = (m_idx // 12) + START_YEAR_DATA
            pop_val = pop_df[(pop_df['Year'] == year) & (pop_df['bucket'] == bucket)]['pop'].values
            if len(pop_val) > 0:
                pop_row[ml] = int(pop_val[0])
            else:
                pop_row[ml] = np.nan
        # Add ref year population to summary columns
        ref_pop = pop_df[(pop_df['Year'] == REFERENCE_YEAR) & (pop_df['bucket'] == bucket)]['pop'].values
        pop_row['Row Max Sc'] = int(ref_pop[0]) if len(ref_pop) > 0 else np.nan
        pop_row['Row Max'] = int(ref_pop[0]) if len(ref_pop) > 0 else np.nan
        pop_row['Year Max'] = f'{REFERENCE_YEAR}-01'
        rows.append(pop_row)

        # pop/pop2023 row
        factor_row = {'Label': 'pop/pop2023'}
        max_factor = 0
        max_factor_month = ''
        for m_idx in range(n_months):
            ml = month_labels[m_idx]
            year = (m_idx // 12) + START_YEAR_DATA
            if year in scaling_factors[bucket]:
                f = scaling_factors[bucket][year]
                factor_row[ml] = f
                if f > max_factor:
                    max_factor = f
                    max_factor_month = ml
            else:
                factor_row[ml] = np.nan
        factor_row['Row Max Sc'] = max_factor
        factor_row['Row Max'] = max_factor
        factor_row['Year Max'] = max_factor_month
        rows.append(factor_row)

        # Empty row between sections
        empty_row = {'Label': ''}
        for ml in month_labels:
            empty_row[ml] = np.nan
        empty_row['Row Max Sc'] = np.nan
        empty_row['Row Max'] = np.nan
        empty_row['Year Max'] = np.nan
        rows.append(empty_row)

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Reorder columns: Label, months, then summary columns
    cols = ['Label'] + month_labels + ['Row Max Sc', 'Row Max', 'Year Max']
    df = df[cols]

    # Write to Excel
    scheme_name = {1: 'W0', 2: 'W1', 4: 'W2', 5: 'W2A'}.get(scheme_num, f'S{scheme_num}')
    df.to_excel(out_path, index=False, sheet_name=f'{scheme_name}_PopScaled')
    print(f"[INFO] Saved {out_path}", file=sys.stderr)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Generate population-scaled mortality plots for all weighting schemes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weighting Schemes:
  W0 (Scheme 1):  UCOD only - opaque white background (base layer)
  W1 (Scheme 2):  Dobson - transparent (overlay)
  W2 (Scheme 4):  Equal/unique - transparent (overlay)
  W2A (Scheme 5): Equal/all positions - transparent (overlay)

Examples:
  # Single scheme
  python mortality_plot_popscaled_v6.py --agg_csv data.csv --pop_txt pop.txt --out_dir out --scheme W0
  
  # All schemes
  python mortality_plot_popscaled_v6.py --agg_csv data.csv --pop_txt pop.txt --out_dir out --all_schemes
        """)
    
    parser.add_argument('--agg_csv', required=True, help='Path to month_idx_agg.csv with schemes 1,2,4,5')
    parser.add_argument('--pop_txt', required=True, help='Path to USA_Population5.txt')
    parser.add_argument('--out_dir', required=True, help='Output directory')
    parser.add_argument('--scheme', default='W0', choices=['W0', 'W1', 'W2', 'W2A'],
                        help='Scheme to process (default: W0)')
    parser.add_argument('--start_year', type=int, default=2003, help='Start year for plots')
    parser.add_argument('--end_year', type=int, default=2023, help='End year for plots')
    parser.add_argument('--no_excel', action='store_true', help='Skip Excel output')
    parser.add_argument('--no_normalize', action='store_true', help='Skip population normalization (use raw counts)')
    parser.add_argument('--all_schemes', action='store_true', help='Process all 4 schemes (W0, W1, W2, W2A)')

    args = parser.parse_args()

    agg_path = Path(args.agg_csv)
    pop_path = Path(args.pop_txt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load population data and compute scaling factors
    pop_df = load_population_data(pop_path)
    scaling_factors = compute_scaling_factors(pop_df)

    print("[INFO] Scaling factors (pop_2023 / pop_year):", file=sys.stderr)
    for bucket in ['ALL', 'PLUS65', 'UNDER65']:
        print(f"  {bucket}: 2003={scaling_factors[bucket][2003]:.4f}, "
              f"2023={scaling_factors[bucket][2023]:.4f}", file=sys.stderr)

    # Determine normalization
    normalize = not args.no_normalize
    if normalize:
        print("[INFO] Population normalization: ENABLED", file=sys.stderr)
    else:
        print("[INFO] Population normalization: DISABLED (raw counts)", file=sys.stderr)

    # Determine which schemes to process
    if args.all_schemes:
        schemes_to_process = ['W0', 'W1', 'W2', 'W2A']
    else:
        schemes_to_process = [args.scheme]

    # Calculate month indices
    start_month = (args.start_year - START_YEAR_DATA) * 12
    end_month = (args.end_year - START_YEAR_DATA) * 12 + 11

    for scheme_name in schemes_to_process:
        scheme_num = SCHEME_MAP[scheme_name]
        print(f"\n[INFO] Processing {scheme_name} (Scheme {scheme_num})", file=sys.stderr)

        death_df = load_death_data(agg_path, scheme_num)
        
        if len(death_df) == 0:
            print(f"[WARN] No data found for scheme {scheme_num}, skipping", file=sys.stderr)
            continue

        # Prepare scaled matrices for all buckets
        all_mats = {}
        for bucket in ['ALL', 'PLUS65', 'UNDER65']:
            all_mats[bucket] = prepare_monthly_matrix(death_df, bucket, scaling_factors, normalize=normalize)

        # Check for overflow before drawing
        print(f"\n[INFO] Checking for overflow in {scheme_name}...", file=sys.stderr)
        any_overflow = False
        for bucket, label in [('ALL', 'ALL'), ('PLUS65', 'GE65'), ('UNDER65', 'LT65')]:
            if check_overflow(all_mats[bucket], label, scheme_num):
                any_overflow = True
        if not any_overflow:
            print("[INFO] No overflow detected.", file=sys.stderr)

        # Draw plots
        for bucket, label in [('ALL', 'ALL'), ('PLUS65', 'GE65'), ('UNDER65', 'LT65')]:
            if args.start_year == 2003 and args.end_year == 2023:
                draw_plot(all_mats[bucket], scheme_num, label, out_dir, 
                          start_month=0, end_month=251, normalize=normalize)
            else:
                draw_plot(all_mats[bucket], scheme_num, label, out_dir,
                          start_month=start_month, end_month=end_month, normalize=normalize)

        # Write Excel
        if not args.no_excel:
            nn_suffix = '' if normalize else '_nn'
            excel_path = out_dir / f'monthly_deaths_popscaled_{scheme_name}{nn_suffix}.xlsx'
            write_excel(all_mats, pop_df, scaling_factors, excel_path, scheme_num, normalize=normalize)

    print("\n[OK] Done!", file=sys.stderr)


if __name__ == '__main__':
    main()
