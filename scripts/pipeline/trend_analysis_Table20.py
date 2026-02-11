#!/usr/bin/env python3
"""
Table 20: Linear Trend Analysis of Monthly Deaths by Disease Category

Analyzes population-scaled monthly death trends across 4 weighting schemes:
  W0 (Scheme 1): UCOD only - first letter gets 100%
  W1 (Scheme 2): Dobson - first letter 50%, others share 50%
  W2 (Scheme 4): Equal/unique - each unique letter gets 1/Ls
  W2A (Scheme 5): Equal/all - each position gets 1/Ls (repeats count)

Input files required:
  1. month_idx_agg_all_schemes.csv - Aggregated death data by month/scheme/bucket/letter
  2. USA_Population5.txt - Population data for normalization (optional)

Output:
  Table20_trend_analysis.xlsx - Excel file with trend analysis results

Usage:
  python trend_analysis_Table20.py month_idx_agg_all_schemes.csv [population_file.txt]

Example:
  python trend_analysis_Table20.py month_idx_agg_all_schemes.csv USA_Population5.txt
"""

import pandas as pd
import numpy as np
from scipy import stats
import sys
import os

# ============================================================================
# CONFIGURATION
# ============================================================================

START_YEAR_DATA = 2003
REFERENCE_YEAR = 2023  # Year to scale population to

DISEASE_NAMES = {
    'A': 'Alcohol-related', 'B': 'Circulatory', 'C': 'Cancer', 'D': 'Digestive',
    'E': 'Endocrine', 'F': 'Falls', 'H': 'Homicide', 'N': 'Other natural',
    'P': 'Drug poisoning', 'R': 'Respiratory', 'S': 'Suicide', 'T': 'Transport',
    'U': 'Unknown', 'V': 'COVID-19', 'X': 'Other external'
}

DISEASE_ORDER = ['V', 'B', 'C', 'N', 'R', 'E', 'D', 'P', 'X', 'T', 'S', 'A', 'F', 'H']

SCHEME_MAP = {1: 'W0', 2: 'W1', 4: 'W2', 5: 'W2A'}
SCHEME_NAMES = {
    'W0': 'UCOD (first=100%)',
    'W1': 'Dobson (first=50%)',
    'W2': 'Equal/unique',
    'W2A': 'Equal/all'
}

# ============================================================================
# FUNCTIONS
# ============================================================================

def load_population(pop_path):
    """Load and parse population data from HMD-style file."""
    pop_raw = pd.read_csv(pop_path, skiprows=3, sep=r'\s+', engine='python',
                          header=None, names=['Year','Age','Pop_F','Pop_M','Pop_T'], dtype=str)
    pop_raw['Year'] = pop_raw['Year'].str.extract(r'(\d{4})').astype(int)
    pop_raw['Pop_T'] = pd.to_numeric(pop_raw['Pop_T'], errors='coerce')
    return pop_raw


def get_pop_by_bucket(pop_raw):
    """Compute population by age bucket (ALL, GE65, LT65) and year."""
    results = {}
    for bucket in ['ALL', 'GE65', 'LT65']:
        results[bucket] = {}
        for year in pop_raw['Year'].unique():
            yr_data = pop_raw[pop_raw['Year'] == year]
            if bucket == 'ALL':
                results[bucket][year] = yr_data['Pop_T'].sum()
            else:
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
                if bucket == 'GE65':
                    results[bucket][year] = plus65_pop
                else:
                    results[bucket][year] = under65_pop
    return results


def compute_scaling_factors(pop_by_bucket, ref_year):
    """Compute population scaling factors relative to reference year."""
    scaling_factors = {}
    for bucket in ['ALL', 'GE65', 'LT65']:
        ref_pop = pop_by_bucket[bucket][ref_year]
        scaling_factors[bucket] = {}
        for year in pop_by_bucket[bucket]:
            scaling_factors[bucket][year] = ref_pop / pop_by_bucket[bucket][year]
    return scaling_factors


def fit_linear(x, y):
    """Fit linear regression and return intercept, slope, R2."""
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return intercept, slope, r_value**2


def analyze_trends(df, scaling_factors=None):
    """
    Analyze linear trends for all disease letters across schemes and buckets.
    
    Parameters:
        df: DataFrame with columns [scheme, bucket, month_idx, letter, weight]
        scaling_factors: Optional dict of population scaling factors
    
    Returns:
        DataFrame with trend analysis results
    """
    rows = []
    
    for bucket_label in ['ALL', 'GE65', 'LT65']:
        for ltr in DISEASE_ORDER + ['U']:
            for normed in ['No', 'Yes']:
                for scheme_num, scheme_name in SCHEME_MAP.items():
                    
                    # Filter data
                    mask = (df['scheme'] == scheme_num) & (df['bucket'] == bucket_label) & (df['letter'] == ltr)
                    scheme_df = df[mask]
                    
                    if len(scheme_df) == 0:
                        continue
                    
                    # Aggregate by month_idx
                    pivot = scheme_df.groupby('month_idx')['weight'].sum()
                    
                    if len(pivot) < 12:
                        continue
                    
                    x = pivot.index.values
                    y = pivot.values.copy().astype(float)
                    
                    # Apply population normalization if requested and available
                    if normed == 'Yes' and scaling_factors is not None:
                        for i, m_idx in enumerate(pivot.index):
                            year = (m_idx // 12) + START_YEAR_DATA
                            if bucket_label in scaling_factors and year in scaling_factors[bucket_label]:
                                y[i] = y[i] * scaling_factors[bucket_label][year]
                    
                    # Skip normalization if no scaling factors
                    if normed == 'Yes' and scaling_factors is None:
                        continue
                    
                    # Fit linear trend
                    A, B, R2 = fit_linear(x, y)
                    mean_val = np.mean(y)
                    
                    # Calculate derived values
                    A_pct = round((A / mean_val) * 100) if mean_val > 0 else 0
                    B_year = B * 12  # Slope per year
                    B_year_pct = (B_year / mean_val) * 100 if mean_val > 0 else 0
                    
                    rows.append({
                        'Bucket': bucket_label,
                        'Letter': ltr,
                        'Disease': DISEASE_NAMES.get(ltr, ltr),
                        'Normed': normed,
                        'Scheme': scheme_name,
                        'A': round(A),
                        'B': round(B, 1),
                        'B/year': round(B_year),
                        'R2': round(R2, 2),
                        'Mean': round(mean_val),
                        'A%': A_pct,
                        'B/year%': round(B_year_pct, 2)
                    })
    
    return pd.DataFrame(rows)


def print_summary(results_df, normed='Yes'):
    """Print formatted summary table."""
    print("\n" + "="*115)
    print("TABLE 20: LINEAR TREND ANALYSIS - ALL 4 WEIGHTING SCHEMES (2003-2023)")
    if normed == 'Yes':
        print("Population-Normalized Monthly Deaths")
    else:
        print("Raw Monthly Deaths (not population-normalized)")
    print("="*115)
    print("\nB/year% = annual percent change (positive = increasing, negative = decreasing)")
    print("R2 = goodness of fit (higher = more linear trend)")
    
    for bucket in ['ALL', 'GE65', 'LT65']:
        print(f"\n{'='*115}")
        print(f"  {bucket}")
        print(f"{'='*115}")
        print(f"{'DL':<3} {'Disease':<18} | {'W0 B/yr%':>9} {'R2':>5} | {'W1 B/yr%':>9} {'R2':>5} | {'W2 B/yr%':>9} {'R2':>5} | {'W2A B/yr%':>9} {'R2':>5}")
        print("-"*115)
        
        for ltr in DISEASE_ORDER:
            disease = DISEASE_NAMES.get(ltr, ltr)[:18]
            row_str = f"{ltr:<3} {disease:<18} |"
            
            for scheme in ['W0', 'W1', 'W2', 'W2A']:
                mask = (results_df['Scheme']==scheme) & (results_df['Bucket']==bucket) & \
                       (results_df['Letter']==ltr) & (results_df['Normed']==normed)
                r = results_df[mask]
                if len(r) > 0:
                    r = r.iloc[0]
                    row_str += f" {r['B/year%']:>+8.2f}% {r['R2']:>5.2f} |"
                else:
                    row_str += f" {'N/A':>9} {'':>5} |"
            
            print(row_str)


def save_to_excel(results_df, output_path):
    """Save results to Excel with multiple sheets."""
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Full results
        results_df.to_excel(writer, sheet_name='All_Results', index=False)
        
        # Wide format comparison for each bucket (normalized)
        for bucket in ['ALL', 'GE65', 'LT65']:
            wide_rows = []
            for ltr in DISEASE_ORDER:
                row = {'Letter': ltr, 'Disease': DISEASE_NAMES.get(ltr, ltr)}
                for scheme in ['W0', 'W1', 'W2', 'W2A']:
                    mask = (results_df['Scheme']==scheme) & (results_df['Bucket']==bucket) & \
                           (results_df['Letter']==ltr) & (results_df['Normed']=='Yes')
                    r = results_df[mask]
                    if len(r) > 0:
                        r = r.iloc[0]
                        row[f'{scheme}_A'] = r['A']
                        row[f'{scheme}_B/yr'] = r['B/year']
                        row[f'{scheme}_B/yr%'] = r['B/year%']
                        row[f'{scheme}_R2'] = r['R2']
                        row[f'{scheme}_Mean'] = r['Mean']
                wide_rows.append(row)
            
            if wide_rows:
                pd.DataFrame(wide_rows).to_excel(writer, sheet_name=f'{bucket}_Normed', index=False)
        
        # Strong trends summary
        strong = results_df[(abs(results_df['B/year%']) > 3) & (results_df['R2'] > 0.5) & 
                           (results_df['Normed']=='Yes')].copy()
        if len(strong) > 0:
            strong = strong.sort_values('B/year%', ascending=False)
            strong.to_excel(writer, sheet_name='Strong_Trends', index=False)


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    data_path = sys.argv[1]
    pop_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Output path
    output_path = 'Table20_trend_analysis.xlsx'
    
    print("="*70)
    print("TABLE 20: LINEAR TREND ANALYSIS")
    print("="*70)
    
    # Load death data
    print(f"\nLoading data: {data_path}")
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Letters: {sorted(df['letter'].unique())}")
    print(f"  Buckets: {list(df['bucket'].unique())}")
    print(f"  Schemes: {sorted(df['scheme'].unique())}")
    
    # Create ALL bucket by summing GE65 + LT65
    if 'ALL' not in df['bucket'].unique():
        print("\nCreating ALL bucket from GE65 + LT65...")
        df_ge65 = df[df['bucket'] == 'GE65'].copy()
        df_lt65 = df[df['bucket'] == 'LT65'].copy()
        df_all = pd.concat([df_ge65, df_lt65]).groupby(['scheme', 'month_idx', 'letter'])['weight'].sum().reset_index()
        df_all['bucket'] = 'ALL'
        df = pd.concat([df, df_all], ignore_index=True)
        print(f"  Now have {len(df):,} rows")
    
    # Load population data if provided
    scaling_factors = None
    if pop_path and os.path.exists(pop_path):
        print(f"\nLoading population: {pop_path}")
        pop_raw = load_population(pop_path)
        pop_by_bucket = get_pop_by_bucket(pop_raw)
        scaling_factors = compute_scaling_factors(pop_by_bucket, REFERENCE_YEAR)
        print(f"  Population scaling factors computed (ref year: {REFERENCE_YEAR})")
    else:
        print("\nNo population file provided - skipping normalization")
    
    # Analyze trends
    print("\nAnalyzing trends...")
    results_df = analyze_trends(df, scaling_factors)
    print(f"  Generated {len(results_df)} trend results")
    
    # Print summary
    if scaling_factors:
        print_summary(results_df, normed='Yes')
    else:
        print_summary(results_df, normed='No')
    
    # Save to Excel
    print(f"\n\nSaving to: {output_path}")
    save_to_excel(results_df, output_path)
    print("Done!")


if __name__ == "__main__":
    main()
