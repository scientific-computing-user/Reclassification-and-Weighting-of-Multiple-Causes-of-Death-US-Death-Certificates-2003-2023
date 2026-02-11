#!/usr/bin/env python3
"""
compute_all_schemes.py
======================
Compute monthly aggregated death counts using four weighting schemes:

  Scheme 1 (W0):  UCOD only - 100% weight to first letter
  Scheme 2 (W1):  Dobson - 50% to first letter, 50% split among other POSITIONS
  Scheme 4 (W2):  Equal on UNIQUE letters - 1/Ls where Ls = len(rUDS)
  Scheme 5 (W2A): Equal on ALL positions - 1/Ls where Ls = len(rADS)

Key distinction:
  - W0, W1, W2 use rUDS (unique disease string) 
  - W2A uses rADS (all disease string with repeats)

Input CSV columns: Count, rADS, rUDS, year, month, age_group, [sex, race]

Usage:
    python compute_all_schemes.py input.csv -o output.csv
    python compute_all_schemes.py input.csv --years 2020-2023 --fix_rads
"""

import pandas as pd
import numpy as np
import argparse
import sys
from collections import defaultdict

# Disease letter order
ALLOWED_LETTERS = ['B', 'C', 'N', 'R', 'E', 'D', 'V', 'P', 'T', 'S', 'A', 'X', 'F', 'H', 'U']
ALLOWED_SET = set(ALLOWED_LETTERS)

START_YEAR = 2003  # For month_idx calculation

# Age group mapping
AGE_BUCKET_MAP = {
    '0': 'LT65', '1-4': 'LT65', '5-14': 'LT65', '15-24': 'LT65',
    '25-34': 'LT65', '35-44': 'LT65', '45-54': 'LT65', '55-64': 'LT65',
    '65-74': 'GE65', '75-84': 'GE65', '85+': 'GE65',
    'LT65': 'LT65', 'GE65': 'GE65', 'UNDER65': 'LT65', 'PLUS65': 'GE65'
}


def clean_string(s, allowed_set):
    """Keep only allowed letters from string."""
    if not isinstance(s, str):
        return ''
    return ''.join(c for c in s.upper() if c in allowed_set)


def fix_rads_duplicate(ads):
    """
    Fix rADS where first letter is duplicated in second position.
    e.g., "CCBN" -> "CBN", "BB" -> "B", "NNN" -> "NN"
    
    This corrects an upstream data issue where the first letter
    sometimes appears twice at the start.
    """
    if len(ads) >= 2 and ads[0] == ads[1]:
        return ads[1:]
    return ads


def compute_w0(uds, count):
    """
    W0 / Scheme 1: UCOD only
    - First letter gets 100% of count
    - All other letters get 0
    """
    if not uds:
        return {}
    return {uds[0]: count}


def compute_w1(uds, count):
    """
    W1 / Scheme 2: Dobson hierarchical weighting
    - If Ls=1: first letter gets 100%
    - If Ls>1: first letter gets 50%, other POSITIONS share 50%
    
    Applied to rUDS (unique letters only).
    Each position after first gets 0.5/(Ls-1).
    """
    if not uds:
        return {}
    
    Ls = len(uds)
    weights = {}
    
    if Ls == 1:
        weights[uds[0]] = count
    else:
        # First letter gets 50%
        weights[uds[0]] = 0.5 * count
        # Each subsequent position gets equal share of remaining 50%
        w_other = 0.5 * count / (Ls - 1)
        for letter in uds[1:]:
            weights[letter] = weights.get(letter, 0) + w_other
    
    return weights


def compute_w2(uds, count):
    """
    W2 / Scheme 4: Equal weight on UNIQUE letters
    - Each unique letter gets count/Ls where Ls = len(rUDS)
    
    Applied to rUDS (unique letters only).
    """
    if not uds:
        return {}
    
    Ls = len(uds)
    w = count / Ls
    
    weights = {}
    for letter in uds:
        weights[letter] = weights.get(letter, 0) + w
    
    return weights


def compute_w2a(ads, count):
    """
    W2A / Scheme 5: Equal weight on ALL positions
    - Each position gets count/Ls where Ls = len(rADS)
    
    Applied to rADS (all letters including repeats).
    e.g., "BNNRRR" has Ls=6, each position gets count/6
    """
    if not ads:
        return {}
    
    Ls = len(ads)
    w = count / Ls
    
    weights = {}
    for letter in ads:
        weights[letter] = weights.get(letter, 0) + w
    
    return weights


def get_bucket(age_group):
    """Map age_group to bucket (ALL, LT65, GE65)."""
    if age_group in AGE_BUCKET_MAP:
        return AGE_BUCKET_MAP[age_group]
    return 'UNKNOWN'


def process_dataframe(df, fix_rads=False, start_year=2003):
    """
    Process entire DataFrame and compute all 4 weighting schemes.
    
    Returns list of dicts with: scheme, bucket, month_idx, letter, weight
    """
    results = []
    
    for idx, row in df.iterrows():
        count = row['Count']
        if count <= 0:
            continue
        
        # Get strings
        rUDS = clean_string(row.get('rUDS', ''), ALLOWED_SET)
        rADS = clean_string(row.get('rADS', ''), ALLOWED_SET)
        
        # Optional fix for rADS duplicate first letter
        if fix_rads:
            rADS = fix_rads_duplicate(rADS)
        
        # Skip if no data
        if not rUDS and not rADS:
            continue
        
        # Use rUDS for W0, W1, W2; use rADS for W2A
        # If rUDS missing, derive from rADS
        if not rUDS and rADS:
            seen = []
            for c in rADS:
                if c not in seen:
                    seen.append(c)
            rUDS = ''.join(seen)
        
        # If rADS missing, use rUDS
        if not rADS:
            rADS = rUDS
        
        # Get time and demographics
        year = int(row['year'])
        month = int(row['month'])
        month_idx = (year - start_year) * 12 + (month - 1)
        
        age_group = str(row.get('age_group', 'ALL'))
        bucket = get_bucket(age_group)
        
        sex = str(row.get('sex', 'any')).lower() or 'any'
        race = str(row.get('race', 'any')).lower() or 'any'
        
        # Compute all schemes
        w0 = compute_w0(rUDS, count)
        w1 = compute_w1(rUDS, count)
        w2 = compute_w2(rUDS, count)
        w2a = compute_w2a(rADS, count)
        
        # Add to results
        for letter, weight in w0.items():
            results.append({
                'scheme': 1, 'bucket': bucket, 'sex': sex, 'race': race,
                'month_idx': month_idx, 'letter': letter, 'weight': weight,
                'age_group': age_group
            })
        
        for letter, weight in w1.items():
            results.append({
                'scheme': 2, 'bucket': bucket, 'sex': sex, 'race': race,
                'month_idx': month_idx, 'letter': letter, 'weight': weight,
                'age_group': age_group
            })
        
        for letter, weight in w2.items():
            results.append({
                'scheme': 4, 'bucket': bucket, 'sex': sex, 'race': race,
                'month_idx': month_idx, 'letter': letter, 'weight': weight,
                'age_group': age_group
            })
        
        for letter, weight in w2a.items():
            results.append({
                'scheme': 5, 'bucket': bucket, 'sex': sex, 'race': race,
                'month_idx': month_idx, 'letter': letter, 'weight': weight,
                'age_group': age_group
            })
        
        if (idx + 1) % 100000 == 0:
            print(f"  Processed {idx + 1:,} records...", file=sys.stderr)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Compute all 4 weighting schemes (W0, W1, W2, W2A)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Weighting Schemes:
  Scheme 1 (W0):  UCOD only - 100% to first letter
  Scheme 2 (W1):  Dobson - 50% to first, 50% split among other positions
  Scheme 4 (W2):  Equal on unique letters - 1/len(rUDS) each
  Scheme 5 (W2A): Equal on all positions - 1/len(rADS) each

Example:
  python compute_all_schemes.py deaths.csv -o monthly_agg.csv --years 2020-2023
        """)
    
    parser.add_argument('input', help='Input CSV file (use - for stdin)')
    parser.add_argument('-o', '--output', default='month_idx_agg_all_schemes.csv',
                        help='Output CSV file')
    parser.add_argument('--years', '-y', type=str, default=None,
                        help='Year range filter (e.g., 2020-2023)')
    parser.add_argument('--start_year', type=int, default=2003,
                        help='Start year for month_idx calculation (default: 2003)')
    parser.add_argument('--no_fix_rads', action='store_true',
                        help='Disable rADS duplicate first letter fix (default: fix IS applied)')
    parser.add_argument('--by_age_group', action='store_true',
                        help='Keep detailed age_group instead of bucket (LT65/GE65)')
    
    args = parser.parse_args()
    
    print("=" * 70, file=sys.stderr)
    print("COMPUTING ALL WEIGHTING SCHEMES (W0, W1, W2, W2A)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    # Read input
    print(f"\nReading: {args.input}", file=sys.stderr)
    if args.input == '-':
        df = pd.read_csv(sys.stdin, keep_default_na=False, na_values=[''])
    else:
        df = pd.read_csv(args.input, keep_default_na=False, na_values=[''])
    
    print(f"  Loaded {len(df):,} records", file=sys.stderr)
    print(f"  Columns: {list(df.columns)}", file=sys.stderr)
    
    # Check required columns
    required = ['Count', 'year', 'month']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)
    
    # Check for UDS/ADS columns
    has_ruds = 'rUDS' in df.columns
    has_rads = 'rADS' in df.columns
    if not has_ruds and not has_rads:
        print("ERROR: Need at least one of rUDS or rADS columns", file=sys.stderr)
        sys.exit(1)
    
    print(f"  Has rUDS: {has_ruds}, Has rADS: {has_rads}", file=sys.stderr)
    
    # Filter by year if specified
    if args.years:
        parts = args.years.split('-')
        if len(parts) == 2:
            start_yr, end_yr = int(parts[0]), int(parts[1])
            df = df[(df['year'] >= start_yr) & (df['year'] <= end_yr)].copy()
            print(f"  Filtered to {start_yr}-{end_yr}: {len(df):,} records", file=sys.stderr)
    
    total_deaths = df['Count'].sum()
    print(f"  Total deaths: {total_deaths:,}", file=sys.stderr)
    
    fix_rads = not args.no_fix_rads
    if fix_rads:
        print("  [DEFAULT] Fixing rADS duplicate first letter (use --no_fix_rads to disable)", file=sys.stderr)
    else:
        print("  [DISABLED] rADS duplicate first letter fix", file=sys.stderr)
    
    # Process data
    print("\nProcessing...", file=sys.stderr)
    results = process_dataframe(df, fix_rads=fix_rads, start_year=args.start_year)
    
    print(f"\nGenerated {len(results):,} weighted entries", file=sys.stderr)
    
    # Convert to DataFrame
    result_df = pd.DataFrame(results)
    
    # Aggregate
    if args.by_age_group:
        group_cols = ['scheme', 'age_group', 'sex', 'race', 'month_idx', 'letter']
    else:
        group_cols = ['scheme', 'bucket', 'sex', 'race', 'month_idx', 'letter']
    
    agg_df = result_df.groupby(group_cols, as_index=False)['weight'].sum()
    
    print(f"Aggregated to {len(agg_df):,} rows", file=sys.stderr)
    
    # Save
    agg_df.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output}", file=sys.stderr)
    
    # Print summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("SUMMARY BY SCHEME", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    scheme_names = {
        1: 'W0 (UCOD only)',
        2: 'W1 (Dobson)',
        4: 'W2 (Equal/unique)',
        5: 'W2A (Equal/all positions)'
    }
    
    for scheme in [1, 2, 4, 5]:
        scheme_data = agg_df[agg_df['scheme'] == scheme]
        total = scheme_data['weight'].sum()
        print(f"  Scheme {scheme} ({scheme_names[scheme]}): {total:,.0f}", file=sys.stderr)
    
    print("\n" + "=" * 70, file=sys.stderr)
    print("SCHEME COMPARISON (First letter weight for 2-letter string)", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("""
  | Scheme | Name              | First Letter | Other Letters | Total |
  |--------|-------------------|--------------|---------------|-------|
  |   1    | W0 (UCOD)         | 100%         | 0%            | 100%  |
  |   2    | W1 (Dobson)       | 50%          | 50%           | 100%  |
  |   4    | W2 (Equal/unique) | 50%          | 50%           | 100%  |
  |   5    | W2A (Equal/all)   | 50%          | 50%           | 100%  |
  
  Note: W1 and W2 differ when Ls >= 3:
    Ls=3: W1 = [50%, 25%, 25%], W2 = [33%, 33%, 33%]
    Ls=4: W1 = [50%, 17%, 17%, 17%], W2 = [25%, 25%, 25%, 25%]
  
  W2A differs from W2 when rADS has repeats:
    rUDS="BNR" (Ls=3): W2 = [33%, 33%, 33%]
    rADS="BNNRRR" (Ls=6): W2A = B=17%, N=33%, R=50%
""", file=sys.stderr)
    
    print("DONE", file=sys.stderr)


if __name__ == '__main__':
    main()
