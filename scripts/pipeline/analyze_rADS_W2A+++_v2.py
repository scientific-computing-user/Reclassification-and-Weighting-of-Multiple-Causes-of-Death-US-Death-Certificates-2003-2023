#!/usr/bin/env python3
"""
Analyze rADS (all disease letters including repeats) with W2A weighting.

W2A scheme: Each letter gets equal weight 1/Ls where Ls = length of string.
For rADS, duplicates are counted (e.g., "AAAAAPBRRNN" has Ls=11).

Also calculates for rUDS (unique letters) for comparison.

Output: CSV with columns Letter,First,Not_First,Only,W1,W2A,Anywhere,Mean_Ls

Usage:
    python analyze_rADS_W2A.py <filename> --years 2020-2023
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import sys
import os
import argparse

# Allowed letters in defined order
ALLOWED_LETTERS = ["B", "C", "N", "R", "E", "D", "V", "P", "T", "S", "A", "X", "F", "H", "U"]
ALLOWED_SET = set(ALLOWED_LETTERS)

AGE_GROUPS = ["ALL", "LT65", "GE65"]

def parse_arguments():
    parser = argparse.ArgumentParser(description='Analyze rADS with W2A weighting')
    parser.add_argument('filename', help='Input CSV file')
    parser.add_argument('--years', '-y', type=str, default=None,
                        help='Year range (e.g., 2020-2023)')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output filename prefix')
    return parser.parse_args()


def parse_year_range(year_str):
    if year_str is None:
        return None, None
    parts = year_str.split('-')
    if len(parts) != 2:
        print(f"ERROR: Invalid year range: {year_str}")
        sys.exit(1)
    return int(parts[0]), int(parts[1])


def calculate_w1_weights(uds_string, count):
    """W1: First letter gets 0.5, others share 0.5 equally."""
    Ls = len(uds_string)
    if Ls == 1:
        return {uds_string[0]: count}
    
    weights = {}
    for i, letter in enumerate(uds_string):
        if i == 0:
            w = count * 0.5
        else:
            w = count * 0.5 / (Ls - 1)
        weights[letter] = weights.get(letter, 0) + w
    return weights


def calculate_w2_weights(ads_string, count):
    """W2A: Each position gets equal weight 1/Ls (duplicates counted)."""
    Ls = len(ads_string)
    if Ls == 0:
        return {}
    
    weight_per_position = count / Ls
    weights = {}
    for letter in ads_string:
        weights[letter] = weights.get(letter, 0) + weight_per_position
    return weights


def analyze_uds_column(df, uds_column, count_column='Count'):
    """
    Analyze a UDS/ADS column.
    
    Returns dict with letter stats:
    - First: count where letter is first
    - Not_First: count where letter appears but not first
    - Only: count where letter is only letter (Ls=1)
    - W1: W1 weighted count
    - W2A: W2A weighted count
    - Anywhere: count where letter appears anywhere
    - Mean_Ls: mean string length when letter is first
    """
    letter_stats = {letter: {
        'First': 0,
        'Not_First': 0,
        'Only': 0,
        'W1': 0.0,
        'W2A': 0.0,
        'Anywhere': 0,
        'Ls_sum': 0,  # For computing mean
        'Ls_count': 0
    } for letter in ALLOWED_LETTERS}
    
    for _, row in df.iterrows():
        uds = row[uds_column]
        count = row[count_column]
        
        if not isinstance(uds, str) or len(uds) == 0:
            continue
        
        Ls = len(uds)
        first_letter = uds[0]
        
        # Track Ls distribution for first letter
        letter_stats[first_letter]['Ls_sum'] += Ls * count
        letter_stats[first_letter]['Ls_count'] += count
        
        # W1 weights (based on unique positions)
        w1_weights = calculate_w1_weights(uds, count)
        
        # W2A weights (each position equal)
        w2_weights = calculate_w2_weights(uds, count)
        
        # Count by position
        seen = set()
        for i, letter in enumerate(uds):
            if letter not in ALLOWED_SET:
                continue
            
            # W1 and W2A
            if letter in w1_weights:
                letter_stats[letter]['W1'] += w1_weights.get(letter, 0)
                # Only add once per record for W1
                w1_weights[letter] = 0
            
            letter_stats[letter]['W2A'] += w2_weights.get(letter, 0) / uds.count(letter)  # Distribute evenly
            
            if letter not in seen:
                # First occurrence of this letter in string
                if i == 0:
                    letter_stats[letter]['First'] += count
                else:
                    letter_stats[letter]['Not_First'] += count
                
                if Ls == 1:
                    letter_stats[letter]['Only'] += count
                
                letter_stats[letter]['Anywhere'] += count
                seen.add(letter)
    
    # Compute Mean_Ls
    for letter in ALLOWED_LETTERS:
        if letter_stats[letter]['Ls_count'] > 0:
            letter_stats[letter]['Mean_Ls'] = letter_stats[letter]['Ls_sum'] / letter_stats[letter]['Ls_count']
        else:
            letter_stats[letter]['Mean_Ls'] = 0.0
    
    return letter_stats


def analyze_ads_column(df, ads_column, count_column='Count'):
    """
    Analyze an ADS column (with repeats).
    
    W1 weighting: first letter gets 0.5, others share 0.5 equally (applied to ALL occurrences)
    W2A weighting: each letter gets 1/Ls for each occurrence
    Anywhere: counts ALL occurrences (not just unique per certificate)
    """
    letter_stats = {letter: {
        'First': 0,
        'Not_First': 0,
        'Only': 0,
        'W1': 0.0,
        'W2A': 0.0,
        'Anywhere': 0,
        'Ls_sum': 0,
        'Ls_count': 0
    } for letter in ALLOWED_LETTERS}
    
    for _, row in df.iterrows():
        ads = row[ads_column]
        count = row[count_column]
        
        if not isinstance(ads, str) or len(ads) == 0:
            continue
        
        # Fix: If first letter is duplicated in second position, remove first letter
        # e.g., "CCBN" -> "CBN", "BB" -> "B"
        if len(ads) >= 2 and ads[0] == ads[1]:
            ads = ads[1:]
        
        Ls = len(ads)
        first_letter = ads[0]
        
        # Track Ls distribution for first letter
        if first_letter in ALLOWED_SET:
            letter_stats[first_letter]['Ls_sum'] += Ls * count
            letter_stats[first_letter]['Ls_count'] += count
        
        # W2A weights: each position gets count/Ls
        w2_per_position = count / Ls
        
        # W1 weights: first position gets 0.5, others share 0.5 equally
        # Applied to ALL positions (including repeats)
        if Ls == 1:
            w1_per_first = count
            w1_per_other = 0
        else:
            w1_per_first = count * 0.5
            w1_per_other = count * 0.5 / (Ls - 1)
        
        # Process each position (count ALL occurrences)
        seen = set()
        for i, letter in enumerate(ads):
            if letter not in ALLOWED_SET:
                continue
            
            # W1: each position gets its weight
            if i == 0:
                letter_stats[letter]['W1'] += w1_per_first
            else:
                letter_stats[letter]['W1'] += w1_per_other
            
            # W2A: each occurrence gets w2_per_position
            letter_stats[letter]['W2A'] += w2_per_position
            
            # Anywhere: count ALL occurrences
            letter_stats[letter]['Anywhere'] += count
            
            # First/Not_First/Only: only count once per letter per record
            if letter not in seen:
                if i == 0:
                    letter_stats[letter]['First'] += count
                else:
                    letter_stats[letter]['Not_First'] += count
                
                if Ls == 1:
                    letter_stats[letter]['Only'] += count
                
                seen.add(letter)
    
    # Compute Mean_Ls
    for letter in ALLOWED_LETTERS:
        if letter_stats[letter]['Ls_count'] > 0:
            letter_stats[letter]['Mean_Ls'] = letter_stats[letter]['Ls_sum'] / letter_stats[letter]['Ls_count']
        else:
            letter_stats[letter]['Mean_Ls'] = 0.0
    
    return letter_stats


def stats_to_dataframe(letter_stats, total_count):
    """Convert letter_stats dict to DataFrame."""
    rows = []
    for letter in ALLOWED_LETTERS:
        stats = letter_stats[letter]
        rows.append({
            'Letter': letter,
            'First': stats['First'],
            'Not_First': stats['Not_First'],
            'Only': stats['Only'],
            'W1': round(stats['W1'], 1),
            'W2A': round(stats['W2A'], 1),
            'Anywhere': stats['Anywhere'],
            'Mean_Ls': round(stats['Mean_Ls'], 3)
        })
    
    # Add All row
    rows.append({
        'Letter': 'All',
        'First': sum(letter_stats[l]['First'] for l in ALLOWED_LETTERS),
        'Not_First': sum(letter_stats[l]['Not_First'] for l in ALLOWED_LETTERS),
        'Only': sum(letter_stats[l]['Only'] for l in ALLOWED_LETTERS),
        'W1': round(sum(letter_stats[l]['W1'] for l in ALLOWED_LETTERS), 1),
        'W2A': round(sum(letter_stats[l]['W2A'] for l in ALLOWED_LETTERS), 1),
        'Anywhere': sum(letter_stats[l]['Anywhere'] for l in ALLOWED_LETTERS),
        'Mean_Ls': ''
    })
    
    return pd.DataFrame(rows)


def main():
    args = parse_arguments()
    
    print("=" * 70)
    print("rADS W2A ANALYSIS (All Letters Including Repeats)")
    print("=" * 70)
    
    # Parse year range
    start_year, end_year = parse_year_range(args.years)
    
    # Read data
    print(f"\nReading: {args.filename}")
    
    # Check if file has header by looking at first line
    with open(args.filename, 'r') as f:
        first_line = f.readline().strip()
    
    # Expected columns
    expected_cols = ['Count', 'rADS', 'rUDS', 'year', 'age_group']
    
    if first_line.startswith('Count,') or 'rUDS' in first_line:
        # Has header
        df = pd.read_csv(args.filename, keep_default_na=False, na_values=[''])
    else:
        # No header - assume format: Count,rADS,rUDS,year,age_group
        df = pd.read_csv(args.filename, header=None, names=expected_cols,
                        keep_default_na=False, na_values=[''])
        print("  (No header detected, using default column names)")
    
    print(f"  Loaded {len(df):,} records")
    
    # Check required columns
    required = ['Count', 'rUDS', 'rADS', 'year', 'age_group']
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Filter by year
    if start_year is not None:
        df = df[(df['year'] >= start_year) & (df['year'] <= end_year)].copy()
        print(f"  Filtered to {start_year}-{end_year}: {len(df):,} records")
    
    total_deaths = df['Count'].sum()
    print(f"  Total deaths: {total_deaths:,}")
    
    # Output filename
    if args.output:
        output_prefix = args.output
    else:
        if start_year:
            output_prefix = f"rADS_W2A_analysis_{start_year}-{end_year}"
        else:
            output_prefix = "rADS_W2A_analysis_all_years"
    
    # Analyze by age group
    all_results = {}
    
    for age_group in AGE_GROUPS:
        if age_group == 'ALL':
            df_subset = df
        else:
            df_subset = df[df['age_group'] == age_group]
        
        if len(df_subset) == 0:
            print(f"\n[SKIP] {age_group}: No data")
            continue
        
        subset_deaths = df_subset['Count'].sum()
        print(f"\n--- {age_group} ({subset_deaths:,} deaths) ---")
        
        # Analyze rUDS (unique letters)
        print("  Analyzing rUDS (unique letters)...")
        rUDS_stats = analyze_ads_column(df_subset, 'rUDS')
        df_rUDS = stats_to_dataframe(rUDS_stats, subset_deaths)
        
        # Analyze rADS (all letters with repeats)
        print("  Analyzing rADS (all letters with repeats)...")
        rADS_stats = analyze_ads_column(df_subset, 'rADS')
        df_rADS = stats_to_dataframe(rADS_stats, subset_deaths)
        
        all_results[age_group] = {
            'rUDS': df_rUDS,
            'rADS': df_rADS,
            'deaths': subset_deaths
        }
        
        # Print summary
        print(f"\n  rUDS W2A total: {df_rUDS[df_rUDS['Letter']=='All']['W2A'].values[0]:,.1f}")
        print(f"  rADS W2A total: {df_rADS[df_rADS['Letter']=='All']['W2A'].values[0]:,.1f}")
    
    # Save results
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Create stacked output for each type (rUDS, rADS)
    for data_type in ['rUDS', 'rADS']:
        rows = []
        for age_group in AGE_GROUPS:
            if age_group not in all_results:
                continue
            
            df_ag = all_results[age_group][data_type]
            
            # Header row
            rows.append({'Letter': f'=== {age_group} ===', 'First': '', 'Not_First': '', 
                        'Only': '', 'W1': '', 'W2A': '', 'Anywhere': '', 'Mean_Ls': ''})
            
            # Data rows
            for _, row in df_ag.iterrows():
                rows.append(row.to_dict())
            
            # Blank separator
            rows.append({'Letter': '', 'First': '', 'Not_First': '', 
                        'Only': '', 'W1': '', 'W2A': '', 'Anywhere': '', 'Mean_Ls': ''})
        
        df_out = pd.DataFrame(rows)
        
        # Save CSV
        csv_file = f"{output_prefix}_{data_type}.csv"
        df_out.to_csv(csv_file, index=False)
        print(f"  Saved: {csv_file}")
    
    # Save Excel with both sheets
    xlsx_file = f"{output_prefix}.xlsx"
    with pd.ExcelWriter(xlsx_file, engine='openpyxl') as writer:
        for data_type in ['rUDS', 'rADS']:
            rows = []
            for age_group in AGE_GROUPS:
                if age_group not in all_results:
                    continue
                df_ag = all_results[age_group][data_type]
                rows.append({'Letter': f'=== {age_group} ==='})
                for _, row in df_ag.iterrows():
                    rows.append(row.to_dict())
                rows.append({'Letter': ''})
            
            df_out = pd.DataFrame(rows)
            df_out.to_excel(writer, sheet_name=data_type, index=False)
    
    print(f"  Saved: {xlsx_file}")
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
