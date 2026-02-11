#!/usr/bin/env python3
"""
Death Certificate UDS (Unique Disease String) Analysis Script - Version 14
DEFINITIVE VERSION with Dobson Data-Driven Weighting (W3)

This script analyzes rUDS (Record Axis) and eUDS (Entity Axis) strings from 
US Death Certificates, counting letter frequencies under various conditions
and weighting schemes.

The first letter in each string represents the Underlying Cause of Death (UCD).
Subsequent letters represent contributing causes.

TERMINOLOGY:
- Entity Axis (eUDS): Physician's original coding on the death certificate
- Record Axis (rUDS): NCHS/ACME automated system's final determination

VERSION 14 FEATURES:
- All v13 features preserved
- NEW: W3 Dobson data-driven weighting scheme
  Based on: Dobson et al. BMC Medical Research Methodology (2023) 23:83
  "A new data driven method for summarising multiple cause of death data"
  
  W3 computes weights based on observed co-occurrence patterns:
  - x_uc = N_c|u / N_u (proportion of deaths with UCoD=u that have CCoD=c)
  - For each death with UCoD=u and n contributing causes:
    * CCoD weight: w_c = x_uc / n
    * UCoD weight: w_u = 1 - sum(w_c for all CCoDs)
  - This method attributes more weight to contributing causes that
    commonly co-occur with the underlying cause.

- NEW: x_uc matrix can be INPUT from CSV file (--xmatrix option)
  or computed from the input data (default)
- NEW: x_uc matrix output tables (rUDS_x_uc_matrix, eUDS_x_uc_matrix)

INPUT CSV FORMAT:
    Count,rUDS,eUDS,year,age_group
    342034,B,B,2003,GE65
    567,C,C,2021,LT65

x_uc MATRIX CSV FORMAT (for --xmatrix option):
    UCoD,B,C,N,R,E,D,V,P,T,S,A,X,F,H,U
    B,0,0.034,0.335,0.192,...
    C,0.291,0,0.279,0.200,...
    ...

USAGE:
    python analyze_uds_v14.py <filename> --years 2020-2023
    python analyze_uds_v14.py <filename> --years 2020-2023 --xmatrix x_uc_rUDS_ALL.csv
    python analyze_uds_v14.py <filename>  # Uses all years, computes x_uc from data

Author: Claude & Michael Levitt
Date: 2024-2025
"""

import pandas as pd
import numpy as np
from collections import defaultdict, Counter
import sys
import os
import argparse

# ============================================================================
# CONFIGURATION
# ============================================================================

# Allowed letters in defined order (disease categories)
ALLOWED_LETTERS = ["B", "C", "N", "R", "E", "D", "V", "P", "T", "S", "A", "X", "F", "H", "U"]
ALLOWED_SET = set(ALLOWED_LETTERS)

# Age groups to analyze (in output order)
AGE_GROUPS = ["ALL", "LT65", "GE65"]

# Floating point tolerance for validation checks
EPSILON = 0.1

# Column name mapping (handles variations in input files)
YEAR_COLUMNS = ['year', 'Year', 'YEAR']
AGE_COLUMNS = ['age_group', 'AgeGroup', 'agegroup', 'AGE_GROUP']

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Analyze Death Certificate UDS strings with year and age group filtering.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python analyze_uds_v14.py data.csv --years 2020-2023
    python analyze_uds_v14.py data.csv --years 2020-2023 --xmatrix x_uc_matrix.csv
    python analyze_uds_v14.py data.csv  # Uses all years, computes x_uc from data
        """
    )
    
    parser.add_argument('filename', nargs='?', default=None,
                        help='Input CSV file (or use stdin)')
    
    parser.add_argument('--years', '-y', type=str, default=None,
                        help='Year range in format START-END (e.g., 2020-2023)')
    
    parser.add_argument('--xmatrix', '-x', type=str, default=None,
                        help='External x_uc matrix CSV file (if not provided, computed from data)')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed diagnostics')
    
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output Excel filename (default: auto-generated based on year range)')
    
    return parser.parse_args()


def get_default_output_filename(year_range):
    """Generate default output filename based on year range."""
    start_year, end_year = year_range
    
    if start_year is not None:
        if start_year == end_year:
            return f'uds_analysis_results_v14_{start_year}.xlsx'
        else:
            return f'uds_analysis_results_v14_{start_year}-{end_year}.xlsx'
    else:
        return 'uds_analysis_results_v14_all_years.xlsx'


def parse_year_range(year_str):
    """Parse year range string into start and end years."""
    if year_str is None:
        return None, None
    
    try:
        parts = year_str.split('-')
        if len(parts) != 2:
            raise ValueError(f"Invalid year range format: {year_str}. Use START-END (e.g., 2020-2023)")
        
        start_year = int(parts[0])
        end_year = int(parts[1])
        
        if start_year > end_year:
            start_year, end_year = end_year, start_year
            print(f"[NOTE] Swapped year range to {start_year}-{end_year}")
        
        return start_year, end_year
    
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)


def find_column(df, possible_names):
    """Find a column by checking multiple possible names."""
    for name in possible_names:
        if name in df.columns:
            return name
    return None


# ============================================================================
# x_uc MATRIX I/O
# ============================================================================

def load_x_matrix_from_csv(filename):
    """
    Load x_uc matrix from a CSV file.
    
    Expected CSV format:
        UCoD,B,C,N,R,E,D,V,P,T,S,A,X,F,H,U
        B,0,0.034,0.335,...
        C,0.291,0,0.279,...
        ...
    
    Returns:
        dict: {u: {c: x_uc}} co-occurrence proportions
    """
    df = pd.read_csv(filename)
    
    # Find the UCoD column (first column or named 'UCoD')
    if 'UCoD' in df.columns:
        ucod_col = 'UCoD'
    else:
        ucod_col = df.columns[0]
    
    x_matrix = {u: {} for u in ALLOWED_LETTERS}
    
    for _, row in df.iterrows():
        u = row[ucod_col]
        if u not in ALLOWED_SET:
            continue
        for c in ALLOWED_LETTERS:
            if c in df.columns:
                val = row[c]
                # Handle empty or non-numeric values
                if pd.isna(val) or val == '':
                    x_matrix[u][c] = 0.0
                else:
                    x_matrix[u][c] = float(val)
            else:
                x_matrix[u][c] = 0.0
    
    return x_matrix


def save_x_matrix_to_dataframe(x_matrix, N_u=None):
    """
    Convert x_uc matrix to a DataFrame for output.
    
    Args:
        x_matrix: {u: {c: x_uc}} co-occurrence proportions
        N_u: {u: count} deaths per UCoD (optional, for reference)
    
    Returns:
        pd.DataFrame with UCoD as rows and letters as columns
    """
    rows = []
    for u in ALLOWED_LETTERS:
        row = {'UCoD': u}
        if N_u is not None:
            row['N_u'] = N_u.get(u, 0)
        for c in ALLOWED_LETTERS:
            row[c] = round(x_matrix[u].get(c, 0.0), 4)
        # Row sum (comorbidity load)
        row['Row_Sum'] = round(sum(x_matrix[u].get(c, 0.0) for c in ALLOWED_LETTERS), 4)
        rows.append(row)
    
    return pd.DataFrame(rows)


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_input_data(df, year_col=None, age_col=None):
    """Validate the input dataframe structure and content."""
    errors = []
    
    # Check required columns
    required_cols = ['Count', 'rUDS', 'eUDS']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors
    
    # Check for empty dataframe
    if len(df) == 0:
        errors.append("Input dataframe is empty")
        return False, errors
    
    # Check Count column
    if not pd.api.types.is_numeric_dtype(df['Count']):
        errors.append("Count column must be numeric")
    
    if (df['Count'] <= 0).any():
        errors.append(f"Count column contains non-positive values")
    
    # Check for NaN values
    nan_rUDS = df['rUDS'].isna()
    nan_eUDS = df['eUDS'].isna()
    
    if nan_rUDS.any():
        errors.append(f"rUDS contains {nan_rUDS.sum()} NaN values")
    
    if nan_eUDS.any():
        errors.append(f"eUDS contains {nan_eUDS.sum()} NaN values")
    
    # Check for empty strings (only on non-NaN values)
    valid_rows = ~(nan_rUDS | nan_eUDS)
    if valid_rows.any():
        empty_rUDS = (df.loc[valid_rows, 'rUDS'] == '')
        empty_eUDS = (df.loc[valid_rows, 'eUDS'] == '')
        
        if empty_rUDS.any():
            errors.append(f"rUDS contains {empty_rUDS.sum()} empty strings")
        if empty_eUDS.any():
            errors.append(f"eUDS contains {empty_eUDS.sum()} empty strings")
    
    # Check for invalid letters (only on valid values)
    valid_rUDS = df.loc[valid_rows & (df['rUDS'] != ''), 'rUDS']
    valid_eUDS = df.loc[valid_rows & (df['eUDS'] != ''), 'eUDS']
    
    if len(valid_rUDS) > 0:
        all_letters_rUDS = set(''.join(valid_rUDS.astype(str)))
        invalid_rUDS = all_letters_rUDS - ALLOWED_SET
        if invalid_rUDS:
            errors.append(f"rUDS contains invalid letters: {sorted(invalid_rUDS)}")
    
    if len(valid_eUDS) > 0:
        all_letters_eUDS = set(''.join(valid_eUDS.astype(str)))
        invalid_eUDS = all_letters_eUDS - ALLOWED_SET
        if invalid_eUDS:
            errors.append(f"eUDS contains invalid letters: {sorted(invalid_eUDS)}")
    
    # Check Year column if present
    if year_col and year_col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[year_col]):
            errors.append(f"Year column '{year_col}' must be numeric")
    
    # Check AgeGroup column if present
    if age_col and age_col in df.columns:
        unique_age_groups = set(df[age_col].dropna().unique())
        expected_age_groups = {'LT65', 'GE65'}
        unexpected = unique_age_groups - expected_age_groups - {''}
        if unexpected:
            errors.append(f"Age group column contains unexpected values: {unexpected}")
    
    return len(errors) == 0, errors


def validate_string_lengths(df):
    """Validate that UDS strings have unique letters (no duplicates)."""
    errors = []
    
    for idx, row in df.iterrows():
        rUDS_len = len(row['rUDS'])
        eUDS_len = len(row['eUDS'])
        
        if len(set(row['rUDS'])) != rUDS_len:
            errors.append(f"Row {idx}: rUDS '{row['rUDS']}' contains duplicate letters")
        if len(set(row['eUDS'])) != eUDS_len:
            errors.append(f"Row {idx}: eUDS '{row['eUDS']}' contains duplicate letters")
        
        if len(errors) >= 10:
            errors.append("... (showing first 10 string validation errors)")
            break
    
    return len(errors) == 0, errors


def validate_weights(records_with_weights):
    """Validate that weight calculations sum correctly for each record."""
    errors = []
    error_count = 0
    
    for rec in records_with_weights:
        count = rec['count']
        
        w1_sum = sum(rec['w1_weights'])
        if abs(w1_sum - count) > EPSILON:
            error_count += 1
            if error_count <= 10:
                errors.append(f"Record {rec['rUDS']}/{rec['eUDS']}: W1 sum {w1_sum:.6f} != Count {count}")
        
        w2_sum = sum(rec['w2_weights'])
        if abs(w2_sum - count) > EPSILON:
            error_count += 1
            if error_count <= 10:
                errors.append(f"Record {rec['rUDS']}/{rec['eUDS']}: W2 sum {w2_sum:.6f} != Count {count}")
        
        w3_sum = sum(rec['w3_weights'])
        if abs(w3_sum - count) > EPSILON:
            error_count += 1
            if error_count <= 10:
                errors.append(f"Record {rec['rUDS']}/{rec['eUDS']}: W3 sum {w3_sum:.6f} != Count {count}")
    
    if error_count > 10:
        errors.append(f"... ({error_count - 10} more weight validation errors)")
    
    return len(errors) == 0, errors


def validate_x_matrix(x_matrix):
    """Validate x_uc matrix properties."""
    errors = []
    
    # Check diagonal is 0
    for u in ALLOWED_LETTERS:
        if x_matrix[u].get(u, 0) != 0:
            errors.append(f"x_{u}{u} should be 0, got {x_matrix[u][u]}")
    
    # Check values in [0, 1]
    for u in ALLOWED_LETTERS:
        for c in ALLOWED_LETTERS:
            val = x_matrix[u].get(c, 0)
            if val < 0 or val > 1:
                errors.append(f"x_{u}{c} = {val} is outside [0,1]")
    
    return len(errors) == 0, errors


# ============================================================================
# WEIGHT CALCULATION
# ============================================================================

def calculate_weights(uds_string, count):
    """
    Calculate W1 and W2 weights for each letter in a UDS string.
    
    W1 scheme:
        - If Ls=1: w=1
        - If Ls>1: first letter w=0.5, others w=0.5/(Ls-1)
    
    W2 scheme:
        - If Ls=1: w=1
        - If Ls>1: all letters w=1.0/Ls (equal weighting)
    
    Note: W3 (Dobson) weights are calculated separately after computing
    the x_uc matrix from the full dataset.
    """
    Ls = len(uds_string)
    
    if Ls == 1:
        return [count], [count]
    
    w1_weights = []
    w2_weights = []
    
    for i, letter in enumerate(uds_string):
        if i == 0:
            w1 = count * 0.5
            w2 = count * 1.0 / Ls
        else:
            w1 = count * 0.5 / (Ls - 1)
            w2 = count * 1.0 / Ls
        
        w1_weights.append(w1)
        w2_weights.append(w2)
    
    return w1_weights, w2_weights


def compute_dobson_x_matrix(df, uds_column):
    """
    Compute the Dobson x_uc co-occurrence proportion matrix from data.
    
    x_uc = N_c|u / N_u  for u != c
    x_uc = 0            for u == c
    
    where:
    - N_c|u = number of deaths with UCoD=u and CCoD=c (c appears after position 1)
    - N_u = total number of deaths with UCoD=u
    
    This is Step 1 of the Dobson method: learn association strengths from data.
    
    Returns:
        tuple: (x_matrix, N_u) where:
            x_matrix: {u: {c: x_uc}} co-occurrence proportions
            N_u: {u: count} deaths per UCoD
    """
    # Count N_u: deaths per UCoD
    N_u = defaultdict(int)
    
    # Count N_c|u: co-occurrences of CCoD c with UCoD u
    N_c_given_u = defaultdict(lambda: defaultdict(int))
    
    for idx, row in df.iterrows():
        uds = row[uds_column]
        count = row['Count']
        
        # Skip invalid entries
        if not isinstance(uds, str) or len(uds) == 0:
            continue
        
        u = uds[0]  # UCoD (first letter)
        N_u[u] += count
        
        # Count contributing causes (positions 2+)
        ccods = set(uds[1:])  # unique contributing causes
        for c in ccods:
            N_c_given_u[u][c] += count
    
    # Compute x_uc matrix
    x_matrix = {u: {} for u in ALLOWED_LETTERS}
    
    for u in ALLOWED_LETTERS:
        total_u = N_u[u]
        for c in ALLOWED_LETTERS:
            if u == c:
                x_matrix[u][c] = 0.0  # x_uu = 0 by definition
            elif total_u > 0:
                x_matrix[u][c] = N_c_given_u[u][c] / total_u
            else:
                x_matrix[u][c] = 0.0
    
    return x_matrix, dict(N_u)


def calculate_w3_weights(uds_string, count, x_matrix):
    """
    Calculate W3 (Dobson data-driven) weights for each letter in a UDS string.
    
    W3 scheme (Dobson et al. 2023):
        - For single-cause death (Ls=1): UCoD weight = 1
        - For multi-cause death (Ls>1):
            * For each CCoD c: w_c = x_uc / n  (where n = number of CCoDs)
            * For UCoD u: w_u = 1 - sum(w_c for all CCoDs)
    
    This weights contributing causes by how often they co-occur with the UCoD
    in the population, rather than treating all CCoDs equally.
    
    Args:
        uds_string: The UDS string (first letter = UCoD, rest = CCoDs)
        count: Death count for this record
        x_matrix: Pre-computed {u: {c: x_uc}} co-occurrence proportions
    
    Returns:
        list: W3 weights for each position in the string, scaled by count
    """
    Ls = len(uds_string)
    
    if Ls == 1:
        return [count]  # Single cause: UCoD gets full weight
    
    u = uds_string[0]  # UCoD
    ccods = list(uds_string[1:])  # Contributing causes
    n = len(ccods)  # Number of CCoDs
    
    w3_weights = []
    
    # Calculate weights for each CCoD
    ccod_weight_sum = 0.0
    ccod_weights = []
    
    for c in ccods:
        # w_c = x_uc / n
        x_uc = x_matrix[u].get(c, 0.0)
        w_c = x_uc / n
        ccod_weights.append(w_c)
        ccod_weight_sum += w_c
    
    # UCoD weight = 1 - sum of CCoD weights
    w_u = 1.0 - ccod_weight_sum
    
    # Ensure w_u is non-negative (in case x_uc values sum to >1)
    if w_u < 0:
        w_u = 0.0
    
    # Build weight list in string order: [UCoD, CCoD1, CCoD2, ...]
    w3_weights.append(w_u * count)
    for w_c in ccod_weights:
        w3_weights.append(w_c * count)
    
    return w3_weights


# ============================================================================
# ANALYSIS FUNCTIONS - LEVEL A (Individual UDS)
# ============================================================================

def analyze_individual_uds(df, uds_column, x_matrix=None):
    """
    Perform Level A analysis on a single UDS column (rUDS or eUDS).
    
    If x_matrix is provided, also computes W3 (Dobson) weights.
    """
    results = {
        'column_name': uds_column,
        'total_count': df['Count'].sum(),
        'records_processed': len(df)
    }
    
    # Initialize counters for each letter
    letter_stats = {letter: {
        'first': 0,
        'not_first': 0,
        'only': 0,
        'w1': 0.0,
        'w2': 0.0,
        'w3': 0.0,
        'anywhere': 0,
        'Ls_distribution': defaultdict(int),
        'Ls_values': []
    } for letter in ALLOWED_LETTERS}
    
    # Co-occurrence counters
    cooccur_unweighted = {letter: Counter() for letter in ALLOWED_LETTERS}
    cooccur_w1 = {letter: defaultdict(float) for letter in ALLOWED_LETTERS}
    cooccur_w2 = {letter: defaultdict(float) for letter in ALLOWED_LETTERS}
    cooccur_w3 = {letter: defaultdict(float) for letter in ALLOWED_LETTERS}
    
    # Store records with weights for validation
    records_with_weights = []
    
    # Process each record
    for idx, row in df.iterrows():
        uds = row[uds_column]
        count = row['Count']
        Ls = len(uds)
        
        w1_weights, w2_weights = calculate_weights(uds, count)
        
        # Calculate W3 weights if x_matrix is provided
        if x_matrix is not None:
            w3_weights = calculate_w3_weights(uds, count, x_matrix)
        else:
            w3_weights = [0.0] * Ls  # Placeholder
        
        records_with_weights.append({
            'rUDS': row['rUDS'],
            'eUDS': row['eUDS'],
            'uds': uds,
            'count': count,
            'Ls': Ls,
            'w1_weights': w1_weights,
            'w2_weights': w2_weights,
            'w3_weights': w3_weights
        })
        
        first_letter = uds[0]
        
        letter_stats[first_letter]['Ls_distribution'][Ls] += count
        letter_stats[first_letter]['Ls_values'].extend([Ls] * int(count))
        
        for i, letter in enumerate(uds):
            if i == 0:
                letter_stats[letter]['first'] += count
            if i > 0:
                letter_stats[letter]['not_first'] += count
            if Ls == 1:
                letter_stats[letter]['only'] += count
            
            letter_stats[letter]['w1'] += w1_weights[i]
            letter_stats[letter]['w2'] += w2_weights[i]
            letter_stats[letter]['w3'] += w3_weights[i]
            letter_stats[letter]['anywhere'] += count
            
            if Ls == 1 and i == 0:
                cooccur_unweighted[first_letter][first_letter] += count
                cooccur_w1[first_letter][first_letter] += w1_weights[i]
                cooccur_w2[first_letter][first_letter] += w2_weights[i]
                cooccur_w3[first_letter][first_letter] += w3_weights[i]
            elif i > 0:
                cooccur_unweighted[first_letter][letter] += count
                cooccur_w1[first_letter][letter] += w1_weights[i]
                cooccur_w2[first_letter][letter] += w2_weights[i]
                cooccur_w3[first_letter][letter] += w3_weights[i]
    
    # Calculate mean Ls for each first letter
    for letter in ALLOWED_LETTERS:
        if letter_stats[letter]['Ls_values']:
            letter_stats[letter]['mean_Ls'] = np.mean(letter_stats[letter]['Ls_values'])
        else:
            letter_stats[letter]['mean_Ls'] = 0.0
        del letter_stats[letter]['Ls_values']
    
    results['letter_stats'] = letter_stats
    results['cooccur_unweighted'] = cooccur_unweighted
    results['cooccur_w1'] = cooccur_w1
    results['cooccur_w2'] = cooccur_w2
    results['cooccur_w3'] = cooccur_w3
    results['records_with_weights'] = records_with_weights
    
    return results


# ============================================================================
# ANALYSIS FUNCTIONS - LEVEL B (Comparative) - BIDIRECTIONAL
# ============================================================================

def analyze_comparative(df):
    """
    Perform Level B analysis comparing rUDS and eUDS.
    
    Creates BIDIRECTIONAL transition matrices and position analyses.
    """
    results = {
        'total_count': df['Count'].sum()
    }
    
    # BIDIRECTIONAL Transition matrices
    # Entity -> Record (STANDARD): What did physician's code become?
    transition_e_to_r = {e1: {r1: 0 for r1 in ALLOWED_LETTERS} 
                         for e1 in ALLOWED_LETTERS}
    
    # Record -> Entity (REVERSE): What was original code for this output?
    transition_r_to_e = {r1: {e1: 0 for e1 in ALLOWED_LETTERS} 
                         for r1 in ALLOWED_LETTERS}
    
    # BIDIRECTIONAL Position distributions
    # Position of r1 (Record first) in eUDS string
    position_r1_in_eUDS = {letter: defaultdict(int) for letter in ALLOWED_LETTERS}
    
    # Position of e1 (Entity first) in rUDS string
    position_e1_in_rUDS = {letter: defaultdict(int) for letter in ALLOWED_LETTERS}
    
    concordance_count = 0
    
    for idx, row in df.iterrows():
        rUDS = row['rUDS']
        eUDS = row['eUDS']
        count = row['Count']
        
        r1 = rUDS[0]  # Record Axis first letter
        e1 = eUDS[0]  # Entity Axis first letter
        
        # Update BOTH transition matrices
        transition_e_to_r[e1][r1] += count  # Entity -> Record
        transition_r_to_e[r1][e1] += count  # Record -> Entity
        
        # Check concordance
        if r1 == e1:
            concordance_count += count
        
        # Position of r1 in eUDS
        if r1 in eUDS:
            position = eUDS.index(r1) + 1
            position_r1_in_eUDS[r1][position] += count
        else:
            position_r1_in_eUDS[r1][0] += count  # 0 = absent
        
        # Position of e1 in rUDS
        if e1 in rUDS:
            position = rUDS.index(e1) + 1
            position_e1_in_rUDS[e1][position] += count
        else:
            position_e1_in_rUDS[e1][0] += count  # 0 = absent
    
    results['transition_e_to_r'] = transition_e_to_r
    results['transition_r_to_e'] = transition_r_to_e
    results['position_r1_in_eUDS'] = position_r1_in_eUDS
    results['position_e1_in_rUDS'] = position_e1_in_rUDS
    results['concordance_count'] = concordance_count
    results['concordance_rate'] = concordance_count / results['total_count'] if results['total_count'] > 0 else 0
    
    return results


# ============================================================================
# VALIDATION FUNCTIONS - CROSS-CHECKS
# ============================================================================

def validate_level_a_results(results):
    """Validate Level A analysis results for internal consistency."""
    errors = []
    letter_stats = results['letter_stats']
    total_count = results['total_count']
    
    first_total = sum(letter_stats[letter]['first'] for letter in ALLOWED_LETTERS)
    if abs(first_total - total_count) > EPSILON:
        errors.append(f"{results['column_name']}: first sum ({first_total}) != total count ({total_count})")
    
    for letter in ALLOWED_LETTERS:
        first = letter_stats[letter]['first']
        not_first = letter_stats[letter]['not_first']
        anywhere = letter_stats[letter]['anywhere']
        
        if abs(first + not_first - anywhere) > EPSILON:
            errors.append(f"{results['column_name']}, letter {letter}: first + not_first != anywhere")
    
    w1_total = sum(letter_stats[letter]['w1'] for letter in ALLOWED_LETTERS)
    if abs(w1_total - total_count) > EPSILON:
        errors.append(f"{results['column_name']}: W1 total ({w1_total:.2f}) != total count ({total_count})")
    
    w2_total = sum(letter_stats[letter]['w2'] for letter in ALLOWED_LETTERS)
    if abs(w2_total - total_count) > EPSILON:
        errors.append(f"{results['column_name']}: W2 total ({w2_total:.2f}) != total count ({total_count})")
    
    w3_total = sum(letter_stats[letter]['w3'] for letter in ALLOWED_LETTERS)
    if abs(w3_total - total_count) > EPSILON:
        errors.append(f"{results['column_name']}: W3 total ({w3_total:.2f}) != total count ({total_count})")
    
    return len(errors) == 0, errors


def validate_level_b_results(comp_results, total_count):
    """Validate Level B comparative analysis results."""
    errors = []
    
    # Check Entity->Record transition matrix total
    e_to_r_total = sum(sum(comp_results['transition_e_to_r'][e1].values()) 
                       for e1 in ALLOWED_LETTERS)
    
    if abs(e_to_r_total - total_count) > EPSILON:
        errors.append(f"Entity->Record transition total ({e_to_r_total}) != total count ({total_count})")
    
    # Check Record->Entity transition matrix total
    r_to_e_total = sum(sum(comp_results['transition_r_to_e'][r1].values()) 
                       for r1 in ALLOWED_LETTERS)
    
    if abs(r_to_e_total - total_count) > EPSILON:
        errors.append(f"Record->Entity transition total ({r_to_e_total}) != total count ({total_count})")
    
    # Verify matrices are transposes of each other
    for e1 in ALLOWED_LETTERS:
        for r1 in ALLOWED_LETTERS:
            e_to_r_val = comp_results['transition_e_to_r'][e1][r1]
            r_to_e_val = comp_results['transition_r_to_e'][r1][e1]
            if abs(e_to_r_val - r_to_e_val) > EPSILON:
                errors.append(f"Transpose mismatch at ({e1},{r1}): E->R={e_to_r_val}, R->E={r_to_e_val}")
                break
    
    return len(errors) == 0, errors


# ============================================================================
# OUTPUT FUNCTIONS
# ============================================================================

def create_output_dataframes_for_age_group(rUDS_results, eUDS_results, comp_results, 
                                           x_matrix_rUDS, x_matrix_eUDS, 
                                           N_u_rUDS, N_u_eUDS, age_group):
    """Create pandas DataFrames for a single age group (without suffix - will be stacked later)."""
    output_dfs = {}
    
    # ==== Level A Tables ====
    
    # Table 1: Letter statistics for rUDS
    rUDS_stats = []
    for letter in ALLOWED_LETTERS:
        stats = rUDS_results['letter_stats'][letter]
        rUDS_stats.append({
            'Letter': letter,
            'First': stats['first'],
            'Not_First': stats['not_first'],
            'Only': stats['only'],
            'W1': stats['w1'],
            'W2': stats['w2'],
            'W3': stats['w3'],
            'Anywhere': stats['anywhere'],
            'Mean_Ls': stats['mean_Ls']
        })
    
    # Add "All" summary row
    total_ls_weighted = sum(
        ls * count 
        for letter in ALLOWED_LETTERS 
        for ls, count in rUDS_results['letter_stats'][letter]['Ls_distribution'].items()
    )
    total_first = sum(rUDS_results['letter_stats'][l]['first'] for l in ALLOWED_LETTERS)
    all_mean_ls = total_ls_weighted / total_first if total_first > 0 else 0
    
    rUDS_stats.append({
        'Letter': 'All',
        'First': sum(rUDS_results['letter_stats'][l]['first'] for l in ALLOWED_LETTERS),
        'Not_First': sum(rUDS_results['letter_stats'][l]['not_first'] for l in ALLOWED_LETTERS),
        'Only': sum(rUDS_results['letter_stats'][l]['only'] for l in ALLOWED_LETTERS),
        'W1': round(sum(rUDS_results['letter_stats'][l]['w1'] for l in ALLOWED_LETTERS), 1),
        'W2': round(sum(rUDS_results['letter_stats'][l]['w2'] for l in ALLOWED_LETTERS), 1),
        'W3': round(sum(rUDS_results['letter_stats'][l]['w3'] for l in ALLOWED_LETTERS), 1),
        'Anywhere': sum(rUDS_results['letter_stats'][l]['anywhere'] for l in ALLOWED_LETTERS),
        'Mean_Ls': round(all_mean_ls, 3)
    })
    
    df = pd.DataFrame(rUDS_stats)
    df['W1'] = df['W1'].round(1)
    df['W2'] = df['W2'].round(1)
    df['W3'] = df['W3'].round(1)
    df['Mean_Ls'] = df['Mean_Ls'].round(3)
    output_dfs['rUDS_letter_stats'] = df
    
    # Table 2: Letter statistics for eUDS
    eUDS_stats = []
    for letter in ALLOWED_LETTERS:
        stats = eUDS_results['letter_stats'][letter]
        eUDS_stats.append({
            'Letter': letter,
            'First': stats['first'],
            'Not_First': stats['not_first'],
            'Only': stats['only'],
            'W1': stats['w1'],
            'W2': stats['w2'],
            'W3': stats['w3'],
            'Anywhere': stats['anywhere'],
            'Mean_Ls': stats['mean_Ls']
        })
    
    total_ls_weighted = sum(
        ls * count 
        for letter in ALLOWED_LETTERS 
        for ls, count in eUDS_results['letter_stats'][letter]['Ls_distribution'].items()
    )
    total_first = sum(eUDS_results['letter_stats'][l]['first'] for l in ALLOWED_LETTERS)
    all_mean_ls = total_ls_weighted / total_first if total_first > 0 else 0
    
    eUDS_stats.append({
        'Letter': 'All',
        'First': sum(eUDS_results['letter_stats'][l]['first'] for l in ALLOWED_LETTERS),
        'Not_First': sum(eUDS_results['letter_stats'][l]['not_first'] for l in ALLOWED_LETTERS),
        'Only': sum(eUDS_results['letter_stats'][l]['only'] for l in ALLOWED_LETTERS),
        'W1': round(sum(eUDS_results['letter_stats'][l]['w1'] for l in ALLOWED_LETTERS), 1),
        'W2': round(sum(eUDS_results['letter_stats'][l]['w2'] for l in ALLOWED_LETTERS), 1),
        'W3': round(sum(eUDS_results['letter_stats'][l]['w3'] for l in ALLOWED_LETTERS), 1),
        'Anywhere': sum(eUDS_results['letter_stats'][l]['anywhere'] for l in ALLOWED_LETTERS),
        'Mean_Ls': round(all_mean_ls, 3)
    })
    
    df = pd.DataFrame(eUDS_stats)
    df['W1'] = df['W1'].round(1)
    df['W2'] = df['W2'].round(1)
    df['W3'] = df['W3'].round(1)
    df['Mean_Ls'] = df['Mean_Ls'].round(3)
    output_dfs['eUDS_letter_stats'] = df
    
    # Table 3: String length distribution for rUDS
    rUDS_ls_pivot = defaultdict(lambda: {letter: 0 for letter in ALLOWED_LETTERS + ['All']})
    
    for letter in ALLOWED_LETTERS:
        dist = rUDS_results['letter_stats'][letter]['Ls_distribution']
        for ls, count in dist.items():
            if ls <= 10:
                rUDS_ls_pivot[ls][letter] = count
                rUDS_ls_pivot[ls]['All'] += count
    
    rUDS_ls_data = []
    for ls in range(1, 11):
        row = {'String_Length': ls}
        for letter in ALLOWED_LETTERS:
            row[letter] = rUDS_ls_pivot[ls][letter]
        row['All'] = rUDS_ls_pivot[ls]['All']
        rUDS_ls_data.append(row)
    
    mean_row = {'String_Length': 'Mean'}
    for letter in ALLOWED_LETTERS + ['All']:
        total_count = sum(rUDS_ls_pivot[ls][letter] for ls in range(1, 11))
        if total_count > 0:
            weighted_sum = sum(ls * rUDS_ls_pivot[ls][letter] for ls in range(1, 11))
            mean_row[letter] = round(weighted_sum / total_count, 3)
        else:
            mean_row[letter] = 0.0
    rUDS_ls_data.append(mean_row)
    
    output_dfs['rUDS_length_distribution'] = pd.DataFrame(rUDS_ls_data)
    
    # Table 4: String length distribution for eUDS
    eUDS_ls_pivot = defaultdict(lambda: {letter: 0 for letter in ALLOWED_LETTERS + ['All']})
    
    for letter in ALLOWED_LETTERS:
        dist = eUDS_results['letter_stats'][letter]['Ls_distribution']
        for ls, count in dist.items():
            if ls <= 10:
                eUDS_ls_pivot[ls][letter] = count
                eUDS_ls_pivot[ls]['All'] += count
    
    eUDS_ls_data = []
    for ls in range(1, 11):
        row = {'String_Length': ls}
        for letter in ALLOWED_LETTERS:
            row[letter] = eUDS_ls_pivot[ls][letter]
        row['All'] = eUDS_ls_pivot[ls]['All']
        eUDS_ls_data.append(row)
    
    mean_row = {'String_Length': 'Mean'}
    for letter in ALLOWED_LETTERS + ['All']:
        total_count = sum(eUDS_ls_pivot[ls][letter] for ls in range(1, 11))
        if total_count > 0:
            weighted_sum = sum(ls * eUDS_ls_pivot[ls][letter] for ls in range(1, 11))
            mean_row[letter] = round(weighted_sum / total_count, 3)
        else:
            mean_row[letter] = 0.0
    eUDS_ls_data.append(mean_row)
    
    output_dfs['eUDS_length_distribution'] = pd.DataFrame(eUDS_ls_data)
    
    # Table 5: rUDS Co-occurrence Count
    rUDS_count_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0
        for other_letter in ALLOWED_LETTERS:
            count = rUDS_results['cooccur_unweighted'][first_letter].get(other_letter, 0)
            row[other_letter] = count
            row_total += count
        row['All'] = row_total
        rUDS_count_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        all_row[other_letter] = sum(rUDS_results['cooccur_unweighted'][fl].get(other_letter, 0) 
                                     for fl in ALLOWED_LETTERS)
    all_row['All'] = sum(all_row[l] for l in ALLOWED_LETTERS)
    rUDS_count_matrix.append(all_row)
    
    output_dfs['rUDS_cooccur_Count'] = pd.DataFrame(rUDS_count_matrix)
    
    # Table 6: rUDS Co-occurrence W1
    rUDS_w1_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w1 = rUDS_results['cooccur_w1'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w1, 1)
            row_total += w1
        row['All'] = round(row_total, 1)
        rUDS_w1_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(rUDS_results['cooccur_w1'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    rUDS_w1_matrix.append(all_row)
    
    output_dfs['rUDS_cooccur_W1'] = pd.DataFrame(rUDS_w1_matrix)
    
    # Table 7: rUDS Co-occurrence W2
    rUDS_w2_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w2 = rUDS_results['cooccur_w2'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w2, 1)
            row_total += w2
        row['All'] = round(row_total, 1)
        rUDS_w2_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(rUDS_results['cooccur_w2'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    rUDS_w2_matrix.append(all_row)
    
    output_dfs['rUDS_cooccur_W2'] = pd.DataFrame(rUDS_w2_matrix)
    
    # Table 7b: rUDS Co-occurrence W3 (Dobson)
    rUDS_w3_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w3 = rUDS_results['cooccur_w3'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w3, 1)
            row_total += w3
        row['All'] = round(row_total, 1)
        rUDS_w3_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(rUDS_results['cooccur_w3'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    rUDS_w3_matrix.append(all_row)
    
    output_dfs['rUDS_cooccur_W3'] = pd.DataFrame(rUDS_w3_matrix)
    
    # Table 7c: rUDS x_uc matrix (Dobson proportions)
    output_dfs['rUDS_x_uc_matrix'] = save_x_matrix_to_dataframe(x_matrix_rUDS, N_u_rUDS)
    
    # Table 8: eUDS Co-occurrence Count
    eUDS_count_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0
        for other_letter in ALLOWED_LETTERS:
            count = eUDS_results['cooccur_unweighted'][first_letter].get(other_letter, 0)
            row[other_letter] = count
            row_total += count
        row['All'] = row_total
        eUDS_count_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        all_row[other_letter] = sum(eUDS_results['cooccur_unweighted'][fl].get(other_letter, 0) 
                                     for fl in ALLOWED_LETTERS)
    all_row['All'] = sum(all_row[l] for l in ALLOWED_LETTERS)
    eUDS_count_matrix.append(all_row)
    
    output_dfs['eUDS_cooccur_Count'] = pd.DataFrame(eUDS_count_matrix)
    
    # Table 9: eUDS Co-occurrence W1
    eUDS_w1_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w1 = eUDS_results['cooccur_w1'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w1, 1)
            row_total += w1
        row['All'] = round(row_total, 1)
        eUDS_w1_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(eUDS_results['cooccur_w1'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    eUDS_w1_matrix.append(all_row)
    
    output_dfs['eUDS_cooccur_W1'] = pd.DataFrame(eUDS_w1_matrix)
    
    # Table 10: eUDS Co-occurrence W2
    eUDS_w2_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w2 = eUDS_results['cooccur_w2'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w2, 1)
            row_total += w2
        row['All'] = round(row_total, 1)
        eUDS_w2_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(eUDS_results['cooccur_w2'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    eUDS_w2_matrix.append(all_row)
    
    output_dfs['eUDS_cooccur_W2'] = pd.DataFrame(eUDS_w2_matrix)
    
    # Table 10b: eUDS Co-occurrence W3 (Dobson)
    eUDS_w3_matrix = []
    for first_letter in ALLOWED_LETTERS:
        row = {'First_Letter': first_letter}
        row_total = 0.0
        for other_letter in ALLOWED_LETTERS:
            w3 = eUDS_results['cooccur_w3'][first_letter].get(other_letter, 0.0)
            row[other_letter] = round(w3, 1)
            row_total += w3
        row['All'] = round(row_total, 1)
        eUDS_w3_matrix.append(row)
    
    all_row = {'First_Letter': 'All'}
    for other_letter in ALLOWED_LETTERS:
        val = sum(eUDS_results['cooccur_w3'][fl].get(other_letter, 0.0) 
                  for fl in ALLOWED_LETTERS)
        all_row[other_letter] = round(val, 1)
    all_row['All'] = round(sum(all_row[l] for l in ALLOWED_LETTERS), 1)
    eUDS_w3_matrix.append(all_row)
    
    output_dfs['eUDS_cooccur_W3'] = pd.DataFrame(eUDS_w3_matrix)
    
    # Table 10c: eUDS x_uc matrix (Dobson proportions)
    output_dfs['eUDS_x_uc_matrix'] = save_x_matrix_to_dataframe(x_matrix_eUDS, N_u_eUDS)
    
    # ==== Level B Tables - BIDIRECTIONAL ====
    
    # Table 11a: Entity -> Record Transition Matrix (STANDARD)
    trans_e_to_r_data = []
    for e1 in ALLOWED_LETTERS:
        row = {'Entity_First': e1}
        row_total = 0
        for r1 in ALLOWED_LETTERS:
            count = comp_results['transition_e_to_r'][e1][r1]
            row[r1] = count
            row_total += count
        row['All'] = row_total
        trans_e_to_r_data.append(row)
    
    all_row = {'Entity_First': 'All'}
    for r1 in ALLOWED_LETTERS:
        all_row[r1] = sum(comp_results['transition_e_to_r'][e1][r1] for e1 in ALLOWED_LETTERS)
    all_row['All'] = sum(all_row[r1] for r1 in ALLOWED_LETTERS)
    trans_e_to_r_data.append(all_row)
    
    output_dfs['transition_matrix'] = pd.DataFrame(trans_e_to_r_data)
    
    # Table 11b: Record -> Entity Transition Matrix (REVERSE)
    trans_r_to_e_data = []
    for r1 in ALLOWED_LETTERS:
        row = {'Record_First': r1}
        row_total = 0
        for e1 in ALLOWED_LETTERS:
            count = comp_results['transition_r_to_e'][r1][e1]
            row[e1] = count
            row_total += count
        row['All'] = row_total
        trans_r_to_e_data.append(row)
    
    all_row = {'Record_First': 'All'}
    for e1 in ALLOWED_LETTERS:
        all_row[e1] = sum(comp_results['transition_r_to_e'][r1][e1] for r1 in ALLOWED_LETTERS)
    all_row['All'] = sum(all_row[e1] for e1 in ALLOWED_LETTERS)
    trans_r_to_e_data.append(all_row)
    
    output_dfs['trans_Record_to_Entity'] = pd.DataFrame(trans_r_to_e_data)
    
    # Table 12a: Position of r1 (Record first) in eUDS
    max_pos_r1 = 0
    for r1 in ALLOWED_LETTERS:
        for pos in comp_results['position_r1_in_eUDS'][r1].keys():
            if pos > max_pos_r1:
                max_pos_r1 = pos
    
    max_pos_r1 = min(max_pos_r1, 10)
    position_cols_r1 = ['Absent'] + [f'Pos_{i}' for i in range(1, max_pos_r1 + 1)] + ['All']
    position_pivot_r1 = {pos: {letter: 0 for letter in ALLOWED_LETTERS + ['All']} for pos in position_cols_r1}
    
    for r1 in ALLOWED_LETTERS:
        for position, count in comp_results['position_r1_in_eUDS'][r1].items():
            if position == 0:
                pos_label = 'Absent'
            elif position <= max_pos_r1:
                pos_label = f'Pos_{position}'
            else:
                continue
            position_pivot_r1[pos_label][r1] = count
    
    for r1 in ALLOWED_LETTERS:
        position_pivot_r1['All'][r1] = sum(position_pivot_r1[pos][r1] for pos in position_cols_r1 if pos != 'All')
    
    for pos in position_cols_r1:
        if pos != 'All':
            position_pivot_r1[pos]['All'] = sum(position_pivot_r1[pos][letter] for letter in ALLOWED_LETTERS)
    
    position_pivot_r1['All']['All'] = sum(position_pivot_r1['All'][letter] for letter in ALLOWED_LETTERS)
    
    position_data_r1 = []
    for r1 in ALLOWED_LETTERS + ['All']:
        row = {'Record_First': r1}
        for pos in position_cols_r1:
            row[pos] = position_pivot_r1[pos][r1]
        position_data_r1.append(row)
    
    output_dfs['r1_position_in_eUDS'] = pd.DataFrame(position_data_r1)
    
    # Table 12b: Position of e1 (Entity first) in rUDS
    max_pos_e1 = 0
    for e1 in ALLOWED_LETTERS:
        for pos in comp_results['position_e1_in_rUDS'][e1].keys():
            if pos > max_pos_e1:
                max_pos_e1 = pos
    
    max_pos_e1 = min(max_pos_e1, 10)
    position_cols_e1 = ['Absent'] + [f'Pos_{i}' for i in range(1, max_pos_e1 + 1)] + ['All']
    position_pivot_e1 = {pos: {letter: 0 for letter in ALLOWED_LETTERS + ['All']} for pos in position_cols_e1}
    
    for e1 in ALLOWED_LETTERS:
        for position, count in comp_results['position_e1_in_rUDS'][e1].items():
            if position == 0:
                pos_label = 'Absent'
            elif position <= max_pos_e1:
                pos_label = f'Pos_{position}'
            else:
                continue
            position_pivot_e1[pos_label][e1] = count
    
    for e1 in ALLOWED_LETTERS:
        position_pivot_e1['All'][e1] = sum(position_pivot_e1[pos][e1] for pos in position_cols_e1 if pos != 'All')
    
    for pos in position_cols_e1:
        if pos != 'All':
            position_pivot_e1[pos]['All'] = sum(position_pivot_e1[pos][letter] for letter in ALLOWED_LETTERS)
    
    position_pivot_e1['All']['All'] = sum(position_pivot_e1['All'][letter] for letter in ALLOWED_LETTERS)
    
    position_data_e1 = []
    for e1 in ALLOWED_LETTERS + ['All']:
        row = {'Entity_First': e1}
        for pos in position_cols_e1:
            row[pos] = position_pivot_e1[pos][e1]
        position_data_e1.append(row)
    
    output_dfs['pos_e1_in_rUDS'] = pd.DataFrame(position_data_e1)
    
    return output_dfs


def stack_age_group_tables(all_age_group_dfs, age_groups_processed):
    """
    Stack tables from multiple age groups into single DataFrames.
    
    Each table will have all age groups stacked vertically with:
    - A header row showing the age group name
    - The data for that age group
    - One blank row separator between groups
    """
    if not age_groups_processed:
        return {}
    
    first_age_group = age_groups_processed[0]
    table_names = list(all_age_group_dfs[first_age_group].keys())
    
    stacked_dfs = {}
    
    for table_name in table_names:
        stacked_rows = []
        
        for i, age_group in enumerate(age_groups_processed):
            if age_group not in all_age_group_dfs:
                continue
            
            df = all_age_group_dfs[age_group].get(table_name)
            if df is None:
                continue
            
            # Add age group header row
            header_row = {col: '' for col in df.columns}
            first_col = df.columns[0]
            header_row[first_col] = f'=== {age_group} ==='
            stacked_rows.append(header_row)
            
            # Add the data rows
            for _, row in df.iterrows():
                stacked_rows.append(row.to_dict())
            
            # Add ONE blank separator row (except after last group)
            if i < len(age_groups_processed) - 1:
                blank_row = {col: '' for col in df.columns}
                stacked_rows.append(blank_row)
        
        if stacked_rows:
            first_df = all_age_group_dfs[age_groups_processed[0]][table_name]
            stacked_dfs[table_name] = pd.DataFrame(stacked_rows, columns=first_df.columns)
    
    return stacked_dfs


def print_summary_report(all_results, year_range):
    """Print a comprehensive summary report to console."""
    print("\n" + "=" * 80)
    print("DEATH CERTIFICATE UDS ANALYSIS - SUMMARY REPORT (v14)")
    print("=" * 80)
    
    start_year, end_year = year_range
    if start_year:
        print(f"Year Range: {start_year}-{end_year}")
    else:
        print("Year Range: All available years")
    
    print()
    
    for age_group, results in all_results.items():
        rUDS_results = results['rUDS']
        eUDS_results = results['eUDS']
        comp_results = results['comp']
        
        print(f"--- {age_group} ---")
        print(f"  Total Deaths: {rUDS_results['total_count']:,.0f}")
        print(f"  Records Processed: {rUDS_results['records_processed']:,}")
        print(f"  Concordance Rate (r1==e1): {comp_results['concordance_rate']:.1%}")
        print()
        
        # Top 5 by First position for rUDS
        rUDS_first = [(l, rUDS_results['letter_stats'][l]['first']) 
                      for l in ALLOWED_LETTERS]
        rUDS_first.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Top 5 rUDS First Letters:")
        for letter, count in rUDS_first[:5]:
            pct = 100.0 * count / rUDS_results['total_count'] if rUDS_results['total_count'] > 0 else 0
            print(f"    {letter}: {count:>12,} ({pct:5.1f}%)")
        
        # Top 5 by W3 (Dobson) for rUDS
        rUDS_w3 = [(l, rUDS_results['letter_stats'][l]['w3']) 
                   for l in ALLOWED_LETTERS]
        rUDS_w3.sort(key=lambda x: x[1], reverse=True)
        
        print(f"  Top 5 rUDS by W3 (Dobson):")
        for letter, w3 in rUDS_w3[:5]:
            pct = 100.0 * w3 / rUDS_results['total_count'] if rUDS_results['total_count'] > 0 else 0
            print(f"    {letter}: {w3:>14,.1f} ({pct:5.1f}%)")
        
        print()


def save_results_to_excel(output_dfs, filename, year_range, age_groups_processed, x_matrix_source):
    """Save all output DataFrames to an Excel workbook."""
    
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for sheet_name, df in output_dfs.items():
                safe_name = sheet_name[:31]
                df.to_excel(writer, sheet_name=safe_name, index=False)
        
        print(f"[OK] Excel workbook saved: {filename}")
        
    except Exception as e:
        print(f"[ERROR] Failed to save Excel file: {e}")
        return
    
    # Also save individual CSVs
    csv_dir = filename.replace('.xlsx', '_csv')
    os.makedirs(csv_dir, exist_ok=True)
    
    for sheet_name, df in output_dfs.items():
        csv_file = os.path.join(csv_dir, f"{sheet_name}.csv")
        df.to_csv(csv_file, index=False)
    
    print(f"[OK] CSV files saved to: {csv_dir}/")
    
    # Save metadata
    meta_file = filename.replace('.xlsx', '_metadata.txt')
    with open(meta_file, 'w') as f:
        f.write("UDS ANALYSIS RESULTS - VERSION 14\n")
        f.write("=" * 50 + "\n\n")
        
        start_year, end_year = year_range
        if start_year:
            f.write(f"Year Range: {start_year}-{end_year}\n")
        else:
            f.write("Year Range: All available years\n")
        
        f.write(f"Age Groups: {', '.join(age_groups_processed)}\n\n")
        
        f.write("WEIGHTING SCHEMES:\n")
        f.write("  W1: UCoD=0.5, CCoDs share 0.5 equally\n")
        f.write("  W2: All causes share weight equally (1/Ls)\n")
        f.write("  W3: Dobson data-driven (based on co-occurrence patterns)\n")
        f.write("      Reference: Dobson et al. BMC Med Res Methodol (2023) 23:83\n")
        f.write(f"      x_uc matrix source: {x_matrix_source}\n\n")
        
        f.write("TRANSITION MATRICES:\n")
        f.write("  transition_matrix: Rows=Entity First, Cols=Record First\n")
        f.write("    -> 'What did physician's code BECOME?'\n")
        f.write("  trans_Record_to_Entity: Rows=Record First, Cols=Entity First\n")
        f.write("    -> 'What was ORIGINAL code for this output?'\n\n")
        
        f.write("x_uc MATRIX:\n")
        f.write("  rUDS_x_uc_matrix: Co-occurrence proportions for Record Axis\n")
        f.write("  eUDS_x_uc_matrix: Co-occurrence proportions for Entity Axis\n")
        f.write("  x_uc = proportion of deaths with UCoD=u that have CCoD=c\n\n")
        
        f.write("SHEET LIST:\n")
        for i, sheet_name in enumerate(output_dfs.keys(), 1):
            f.write(f"  {i:2d}. {sheet_name}\n")
    
    print(f"[OK] Metadata saved to {meta_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("=" * 80)
    print("DEATH CERTIFICATE UDS ANALYSIS SCRIPT - VERSION 14")
    print("With Dobson Data-Driven Weighting (W3)")
    print("=" * 80)
    print()
    
    # Parse arguments
    args = parse_arguments()
    
    # Parse year range
    start_year, end_year = parse_year_range(args.years)
    
    # Set output filename
    if args.output is None:
        output_filename = get_default_output_filename((start_year, end_year))
    else:
        output_filename = args.output
    
    # Load external x_uc matrix if provided
    external_x_matrix = None
    x_matrix_source = "computed from input data"
    
    if args.xmatrix:
        if not os.path.exists(args.xmatrix):
            print(f"ERROR: x_uc matrix file '{args.xmatrix}' not found!")
            sys.exit(1)
        
        print(f"Loading external x_uc matrix from: {args.xmatrix}")
        external_x_matrix = load_x_matrix_from_csv(args.xmatrix)
        x_matrix_source = f"external file: {args.xmatrix}"
        
        # Validate
        valid, errors = validate_x_matrix(external_x_matrix)
        if not valid:
            print("[WARNING] x_uc matrix validation issues:")
            for error in errors[:5]:
                print(f"  - {error}")
        else:
            print("[OK] External x_uc matrix loaded and validated")
    
    # Determine input source
    if args.filename:
        if not os.path.exists(args.filename):
            print(f"ERROR: Input file '{args.filename}' not found!")
            sys.exit(1)
        input_file = args.filename
        print(f"Reading input file: {input_file}")
    elif not sys.stdin.isatty():
        input_file = None
        print("Reading input from stdin...")
    else:
        print("ERROR: No input file specified!")
        print("Usage: python analyze_uds_v14.py <filename> [--years START-END] [--xmatrix x_uc.csv]")
        sys.exit(1)
    
    # Read the CSV data
    try:
        if input_file:
            df = pd.read_csv(input_file, keep_default_na=False, na_values=[''])
        else:
            df = pd.read_csv(sys.stdin, keep_default_na=False, na_values=[''])
        
        print(f"[OK] Successfully read {len(df):,} records")
        
    except Exception as e:
        print(f"ERROR reading input: {e}")
        sys.exit(1)
    
    # Find year and age_group columns
    year_col = find_column(df, YEAR_COLUMNS)
    age_col = find_column(df, AGE_COLUMNS)
    
    has_year_age = year_col is not None and age_col is not None
    
    if not has_year_age:
        print("[WARNING] Year and/or age_group columns not found!")
        print("         Will analyze ALL data as a single group.")
        year_col = '_year_dummy'
        age_col = '_age_dummy'
        df[year_col] = 0
        df[age_col] = 'ALL'
    else:
        print(f"[OK] Found year column: '{year_col}'")
        print(f"[OK] Found age_group column: '{age_col}'")
    
    # Validate input data
    print("\nValidating data...")
    valid, errors = validate_input_data(df, year_col, age_col)
    
    if not valid:
        print("[FAIL] Validation FAILED:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    print("[OK] Data validation passed")
    
    # Clean data
    initial_rows = len(df)
    df = df.dropna(subset=['rUDS', 'eUDS']).reset_index(drop=True)
    if len(df) < initial_rows:
        print(f"[WARNING] Removed {initial_rows - len(df):,} rows with missing values")
    
    # Filter by year range
    if start_year is not None and has_year_age:
        original_count = len(df)
        df = df[(df[year_col] >= start_year) & (df[year_col] <= end_year)].copy()
        df.reset_index(drop=True, inplace=True)
        print(f"[OK] Filtered to years {start_year}-{end_year}: {len(df):,} records (from {original_count:,})")
        
        if len(df) == 0:
            print("ERROR: No data remaining after year filter!")
            sys.exit(1)
    
    # Show data summary
    print(f"\nData Summary:")
    print(f"  Total records: {len(df):,}")
    print(f"  Total deaths: {df['Count'].sum():,.0f}")
    if has_year_age:
        print(f"  Year range in data: {df[year_col].min()} - {df[year_col].max()}")
        print(f"  Age groups in data: {sorted(df[age_col].unique())}")
    
    # Validate strings
    valid, errors = validate_string_lengths(df)
    if not valid:
        print("[WARNING] String validation issues found:")
        for error in errors[:5]:
            print(f"  - {error}")
    
    print()
    
    # ========================================================================
    # ANALYSIS PHASE - BY AGE GROUP
    # ========================================================================
    
    print("ANALYSIS PHASE")
    print("-" * 80)
    
    all_age_group_dfs = {}
    all_results = {}
    age_groups_processed = []
    
    for age_group in AGE_GROUPS:
        if age_group == 'ALL':
            df_subset = df.copy()
        else:
            df_subset = df[df[age_col] == age_group].copy()
        
        if len(df_subset) == 0:
            print(f"  [SKIP] {age_group}: No data")
            continue
        
        age_groups_processed.append(age_group)
        
        print(f"\nProcessing {age_group}...")
        print(f"  Records: {len(df_subset):,}, Deaths: {df_subset['Count'].sum():,.0f}")
        
        # STEP 1: Get x_uc matrices (external or computed)
        if external_x_matrix is not None:
            # Use external matrix for both rUDS and eUDS
            x_matrix_rUDS = external_x_matrix
            x_matrix_eUDS = external_x_matrix
            N_u_rUDS = {}  # Not available from external
            N_u_eUDS = {}
            print(f"  Using external x_uc matrix")
        else:
            # Compute from data
            print(f"  Computing Dobson x_uc matrix for rUDS...")
            x_matrix_rUDS, N_u_rUDS = compute_dobson_x_matrix(df_subset, 'rUDS')
            
            print(f"  Computing Dobson x_uc matrix for eUDS...")
            x_matrix_eUDS, N_u_eUDS = compute_dobson_x_matrix(df_subset, 'eUDS')
        
        # STEP 2: Run Level A analysis with W3 weights
        print(f"  Analyzing rUDS (with W3)...")
        rUDS_results = analyze_individual_uds(df_subset, 'rUDS', x_matrix_rUDS)
        
        print(f"  Analyzing eUDS (with W3)...")
        eUDS_results = analyze_individual_uds(df_subset, 'eUDS', x_matrix_eUDS)
        
        # Validate Level A
        valid, errors = validate_level_a_results(rUDS_results)
        if not valid:
            print(f"  [WARNING] rUDS Level A validation issues:")
            for error in errors[:3]:
                print(f"    - {error}")
        
        valid, errors = validate_level_a_results(eUDS_results)
        if not valid:
            print(f"  [WARNING] eUDS Level A validation issues:")
            for error in errors[:3]:
                print(f"    - {error}")
        
        # Run Level B analysis (BIDIRECTIONAL)
        print(f"  Comparative analysis (bidirectional)...")
        comp_results = analyze_comparative(df_subset)
        
        # Validate Level B
        valid, errors = validate_level_b_results(comp_results, df_subset['Count'].sum())
        if not valid:
            print(f"  [WARNING] Level B validation issues:")
            for error in errors[:3]:
                print(f"    - {error}")
        
        # Store results
        all_results[age_group] = {
            'rUDS': rUDS_results,
            'eUDS': eUDS_results,
            'comp': comp_results
        }
        
        # Create output DataFrames for this age group
        output_dfs = create_output_dataframes_for_age_group(
            rUDS_results, eUDS_results, comp_results,
            x_matrix_rUDS, x_matrix_eUDS,
            N_u_rUDS, N_u_eUDS,
            age_group
        )
        
        all_age_group_dfs[age_group] = output_dfs
        
        print(f"  [OK] {age_group}: {len(output_dfs)} tables created")
    
    # Stack all age groups into single tables
    print(f"\nStacking age groups into compact format...")
    stacked_output_dfs = stack_age_group_tables(all_age_group_dfs, age_groups_processed)
    print(f"[OK] Created {len(stacked_output_dfs)} stacked tables")
    
    # Print summary report
    print_summary_report(all_results, (start_year, end_year))
    
    # ========================================================================
    # OUTPUT PHASE
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("OUTPUT PHASE")
    print("-" * 80)
    
    save_results_to_excel(stacked_output_dfs, output_filename, (start_year, end_year), 
                          age_groups_processed, x_matrix_source)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print()
    print("Output files created:")
    print(f"  - {output_filename} (Excel workbook with {len(stacked_output_dfs)} sheets)")
    print(f"  - {output_filename.replace('.xlsx', '_csv')}/ (CSV files)")
    print(f"  - {output_filename.replace('.xlsx', '_metadata.txt')} (Metadata)")
    print()
    print(f"Age groups (stacked on each sheet): {', '.join(age_groups_processed)}")
    if start_year:
        print(f"Year range: {start_year}-{end_year}")
    print()
    print("NEW in v14: W3 (Dobson) data-driven weighting")
    print(f"  x_uc matrix source: {x_matrix_source}")
    print("  Reference: Dobson et al. BMC Med Res Methodol (2023) 23:83")
    print()


if __name__ == "__main__":
    main()
