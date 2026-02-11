#!/usr/bin/env python3
"""
Analyze ICD-10 codes for underlying cause of death (UCOD) preference.

For each code, compute:
- N_ucod: times it appears as underlying cause (first position)
- N_total: times it appears anywhere
- N_contrib: times as contributing cause only
- Ratio: N_ucod / N_total (underlying cause preference)

Input: CSV with Count,rec_axis where rec_axis has space-separated ICD codes
       First code is UCOD, rest are contributing causes.
"""

import sys
from collections import defaultdict

try:
    import simple_icd_10 as icd
    HAS_ICD = True
except ImportError:
    HAS_ICD = False
    print("Warning: simple_icd_10 not installed, descriptions will be empty", file=sys.stderr)


def get_description(code):
    if not HAS_ICD:
        return ""
    try:
        desc = icd.get_description(code)
        return desc if desc else ""
    except:
        return ""


def main():
    if len(sys.argv) < 2:
        print("Usage: python ucod_preference.py <input_file> [min_total]", file=sys.stderr)
        print("  min_total: minimum N_total to include in output (default: 100)", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    min_total = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Count UCOD and total appearances for each code
    ucod_count = defaultdict(int)
    total_count = defaultdict(int)
    
    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('Count,'):
                continue
            
            parts = line.split(',', 1)
            if len(parts) < 2:
                continue
            
            try:
                count = int(parts[0])
            except ValueError:
                continue
            
            codes = parts[1].split()
            if not codes:
                continue
            
            # First code is UCOD
            ucod = codes[0]
            ucod_count[ucod] += count
            
            # All codes count toward total (including UCOD)
            for code in codes:
                total_count[code] += count
            
            if line_num % 100000 == 0:
                print(f"Processed {line_num} lines...", file=sys.stderr)
    
    print(f"Processed {line_num} total lines", file=sys.stderr)
    print(f"Found {len(total_count)} unique codes", file=sys.stderr)
    
    # Build results
    results = []
    for code in total_count:
        n_total = total_count[code]
        n_ucod = ucod_count.get(code, 0)
        n_contrib = n_total - n_ucod
        
        if n_total >= min_total:
            ratio = n_ucod / n_total if n_total > 0 else 0
            desc = get_description(code)
            results.append((code, n_ucod, n_total, n_contrib, ratio, desc))
    
    print(f"Output {len(results)} codes with N_total >= {min_total}", file=sys.stderr)
    
    # Sort by ratio descending (most likely to be UCOD first)
    results.sort(key=lambda x: (-x[4], -x[2]))  # by ratio desc, then by total desc
    
    # Output CSV
    print("Code,N_ucod,N_total,N_contrib,Ratio,Description")
    for code, n_ucod, n_total, n_contrib, ratio, desc in results:
        # Escape description for CSV
        if ',' in desc or '"' in desc:
            desc = '"' + desc.replace('"', '""') + '"'
        print(f"{code},{n_ucod},{n_total},{n_contrib},{ratio:.4f},{desc}")


if __name__ == "__main__":
    main()
