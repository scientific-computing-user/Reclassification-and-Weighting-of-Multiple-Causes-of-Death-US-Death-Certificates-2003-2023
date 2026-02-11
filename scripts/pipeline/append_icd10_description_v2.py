#!/usr/bin/env python3
"""
Append ICD-10 description to lines containing ICD-10 codes.

Uses the simple-icd-10 library for standard ICD-10 descriptions.

Usage:
    cat input.csv | python3 append_icd10_description.py --col 2 > output.csv
    
    --col N         Column number (1-based) containing ICD-10 code
    --sep CHAR      Field separator (default: comma)
    --header        First line is header (pass through, append "ICD10_Description")
    
Input:  Read from STDIN
Output: Write to STDOUT with description appended as last column

Requires: pip install simple-icd-10
"""

import sys
import argparse
import csv
from io import StringIO

try:
    import simple_icd_10 as icd
except ImportError:
    print("[ERROR] simple-icd-10 library not installed.", file=sys.stderr)
    print("[ERROR] Install with: pip install simple-icd-10", file=sys.stderr)
    sys.exit(1)

def get_description(code):
    """Get ICD-10 description for a code."""
    # Normalize code
    code = code.strip().upper().replace('.', '')
    
    # Try exact match first
    try:
        desc = icd.get_description(code)
        if desc:
            return desc
    except:
        pass
    
    # Try with dot notation (e.g., A00.1)
    if len(code) > 3:
        dotted = code[:3] + '.' + code[3:]
        try:
            desc = icd.get_description(dotted)
            if desc:
                return desc
        except:
            pass
    
    # Try truncating to find parent code
    trial = code
    while len(trial) > 2:
        try:
            desc = icd.get_description(trial)
            if desc:
                return desc
        except:
            pass
        trial = trial[:-1]
    
    return ''

def main():
    parser = argparse.ArgumentParser(
        description='Append ICD-10 description to lines from STDIN',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--col', type=int, required=True,
                        help='Column number (1-based) containing ICD-10 code')
    parser.add_argument('--sep', default=',',
                        help='Field separator (default: comma)')
    parser.add_argument('--header', action='store_true',
                        help='First line is header')
    
    args = parser.parse_args()
    
    # Convert to 0-based index
    col_idx = args.col - 1
    if col_idx < 0:
        print('[ERROR] Column number must be >= 1', file=sys.stderr)
        sys.exit(1)
    
    # Process STDIN
    line_num = 0
    matched = 0
    unmatched = 0
    
    for line in sys.stdin:
        line_num += 1
        line = line.rstrip('\r\n')
        
        # Handle header
        if args.header and line_num == 1:
            print(f'{line}{args.sep}ICD10_Description')
            continue
        
        # Parse line
        if args.sep == ',':
            # Use CSV parser for comma-separated (handles quotes)
            reader = csv.reader(StringIO(line))
            try:
                fields = next(reader)
            except StopIteration:
                fields = []
        else:
            fields = line.split(args.sep)
        
        # Get ICD-10 code from specified column
        description = ''
        if col_idx < len(fields):
            code = fields[col_idx].strip()
            if code:
                description = get_description(code)
                if description:
                    matched += 1
                else:
                    unmatched += 1
            else:
                unmatched += 1
        else:
            unmatched += 1
        
        # Escape description if it contains separator or quotes
        if args.sep == ',' and (',' in description or '"' in description):
            description = '"' + description.replace('"', '""') + '"'
        
        # Output line with description appended
        print(f'{line}{args.sep}{description}')
    
    # Summary to stderr
    print(f'[INFO] Processed {line_num:,} lines: {matched:,} matched, {unmatched:,} unmatched', 
          file=sys.stderr)

if __name__ == '__main__':
    main()
