#!/usr/bin/env python3
"""
Add ICD-10-CM descriptions to the sorted disease category file.
"""

import sys
import simple_icd_10 as icd

def get_icd_description(code):
    """Get ICD-10-CM description for a code, handling edge cases."""
    try:
        desc = icd.get_description(code)
        if desc:
            return desc
    except Exception:
        pass
    return ""

def main():
    if len(sys.argv) < 3:
        print("Usage: python add_icd_descriptions.py <input_file> <threshold>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    threshold = int(sys.argv[2])
    
    # Output header
    print("Count,r1,rUCOD,Description")
    
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split(',')
            if len(parts) < 3:
                continue
            
            try:
                count = int(parts[0])
            except ValueError:
                # Skip header line
                continue
            r1 = parts[1]
            rucod = parts[2]
            
            # Add description if count >= threshold
            if count >= threshold:
                desc = get_icd_description(rucod)
                # Escape quotes and wrap in quotes if contains comma
                if ',' in desc or '"' in desc:
                    desc = '"' + desc.replace('"', '""') + '"'
            else:
                desc = ""
            
            print(f"{count},{r1},{rucod},{desc}")

if __name__ == "__main__":
    main()
