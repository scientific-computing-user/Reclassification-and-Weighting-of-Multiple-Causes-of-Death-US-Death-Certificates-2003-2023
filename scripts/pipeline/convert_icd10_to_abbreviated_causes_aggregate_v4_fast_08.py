# convert_icd10_to_abbreviated_causes_aggregate_v4_fast_08.py
# ------------------------------------------------------------
"""
Full-featured ICD-10 death certificate aggregation with speed optimizations.
Combines v3_6P_14 functionality with v4_fast_03 performance.

Features:
- Full I/O/G output format with all original columns
- matched_icd10_codes.csv output with counts by position:
  - Position 2: UCOD on Record Axis
  - Position 3: UCOD on Entity Axis
  - Position 4: Contributing causes on Record Axis
  - Position 5: Contributing causes on Entity Axis
- --use-ent-ucod2 flag for sensitivity analysis
- --debug flag for detailed output
- Hash lookup for O(1) ICD-10 conversion
- Optimized aggregation with cached weights/buckets

Weighting schemes:
- W0: First only (100% to first cause)
- W1: Half-plus-share (50% first, 50% split among rest)
- W2: Equal share on unique causes (UDS)
- W2A: Equal share on all causes including repeats (ADS)
"""

import csv, sys, argparse, time
from collections import Counter, defaultdict
from pathlib import Path

# ============================================================ #
# 1. Load lookup tables                                        #
# ============================================================ #
def load_lookup(lookup_file):
    """Load pre-computed ICD10 -> DL hash lookup with rule info.
    Returns (lookup_dict, rule_info_dict) where:
    - lookup_dict: {icd_code: disease_letter}
    - rule_info_dict: {icd_code: (rule_pattern, cause, rule_name)}
    """
    lookup = {}
    rule_info = {}
    with open(lookup_file) as fh:
        header = next(fh).strip()
        has_rule_info = 'Rule' in header
        for line in fh:
            parts = line.rstrip('\r\n').split(',', 4)  # max 5 parts
            if len(parts) >= 2:
                icd = parts[0].strip().upper().replace('.', '')
                dl = parts[1].strip()
                if dl and dl != '?':
                    lookup[icd] = dl
                    if has_rule_info and len(parts) >= 4:
                        rule = parts[2].strip()
                        cause = parts[3].strip()
                        name = parts[4].strip().strip('"') if len(parts) > 4 else ''
                        rule_info[icd] = (rule, cause, name)
                    else:
                        rule_info[icd] = (icd, dl, '')
    print(f'[INFO] Loaded {len(lookup):,} ICD-10 codes into hash lookup')
    return lookup, rule_info

def load_rules(file):
    """Load fallback mapping rules."""
    exact, ranges = {}, []
    with open(file) as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip('\r\n').split(',')
            if len(parts) >= 2:
                icd = parts[0].strip().upper().replace(' ', '').replace('.', '')
                cause = parts[1].strip()
                if '-' in icd:
                    a, b = icd.split('-')
                    ranges.append((a, b, cause))
                else:
                    exact[icd] = cause
    print(f'[INFO] Loaded {len(exact)} exact, {len(ranges)} range rules for fallback')
    return exact, ranges

# ============================================================ #
# 2. Letter lookup with tracking                               #
# ============================================================ #
ABBREV = {
    'Alcohol-related': 'A', 'COVID-19': 'V', 'Cancer': 'C', 'Circulatory': 'B',
    'Digestive': 'D', 'Drug poisoning': 'P', 'Endocrine': 'E', 'Homicide': 'H',
    'Other external': 'X', 'Other natural': 'N', 'Respiratory': 'R',
    'Suicide': 'S', 'Transport': 'T', 'Unknown': 'U', 'Falls': 'F',
}

# Known typo corrections
TYPO_FIXES = {
    'OO83': 'O083',  # Record 41918302: letter O instead of zero
}

def get_letter_and_track(code, lookup, rule_info, exact_rules, range_rules, unknown, counts, position):
    """Get disease letter and track match info. Returns (letter, is_unknown).
    
    position: 2=UCOD_Rec, 3=UCOD_Ent, 4=Contrib_Rec, 5=Contrib_Ent
    """
    raw = code.strip().upper().replace('.', '')
    
    # Apply typo corrections
    if raw in TYPO_FIXES:
        raw = TYPO_FIXES[raw]
    
    # Hash lookup first
    if raw in lookup:
        letter = lookup[raw]
        if raw in rule_info:
            rule, cause, name = rule_info[raw]
            unknown.setdefault(raw, ('hash', rule, cause, name))
        else:
            unknown.setdefault(raw, ('hash', raw, letter, ''))
        counts[raw][position] += 1
        return letter, False
    
    # Fallback to rules
    trial = raw
    while trial:
        if trial in exact_rules:
            cause = exact_rules[trial]
            letter = ABBREV.get(cause, 'U')
            unknown.setdefault(raw, ('exact', trial, cause, ''))
            counts[raw][position] += 1
            return letter, False
        for a, b, cause in range_rules:
            if a <= trial <= b:
                letter = ABBREV.get(cause, 'U')
                unknown.setdefault(raw, ('range', f'{a}-{b}', cause, ''))
                counts[raw][position] += 1
                return letter, False
        trial = trial[:-1]
    
    # T/S/V fallback
    if raw and raw[0] in 'TSV':
        letter = raw[0]
        unknown.setdefault(raw, ('fallback', 'TSV_fallback', raw[0], ''))
        counts[raw][position] += 1
        return letter, False
    else:
        letter = 'U'
        unknown.setdefault(raw, ('none', 'none', 'Unknown', ''))
        counts[raw][position] += 1
        return letter, True  # This is truly unknown

# ============================================================ #
# 3. Global counters                                           #
# ============================================================ #
W_OVER = defaultdict(Counter)
W_YEAR = defaultdict(lambda: defaultdict(Counter))
W_MONTH = defaultdict(lambda: defaultdict(Counter))

P_OVER = defaultdict(lambda: defaultdict(Counter))
P_YEAR = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
P_MONTH = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))

# ============================================================ #
# 4. Cached weight vectors                                     #
# ============================================================ #
WEIGHT_CACHE = {}
def get_weights(n):
    if n not in WEIGHT_CACHE:
        if n == 0:
            WEIGHT_CACHE[n] = ((), (), ())
        else:
            w0 = (1.0,) + (0.0,) * (n-1)
            w1 = (1.0,) if n == 1 else (0.5,) + (0.5/(n-1),) * (n-1)
            w2 = (1.0/n,) * n
            WEIGHT_CACHE[n] = (w0, w1, w2)
    return WEIGHT_CACHE[n]

for i in range(30):
    get_weights(i)

# ============================================================ #
# 5. Cached age buckets                                        #
# ============================================================ #
BUCKET_CACHE = {}
def get_buckets(age_grp):
    if age_grp in BUCKET_CACHE:
        return BUCKET_CACHE[age_grp]
    if not age_grp:
        result = ('UNKNOWN', 'ALL', 'UNKNOWNAGE')
    elif '+' in age_grp:
        result = (age_grp, 'ALL', 'PLUS65')
    elif '-' in age_grp:
        try:
            lo, hi = age_grp.split('-')
            mid = (int(lo) + int(hi)) // 2
            result = (age_grp, 'ALL', 'UNDER65' if mid < 65 else 'PLUS65')
        except:
            result = (age_grp, 'ALL', 'UNKNOWNAGE')
    else:
        result = (age_grp, 'ALL', 'UNKNOWNAGE')
    BUCKET_CACHE[age_grp] = result
    return result

# ============================================================ #
# 6. Unique letters helper                                     #
# ============================================================ #
def unique_letters(letters):
    seen = set()
    result = []
    for c in letters:
        if c not in seen:
            seen.add(c)
            result.append(c)
    return result

# ============================================================ #
# 7. Apply select_ucod transformations                         #
# ============================================================ #
def apply_select_ucod_transforms(ucod, ent_axis_list, ent_lines_list, ent_seq_list, eDLs_list):
    """Apply transformations from select_ucod. Works with lists directly."""
    
    lines_int = []
    for x in ent_lines_list:
        try:
            lines_int.append(int(x))
        except:
            pass
    
    # Reorder: reverse items where ent_lines <= 5
    if lines_int and len(lines_int) == len(ent_axis_list):
        part1_indices = [i for i, v in enumerate(lines_int) if v <= 5]
        if part1_indices:
            reversed_indices = part1_indices[::-1]
            new_axis = ent_axis_list[:]
            new_lines = ent_lines_list[:]
            new_seq = ent_seq_list[:] if ent_seq_list else []
            new_dls = eDLs_list[:] if eDLs_list else []
            
            for new_pos, old_pos in zip(part1_indices, reversed_indices):
                new_axis[new_pos] = ent_axis_list[old_pos]
                new_lines[new_pos] = ent_lines_list[old_pos]
                if ent_seq_list and old_pos < len(ent_seq_list) and new_pos < len(new_seq):
                    new_seq[new_pos] = ent_seq_list[old_pos]
                if eDLs_list and old_pos < len(eDLs_list) and new_pos < len(new_dls):
                    new_dls[new_pos] = eDLs_list[old_pos]
            
            ent_axis_list = new_axis
            ent_lines_list = new_lines
            ent_seq_list = new_seq
            eDLs_list = new_dls
            lines_int = [int(x) for x in ent_lines_list]
    
    # sel_code, sel_pos
    sel_code, sel_pos = '', ''
    if lines_int and ent_axis_list:
        for i, v in enumerate(lines_int):
            if v < 6:
                sel_code = ent_axis_list[i]
                sel_pos = str(i + 1)
                break
    
    # ucod_pos
    if sel_code == ucod:
        ucod_pos = '-1P'
    elif ucod in ent_axis_list:
        ucod_pos = str(ent_axis_list.index(ucod) + 1) + 'P'
    else:
        ucod_pos = '0P'
    
    # dup1
    dup1 = 0
    if lines_int:
        first_val = lines_int[0]
        for v in lines_int:
            if v == first_val:
                dup1 += 1
            else:
                break
    
    # ent_ucod, ent_ucod2, ent_ucod3
    ent_ucod = ent_axis_list[0] if dup1 >= 1 and len(ent_axis_list) >= 1 else ''
    ent_ucod2 = ent_axis_list[1] if dup1 >= 2 and len(ent_axis_list) >= 2 else ''
    ent_ucod3 = ent_axis_list[2] if dup1 >= 3 and len(ent_axis_list) >= 3 else ''
    
    return {
        'ent_axis': ent_axis_list,
        'ent_lines': ent_lines_list,
        'ent_seq': ent_seq_list,
        'eADS': eDLs_list,
        'ent_ucod': ent_ucod,
        'ent_ucod2': ent_ucod2,
        'ent_ucod3': ent_ucod3,
        'sel_code': sel_code,
        'sel_pos': sel_pos,
        'ucod_pos': ucod_pos,
        'dup1': dup1
    }

# ============================================================ #
# 8. Process death record for aggregation                      #
# ============================================================ #
def process_death_record(uds, ads, year, month, age_grp, sex, race):
    """Accumulate weighted stats."""
    if not uds:
        return
    
    sex = sex or 'UNKSEX'
    race = race or 'UNKRACE'
    yr_i = int(year)
    mon_idx = (yr_i - 2003) * 12 + (int(month) - 1)
    buckets = get_buckets(age_grp)
    n_uds = len(uds)
    
    w0, w1, w2 = get_weights(n_uds)
    
    # W0, W1, W2 (use UDS)
    for sid, wvec in enumerate((w0, w1, w2)):
        for buck in buckets:
            key = (sid, buck, sex, race)
            c_over = W_OVER[key]
            c_year = W_YEAR[key][yr_i]
            c_mon = W_MONTH[key][mon_idx]
            for i in range(n_uds):
                wt = wvec[i]
                if wt:
                    ltr = uds[i]
                    c_over[ltr] += wt
                    c_year[ltr] += wt
                    c_mon[ltr] += wt
    
    # W2A (use ADS)
    if ads:
        n_ads = len(ads)
        wt = 1.0 / n_ads
        for buck in buckets:
            key = (3, buck, sex, race)
            c_over = W_OVER[key]
            c_year = W_YEAR[key][yr_i]
            c_mon = W_MONTH[key][mon_idx]
            for ltr in ads:
                c_over[ltr] += wt
                c_year[ltr] += wt
                c_mon[ltr] += wt
    
    # Position tables (use UDS)
    for buck in buckets:
        pkey = (buck, sex, race)
        for pos, ltr in enumerate(uds[:8], start=1):
            P_OVER[pkey][pos][ltr] += 1
            P_YEAR[pkey][yr_i][pos][ltr] += 1
            P_MONTH[pkey][mon_idx][pos][ltr] += 1

# ============================================================ #
# 9. Main processing function                                  #
# ============================================================ #
def process_file(input_csv, lookup, rule_info, exact_rules, range_rules, use_ent_ucod2=False,
                 output_csv='converted_with_original.csv', debug=False):
    
    match_out = 'matched_icd10_codes.csv'
    unknown = {}
    counts = defaultdict(lambda: Counter())  # counts[icd_code][position] = count
    
    # Read header using csv.reader for proper quote handling
    with open(input_csv, newline='') as fh:
        rdr = csv.reader(fh)
        hdr = next(rdr)
    
    col = {name: i for i, name in enumerate(hdr)}
    
    # Build output header
    hdr_output = []
    for c in hdr:
        if c == 'ent_imcod':
            hdr_output.append('ent_ucod2')
        elif c == 'ent_incod':
            hdr_output.append('ent_ucod3')
        else:
            hdr_output.append(c)
    
    out_header = 'I/O,rADS,rUDS,eADS,eUDS,' + ','.join(hdr_output) + ',sel_code,sel_pos,ucod_pos,dup1\n'
    
    # Column indices
    idx_year = col.get('year', -1)
    idx_month = col.get('month', -1)
    idx_age_group = col.get('age_group', -1)
    idx_sex = col.get('sex', -1)
    idx_ucod = col.get('ucod', -1)
    idx_rec_axis = col.get('rec_axis', -1)
    idx_ent_axis = col.get('ent_axis', -1)
    idx_ent_lines = col.get('ent_lines', -1)
    idx_ent_seq = col.get('ent_seq', -1)
    idx_race = col.get('race6_map', col.get('race7_map', col.get('race', -1)))
    
    start = time.time()
    total = 0
    eUDS_used = rUDS_used = differ_count = ucod2_swap_count = 0
    unknown_lines = []  # Track lines with unknown codes
    
    with open(input_csv, newline='') as fh_in, open(output_csv, 'w') as fh_out:
        rdr = csv.reader(fh_in)
        next(rdr)  # skip header
        fh_out.write(out_header)
        
        buffer = []
        
        for row in rdr:
            year = row[idx_year] if idx_year >= 0 else ''
            if year.lower() == 'year':
                continue
            
            total += 1
            
            if total % 100_000 == 0:
                elapsed = time.time() - start
                print(f"Processed {total:,} rows in {elapsed:,.1f}s ({total/elapsed:,.0f} rows/sec)",
                      file=sys.stderr)
            
            # Extract fields
            month = row[idx_month] if idx_month >= 0 else ''
            age_grp = row[idx_age_group] if idx_age_group >= 0 else ''
            sex = row[idx_sex] if idx_sex >= 0 else ''
            race = row[idx_race] if idx_race >= 0 and idx_race < len(row) else ''
            ucod = row[idx_ucod].strip() if idx_ucod >= 0 else ''
            rec_axis_str = row[idx_rec_axis] if idx_rec_axis >= 0 else ''
            ent_axis_str = row[idx_ent_axis] if idx_ent_axis >= 0 else ''
            ent_lines_str = row[idx_ent_lines] if idx_ent_lines >= 0 else ''
            ent_seq_str = row[idx_ent_seq] if idx_ent_seq >= 0 else ''
            
            # Track unknown codes for this line
            line_unknown_codes = []
            
            # Parse rec_axis and reorder with ucod first
            rec_codes = rec_axis_str.split()
            if ucod and ucod in rec_codes:
                rec_codes.remove(ucod)
                rec_codes.insert(0, ucod)
            
            # Convert rec_axis to letters
            rec_letters = []
            rec_pairs = []
            for i, code in enumerate(rec_codes):
                # Position 2 = UCOD on rec_axis (first), Position 4 = Contributing on rec_axis
                pos = 2 if i == 0 else 4
                ltr, is_unknown = get_letter_and_track(code, lookup, rule_info, exact_rules, range_rules, unknown, counts, pos)
                if is_unknown and code not in line_unknown_codes:
                    line_unknown_codes.append(code)
                rec_letters.append(ltr)
                rec_pairs.append(f'{ltr}:{code}')
            
            rADS = ''.join(rec_letters)
            rUDS_list = unique_letters(rec_letters)
            rUDS = ''.join(rUDS_list)
            
            # Parse ent_axis
            orig_ent_axis = ent_axis_str.split() if ent_axis_str else []
            orig_ent_lines = ent_lines_str.split() if ent_lines_str else []
            orig_ent_seq = ent_seq_str.split() if ent_seq_str else []
            
            # Convert ent_axis to letters (preliminary)
            prelim_letters = []
            for i, code in enumerate(orig_ent_axis):
                # Position 3 = UCOD on ent_axis (first), Position 5 = Contributing on ent_axis
                pos = 3 if i == 0 else 5
                ltr, is_unknown = get_letter_and_track(code, lookup, rule_info, exact_rules, range_rules, unknown, counts, pos)
                if is_unknown and code not in line_unknown_codes:
                    line_unknown_codes.append(code)
                prelim_letters.append(ltr)
            
            # Report unknown codes for this line (only once per line)
            if line_unknown_codes:
                orig_line = ','.join(row)
                msg = f"[UNKNOWN] Line {total}: codes {line_unknown_codes} in: {orig_line}"
                print(msg)  # stdout
                print(msg, file=sys.stderr)  # stderr
                unknown_lines.append((total, line_unknown_codes, orig_line))
            
            # Apply select_ucod transforms
            transforms = apply_select_ucod_transforms(
                ucod, orig_ent_axis[:], orig_ent_lines[:], orig_ent_seq[:], prelim_letters[:]
            )
            
            ent_axis_codes = transforms['ent_axis']
            ent_letters = transforms['eADS']
            
            # Handle --use-ent-ucod2
            applied_ucod2_swap = False
            first_line_val = first_code = last_code = None
            
            if orig_ent_lines:
                first_line_val = orig_ent_lines[0]
                for code, line in zip(orig_ent_axis, orig_ent_lines):
                    if line == first_line_val:
                        if first_code is None:
                            first_code = code
                        last_code = code
            
            if use_ent_ucod2 and first_code and last_code and last_code != first_code:
                if last_code in ent_axis_codes:
                    ent_axis_codes.remove(last_code)
                    ent_axis_codes.insert(0, last_code)
                    # Reorder letters too
                    try:
                        idx = orig_ent_axis.index(last_code)
                        if idx < len(ent_letters):
                            ltr = ent_letters.pop(idx)
                            ent_letters.insert(0, ltr)
                    except:
                        pass
                    applied_ucod2_swap = True
                    transforms['ent_ucod2'] = first_code
                    transforms['ent_ucod'] = last_code
            
            eADS = ''.join(ent_letters)
            eUDS_list = unique_letters(ent_letters)
            eUDS = ''.join(eUDS_list)
            
            # Build output row
            out_row = list(row)
            # Ensure row is long enough
            while len(out_row) < len(hdr):
                out_row.append('')
            
            # Update transformed fields
            if idx_rec_axis >= 0:
                out_row[idx_rec_axis] = ' '.join(rec_codes)
            if idx_ent_axis >= 0:
                out_row[idx_ent_axis] = ' '.join(ent_axis_codes)
            if idx_ent_lines >= 0:
                out_row[idx_ent_lines] = ' '.join(transforms['ent_lines'])
            if idx_ent_seq >= 0:
                out_row[idx_ent_seq] = ' '.join(transforms['ent_seq'])
            
            # Handle renamed columns
            for i, c in enumerate(hdr):
                if c == 'ent_ucod':
                    out_row[i] = transforms['ent_ucod']
                elif c == 'ent_imcod':
                    out_row[i] = transforms['ent_ucod2']
                elif c == 'ent_incod':
                    out_row[i] = transforms['ent_ucod3']
            
            updated_line = ','.join(out_row)
            pair_str = ','.join(rec_pairs)
            
            # Build I/O/G lines
            i_line = f'I:,{rADS},{rUDS},{eADS},{eUDS},{updated_line},{transforms["sel_code"]},{transforms["sel_pos"]},{transforms["ucod_pos"]},{transforms["dup1"]}'
            o_line = f'O:,{updated_line}'
            g_line = f'G:,{rADS},{rUDS},{pair_str}'
            
            buffer.append(i_line + '\n' + o_line + '\n' + g_line + '\n')
            
            if len(buffer) >= 5000:
                fh_out.write(''.join(buffer))
                buffer = []
            
            # Debug output (stdout only)
            if debug and applied_ucod2_swap:
                print(f"DEBUG row {total}: REORDERED")
                print(f"  orig ent_axis: {ent_axis_str}")
                print(f"  first_line_val: {first_line_val}")
                print(f"  first_code: {first_code}, last_code: {last_code}")
                print(f"  eADS: {eADS}, eUDS: {eUDS}")
            
            # Aggregation
            if use_ent_ucod2:
                uds_for_agg = eUDS_list
                ads_for_agg = ent_letters
                eUDS_used += 1
                if applied_ucod2_swap:
                    ucod2_swap_count += 1
                if eUDS != rUDS:
                    differ_count += 1
            else:
                uds_for_agg = rUDS_list
                ads_for_agg = rec_letters
                rUDS_used += 1
            
            race_for_agg = race or 'UNKRACE'
            process_death_record(uds_for_agg, ads_for_agg, year, month, age_grp, sex, race_for_agg)
        
        if buffer:
            fh_out.write(''.join(buffer))
    
    # Summary
    elapsed = time.time() - start
    rate = total / elapsed if elapsed > 0 else 0
    print(f"[INFO] Completed {total:,} rows in {elapsed:,.1f}s ({rate:,.0f} rows/sec)")
    
    if use_ent_ucod2:
        print(f"[INFO] --use-ent-ucod2: used eUDS for {eUDS_used} records")
        print(f"[INFO] {ucod2_swap_count} records had ucod2 swap")
        print(f"[INFO] {differ_count} records had eUDS != rUDS")
    else:
        print(f"[INFO] used rUDS for {rUDS_used} records")
    
    # Summary of unknown codes
    if unknown_lines:
        msg = f"[INFO] {len(unknown_lines)} lines with unknown ICD-10 codes"
        print(msg)
        print(msg, file=sys.stderr)
    
    # Write matched_icd10_codes.csv
    with open(match_out, 'w') as fh:
        fh.write('ICD10 Code,UCOD_Rec,UCOD_Ent,Contrib_Rec,Contrib_Ent,Match Type,Rule,Cause,RuleName,Occurrences\n')
        for code in sorted(unknown):
            mtype, rule, cause, name = unknown[code]
            # Escape name if needed
            if ',' in name or '"' in name:
                name = '"' + name.replace('"', '""') + '"'
            c = counts[code]
            total = c[2] + c[3] + c[4] + c[5]
            fh.write(f'{code},{c[2]},{c[3]},{c[4]},{c[5]},{mtype},{rule},{cause},{name},{total}\n')
    print(f"[INFO] wrote {match_out}")
    
    # Write unknown_lines.csv
    unknown_out = 'unknown_lines.csv'
    with open(unknown_out, 'w') as fh:
        fh.write('line_num,unknown_codes,original_line\n')
        for line_num, codes, orig_line in unknown_lines:
            codes_str = ' '.join(codes)
            # Escape quotes in original line
            orig_escaped = orig_line.replace('"', '""')
            fh.write(f'{line_num},"{codes_str}","{orig_escaped}"\n')
    print(f"[INFO] wrote {unknown_out}")

# ============================================================ #
# 10. Write aggregate CSVs                                     #
# ============================================================ #
def write_aggregates(out_dir='output/agg'):
    Path(out_dir).mkdir(exist_ok=True, parents=True)
    
    with open(f'{out_dir}/overall_agg.csv', 'w') as fh:
        fh.write('scheme,bucket,sex,race,letter,weight\n')
        for (sid, buck, sex, race), ctr in W_OVER.items():
            for ltr, wt in ctr.items():
                fh.write(f'{sid+1},{buck},{sex},{race},{ltr},{wt}\n')
    
    with open(f'{out_dir}/year_agg.csv', 'w') as fh:
        fh.write('scheme,bucket,sex,race,year,letter,weight\n')
        for (sid, buck, sex, race), sub in W_YEAR.items():
            for yr, ctr in sub.items():
                for ltr, wt in ctr.items():
                    fh.write(f'{sid+1},{buck},{sex},{race},{yr},{ltr},{wt}\n')
    
    with open(f'{out_dir}/month_idx_agg.csv', 'w') as fh:
        fh.write('scheme,bucket,sex,race,month_idx,letter,weight\n')
        for (sid, buck, sex, race), sub in W_MONTH.items():
            for mon, ctr in sub.items():
                for ltr, wt in ctr.items():
                    fh.write(f'{sid+1},{buck},{sex},{race},{mon},{ltr},{wt}\n')
    
    # Position tables
    with open(f'{out_dir}/pos_overall.csv', 'w') as fh:
        fh.write('bucket,sex,race,overall,position,letter,count\n')
        for (buck, sex, race), posdict in P_OVER.items():
            for pos, ctr in posdict.items():
                for ltr, ct in ctr.items():
                    fh.write(f'{buck},{sex},{race},0,{pos},{ltr},{ct}\n')
    
    with open(f'{out_dir}/pos_year.csv', 'w') as fh:
        fh.write('bucket,sex,race,year,position,letter,count\n')
        for (buck, sex, race), sub in P_YEAR.items():
            for yr, posdict in sub.items():
                for pos, ctr in posdict.items():
                    for ltr, ct in ctr.items():
                        fh.write(f'{buck},{sex},{race},{yr},{pos},{ltr},{ct}\n')
    
    with open(f'{out_dir}/pos_month_idx.csv', 'w') as fh:
        fh.write('bucket,sex,race,month_idx,position,letter,count\n')
        for (buck, sex, race), sub in P_MONTH.items():
            for mon, posdict in sub.items():
                for pos, ctr in posdict.items():
                    for ltr, ct in ctr.items():
                        fh.write(f'{buck},{sex},{race},{mon},{pos},{ltr},{ct}\n')
    
    print(f'[INFO] aggregate CSVs written to ./{out_dir}/')

# ============================================================ #
# 11. CLI                                                      #
# ============================================================ #
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Deaths CSV to process')
    ap.add_argument('--map', required=True, help='ICD->cause mapping CSV')
    ap.add_argument('--lookup', required=True, help='Pre-computed ICD10->DL lookup CSV')
    ap.add_argument('--output', default='converted_with_original.csv')
    ap.add_argument('--use-ent-ucod2', action='store_true',
                    help='Use LAST code on first line as ent_ucod (sensitivity analysis)')
    ap.add_argument('--debug', action='store_true')
    args = ap.parse_args()

    lookup, rule_info = load_lookup(args.lookup)
    exact_rules, range_rules = load_rules(args.map)
    
    process_file(args.input, lookup, rule_info, exact_rules, range_rules, 
                 args.use_ent_ucod2, args.output, args.debug)
    write_aggregates()
