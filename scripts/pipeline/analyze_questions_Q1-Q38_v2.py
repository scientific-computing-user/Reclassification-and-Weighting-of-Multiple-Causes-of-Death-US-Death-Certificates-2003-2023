#!/usr/bin/env python3
"""
Analyze Entity Axis vs Record Axis disease letter coding in US Death Certificates.

Input: Gzipped CSV with columns: Count, rUDS, eUDS, rUCOD, eUCOD
  - rUDS: Record axis Underlying Disease String (sequence of disease letters)
  - eUDS: Entity axis Underlying Disease String
  - rUCOD: Record axis Underlying Cause of Death (ICD-10 code)
  - eUCOD: Entity axis Underlying Cause of Death (ICD-10 code)

Definitions:
  - r1: first letter of rUDS
  - e1: first letter of eUDS
  - lr: length of rUDS
  - le: length of eUDS

Disease Letters:
  Natural: B (Circulatory), C (Cancer), N (Other natural), R (Respiratory),
           E (Endocrine), D (Digestive), V (COVID-19)
  External: P (Drug poisoning), T (Transport), S (Suicide), A (Alcohol-related),
            X (Other external), F (Falls), H (Homicide)
  Unknown: U

Output: Tab-separated table suitable for Excel import.
"""

import sys
import csv
import gzip

# Disease letter order
DL = ['B','C','N','R','E','D','V','P','T','S','A','X','F','H','U']

# Natural vs External classification
NATURAL = set(['B','C','N','R','E','D','V'])
EXTERNAL = set(['P','T','S','A','X','F','H'])

def analyze(input_file, output_file):
    """Run all Q1-Q38 analyses."""
    
    # Initialize counters for each question
    q = {i: {L:0 for L in DL} for i in range(1,39)}
    
    # Q36 is special - indexed by e1 not r1
    q36_by_e1 = {L:0 for L in DL}
    
    # For Q20/Q21 averages (weighted by Count)
    q20_sum = {L:0 for L in DL}  # sum of lr * count
    q20_cnt = {L:0 for L in DL}  # sum of count
    q21_sum = {L:0 for L in DL}  # sum of le * count
    q21_cnt = {L:0 for L in DL}  # sum of count
    
    # Open gzipped or regular CSV
    if input_file.endswith('.gz'):
        f = gzip.open(input_file, 'rt')
    else:
        f = open(input_file, 'r')
    
    reader = csv.DictReader(f)
    
    for row in reader:
        rUDS = row['rUDS']
        eUDS = row['eUDS']
        rUCOD = row['rUCOD']
        eUCOD = row['eUCOD']
        lr = len(rUDS)
        le = len(eUDS)
        r1 = rUDS[0]
        e1 = eUDS[0]
        count = int(row['Count'])
        
        # Find position of r1 in eUDS (1-indexed, 0 if absent)
        pos_r1_in_eUDS = 0
        for i, c in enumerate(eUDS):
            if c == r1:
                pos_r1_in_eUDS = i + 1
                break
        
        # Q1-Q5: lr==1 and le==1 (single disease letter on both axes)
        if lr == 1 and le == 1:
            q[1][r1] += count
            if r1 == e1:
                q[2][r1] += count
            else:
                q[3][r1] += count
            if rUCOD == eUCOD:
                q[4][r1] += count
            else:
                q[5][r1] += count
        
        # Q6, Q8: lr==1 and le>1 (Record single, Entity multiple)
        if lr == 1 and le > 1:
            q[6][r1] += count
            if rUCOD == eUCOD:
                q[8][r1] += count
        
        # Q7, Q9: le==1 and lr>1 (Entity single, Record multiple)
        if le == 1 and lr > 1:
            q[7][r1] += count
            if rUCOD == eUCOD:
                q[9][r1] += count
        
        # Q10-Q12: lr>1 and le>1 (both have multiple disease letters)
        if lr > 1 and le > 1:
            q[10][r1] += count
            if r1 == e1:
                q[11][r1] += count
            else:
                q[12][r1] += count
        
        # Q13-Q15: length comparisons
        if lr > le:
            q[13][r1] += count
        if lr < le:
            q[14][r1] += count
        if lr == le:
            q[15][r1] += count
        
        # Q16-Q17: exact string matches
        if rUDS == eUDS:
            q[16][r1] += count
            if rUCOD == eUCOD:
                q[17][r1] += count
        
        # Q18-Q19: lr>1 and le>1 with UCOD match/mismatch
        if lr > 1 and le > 1:
            if rUCOD == eUCOD:
                q[18][r1] += count
            else:
                q[19][r1] += count
        
        # Q20/Q21: accumulate for average length calculation
        q20_sum[r1] += lr * count
        q20_cnt[r1] += count
        q21_sum[e1] += le * count
        q21_cnt[e1] += count
        
        # Q22-Q23: primary category match (any length)
        if r1 == e1:
            q[22][r1] += count
        else:
            q[23][r1] += count
        
        # Q24-Q27: position of r1 in eUDS
        if pos_r1_in_eUDS == 1:
            q[24][r1] += count
        elif pos_r1_in_eUDS == 2:
            q[25][r1] += count
        elif pos_r1_in_eUDS >= 3:
            q[26][r1] += count
        else:  # absent (pos_r1_in_eUDS == 0)
            q[27][r1] += count
        
        # Q28-Q33: specific important flows (when e1 -> r1)
        if e1 == 'B' and r1 == 'N':
            q[28][r1] += count  # Circulatory -> Other natural
        if e1 == 'N' and r1 == 'B':
            q[29][r1] += count  # Other natural -> Circulatory
        if e1 == 'D' and r1 == 'A':
            q[30][r1] += count  # Digestive -> Alcohol-related
        if e1 == 'R' and r1 == 'P':
            q[31][r1] += count  # Respiratory -> Drug poisoning
        if e1 == 'V' and r1 == 'R':
            q[32][r1] += count  # COVID-19 -> Respiratory
        if e1 == 'R' and r1 == 'V':
            q[33][r1] += count  # Respiratory -> COVID-19
        
        # Q34-Q35: Natural/External boundary crossings
        if e1 in NATURAL and r1 in EXTERNAL:
            q[34][r1] += count  # Natural -> External
        if e1 in EXTERNAL and r1 in NATURAL:
            q[35][r1] += count  # External -> Natural
        
        # Q36: count by e1 (denominator for retention rate calculation)
        q36_by_e1[e1] += count
        
        # Q37-Q38: r1==e1 with UCOD match/mismatch
        if r1 == e1:
            if rUCOD == eUCOD:
                q[37][r1] += count
            else:
                q[38][r1] += count
    
    f.close()
    
    # Calculate averages for Q20/Q21
    q20_avg = {L: q20_sum[L]/q20_cnt[L] if q20_cnt[L] > 0 else 0 for L in DL}
    q21_avg = {L: q21_sum[L]/q21_cnt[L] if q21_cnt[L] > 0 else 0 for L in DL}
    total_q20 = sum(q20_sum.values()) / sum(q20_cnt.values()) if sum(q20_cnt.values()) > 0 else 0
    total_q21 = sum(q21_sum.values()) / sum(q21_cnt.values()) if sum(q21_cnt.values()) > 0 else 0
    
    # Question labels
    labels = {
        1: 'Q1 (lr==le==1)',
        2: 'Q2 (lr==le==1, r1==e1)',
        3: 'Q3 (lr==le==1, r1!=e1)',
        4: 'Q4 (lr==le==1, rUCOD==eUCOD)',
        5: 'Q5 (lr==le==1, rUCOD!=eUCOD)',
        6: 'Q6 (lr==1, le>1)',
        7: 'Q7 (le==1, lr>1)',
        8: 'Q8 (lr==1, le>1, rUCOD==eUCOD)',
        9: 'Q9 (le==1, lr>1, rUCOD==eUCOD)',
        10: 'Q10 (lr>1, le>1)',
        11: 'Q11 (lr>1, le>1, r1==e1)',
        12: 'Q12 (lr>1, le>1, r1!=e1)',
        13: 'Q13 (lr>le)',
        14: 'Q14 (lr<le)',
        15: 'Q15 (lr==le)',
        16: 'Q16 (rUDS==eUDS)',
        17: 'Q17 (rUDS==eUDS, rUCOD==eUCOD)',
        18: 'Q18 (lr>1, le>1, rUCOD==eUCOD)',
        19: 'Q19 (lr>1, le>1, rUCOD!=eUCOD)',
        22: 'Q22 (r1==e1)',
        23: 'Q23 (r1!=e1)',
        24: 'Q24 (r1 at position 1 in eUDS)',
        25: 'Q25 (r1 at position 2 in eUDS)',
        26: 'Q26 (r1 at position 3+ in eUDS)',
        27: 'Q27 (r1 absent from eUDS)',
        28: 'Q28 (e1=B, r1=N) Circulatory->Other natural',
        29: 'Q29 (e1=N, r1=B) Other natural->Circulatory',
        30: 'Q30 (e1=D, r1=A) Digestive->Alcohol-related',
        31: 'Q31 (e1=R, r1=P) Respiratory->Drug poisoning',
        32: 'Q32 (e1=V, r1=R) COVID-19->Respiratory',
        33: 'Q33 (e1=R, r1=V) Respiratory->COVID-19',
        34: 'Q34 (e1=Natural, r1=External)',
        35: 'Q35 (e1=External, r1=Natural)',
        37: 'Q37 (r1==e1, rUCOD==eUCOD)',
        38: 'Q38 (r1==e1, rUCOD!=eUCOD)'
    }
    
    # Write output
    with open(output_file, 'w') as out:
        # Header row
        out.write('DL\t' + '\t'.join(DL) + '\tAny\n')
        
        # Q1-Q19
        for i in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]:
            vals = [str(q[i].get(L,0)) for L in DL]
            out.write(labels[i] + '\t' + '\t'.join(vals) + '\t' + str(sum(q[i].values())) + '\n')
        
        # Q20 (average lr by r1)
        vals = ['%.3f' % q20_avg[L] for L in DL]
        out.write('Q20 (avg lr by r1)\t' + '\t'.join(vals) + '\t%.3f\n' % total_q20)
        
        # Q21 (average le by e1)
        vals = ['%.3f' % q21_avg[L] for L in DL]
        out.write('Q21 (avg le by e1)\t' + '\t'.join(vals) + '\t%.3f\n' % total_q21)
        
        # Q22-Q35
        for i in [22,23,24,25,26,27,28,29,30,31,32,33,34,35]:
            vals = [str(q[i].get(L,0)) for L in DL]
            out.write(labels[i] + '\t' + '\t'.join(vals) + '\t' + str(sum(q[i].values())) + '\n')
        
        # Q36 (count by e1 - note: indexed by e1, not r1)
        vals = [str(q36_by_e1.get(L,0)) for L in DL]
        out.write('Q36 (count by e1, not r1)\t' + '\t'.join(vals) + '\t' + str(sum(q36_by_e1.values())) + '\n')
        
        # Q37-Q38
        for i in [37,38]:
            vals = [str(q[i].get(L,0)) for L in DL]
            out.write(labels[i] + '\t' + '\t'.join(vals) + '\t' + str(sum(q[i].values())) + '\n')
    
    print(f"Results written to {output_file}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python analyze_questions_Q1-Q38.py <input.csv[.gz]> [output.tsv]")
        print("  Input: CSV with columns Count,rUDS,eUDS,rUCOD,eUCOD")
        print("  Output: Tab-separated table (default: questions_Q1-Q38.tsv)")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else 'questions_Q1-Q38.tsv'
    
    analyze(input_file, output_file)
