#!/usr/bin/env python3
"""
Analyze COVID (U071) UCOD promotions and demotions between record axis and entity axis.

Usage: cat file.csv | python3 analyze_U071_ucod_v3.py [N] [-c]
       python3 analyze_U071_ucod_v3.py [N] [-c] < file.csv

  N  = number of top ICD-10 codes to display (default: 50)
  -c = output as CSV format (default: formatted text)

Reads CSV from stdin with columns:
  Count,rADS,rUDS,eADS,eUDS,rUCOD,rec_axis,ent_axis,eUCOD

Outputs:
  1. Total promotions (rUCOD=U071, eUCOD!=U071)
  2. Total demotions (eUCOD=U071, rUCOD!=U071)
  3. Position of V in eUDS for promotions
  4. Disease letters replaced when U071 promoted (eADS[0])
  5. Disease letters that replaced U071 when demoted (rADS[0])
  6. Top N ICD-10 codes replaced when U071 promoted
  7. Top N ICD-10 codes that replaced U071 when demoted
"""

import sys
import csv
from collections import defaultdict

# ICD-10 code descriptions
ICD10_DESC = {
    'A047': 'Enterocolitis due to C. difficile',
    'A410': 'Sepsis due to Staph. aureus',
    'A415': 'Sepsis due to other Gram-negative',
    'A419': 'Sepsis, unspecified organism',
    'A490': 'Staphylococcal infection, unspec',
    'A498': 'Other bacterial infections',
    'A499': 'Bacterial infection, unspecified',
    'B24': 'HIV disease',
    'B342': 'Coronavirus infection, unspec',
    'B348': 'Other viral infections',
    'B349': 'Viral infection, unspecified',
    'B49': 'Unspecified mycosis',
    'B948': 'Sequelae of other infectious diseases',
    'B99': 'Other infectious diseases',
    'C159': 'Esophageal cancer',
    'C169': 'Stomach cancer',
    'C189': 'Colon cancer',
    'C20': 'Rectal cancer',
    'C220': 'Liver cell carcinoma',
    'C221': 'Intrahepatic bile duct carcinoma',
    'C229': 'Liver cancer, unspecified',
    'C259': 'Pancreatic cancer',
    'C349': 'Lung cancer',
    'C439': 'Melanoma',
    'C509': 'Breast cancer',
    'C541': 'Endometrial cancer',
    'C55': 'Uterine cancer, unspecified',
    'C56': 'Ovarian cancer',
    'C61': 'Prostate cancer',
    'C64': 'Kidney cancer',
    'C679': 'Bladder cancer',
    'C719': 'Brain cancer',
    'C80': 'Cancer, unknown primary',
    'C831': 'Mantle cell lymphoma',
    'C833': 'Diffuse large B-cell lymphoma',
    'C851': 'B-cell lymphoma, unspecified',
    'C859': 'Non-Hodgkin lymphoma',
    'C900': 'Multiple myeloma',
    'C911': 'Chronic lymphocytic leukemia',
    'C920': 'Acute myeloid leukemia',
    'C959': 'Leukemia, unspecified',
    'C97': 'Malignant neoplasms, multiple sites',
    'D469': 'Myelodysplastic syndrome',
    'D649': 'Anemia, unspecified',
    'D65': 'DIC',
    'D689': 'Coagulation defect',
    'D869': 'Sarcoidosis',
    'D899': 'Immune disorder',
    'E43': 'Severe malnutrition',
    'E46': 'Protein-calorie malnutrition',
    'E86': 'Volume depletion',
    'E109': 'Type 1 diabetes',
    'E112': 'Type 2 diabetes with kidney complication',
    'E119': 'Type 2 diabetes',
    'E141': 'Diabetes with ketoacidosis',
    'E142': 'Diabetes with kidney complication',
    'E147': 'Diabetes with multiple complications',
    'E149': 'Diabetes, unspecified',
    'E668': 'Other obesity',
    'E669': 'Obesity, unspecified',
    'E785': 'Hyperlipidemia',
    'E870': 'Hyperosmolality',
    'E872': 'Acidosis',
    'E875': 'Hyperkalemia',
    'E889': 'Metabolic disorder',
    'F019': 'Vascular dementia',
    'F03': 'Dementia, unspecified',
    'F059': 'Delirium',
    'F069': 'Mental disorder due to brain damage',
    'F179': 'Tobacco use disorder',
    'F919': 'Conduct disorder',
    'G10': 'Huntington disease',
    'G20': 'Parkinson disease',
    'G049': 'Encephalitis',
    'G122': 'Motor neuron disease',
    'G231': 'Progressive supranuclear palsy',
    'G300': 'Alzheimer early onset',
    'G301': 'Alzheimer late onset',
    'G309': 'Alzheimer disease',
    'G310': 'Frontotemporal dementia',
    'G311': 'Senile degeneration of brain',
    'G318': 'Other degenerative diseases',
    'G319': 'Degenerative disease, unspec',
    'G35': 'Multiple sclerosis',
    'G473': 'Sleep apnea',
    'G700': 'Myasthenia gravis',
    'G710': 'Muscular dystrophy',
    'G809': 'Cerebral palsy',
    'G931': 'Anoxic brain damage',
    'G934': 'Encephalopathy, unspecified',
    'G939': 'Brain disorder',
    'H919': 'Hearing loss',
    'I10': 'Essential hypertension',
    'I110': 'Hypertensive heart disease with HF',
    'I119': 'Hypertensive heart disease',
    'I120': 'Hypertensive CKD',
    'I131': 'Hypertensive heart and CKD with HF',
    'I132': 'Hypertensive heart and CKD with HF and CKD',
    'I214': 'NSTEMI',
    'I219': 'Acute MI, unspecified',
    'I249': 'Acute ischemic heart disease',
    'I250': 'Atherosclerotic CVD',
    'I251': 'Atherosclerotic heart disease',
    'I255': 'Ischemic cardiomyopathy',
    'I259': 'Chronic ischemic heart disease',
    'I269': 'Pulmonary embolism',
    'I272': 'Other pulmonary heart disease',
    'I279': 'Pulmonary heart disease',
    'I350': 'Aortic stenosis',
    'I429': 'Cardiomyopathy, unspecified',
    'I461': 'Sudden cardiac death',
    'I469': 'Cardiac arrest',
    'I48': 'Atrial fibrillation',
    'I499': 'Cardiac arrhythmia',
    'I500': 'Congestive heart failure',
    'I509': 'Heart failure, unspecified',
    'I514': 'Myocarditis, unspecified',
    'I516': 'Cardiovascular disease',
    'I519': 'Heart disease, unspecified',
    'I619': 'Intracerebral hemorrhage',
    'I629': 'Intracranial hemorrhage',
    'I639': 'Cerebral infarction',
    'I64': 'Stroke, unspecified',
    'I672': 'Cerebral atherosclerosis',
    'I679': 'Cerebrovascular disease',
    'I693': 'Sequelae of cerebral infarction',
    'I694': 'Sequelae of stroke',
    'I698': 'Sequelae of cerebrovascular disease',
    'I709': 'Atherosclerosis',
    'I739': 'Peripheral vascular disease',
    'I802': 'Venous thrombosis',
    'I959': 'Hypotension',
    'I99': 'Circulatory disorder',
    'J13': 'Pneumonia due to S. pneumoniae',
    'J22': 'Acute lower respiratory infection',
    'J40': 'Bronchitis',
    'J100': 'Influenza with pneumonia, identified',
    'J101': 'Influenza with pneumonia',
    'J111': 'Influenza with pneumonia',
    'J128': 'Other viral pneumonia',
    'J129': 'Viral pneumonia, unspecified',
    'J150': 'Pneumonia due to K. pneumoniae',
    'J151': 'Pneumonia due to Pseudomonas',
    'J152': 'Pneumonia due to Staphylococcus',
    'J154': 'Pneumonia due to other strep',
    'J159': 'Bacterial pneumonia, unspecified',
    'J180': 'Bronchopneumonia',
    'J181': 'Lobar pneumonia',
    'J189': 'Pneumonia, unspecified',
    'J439': 'Emphysema',
    'J440': 'COPD with acute exacerbation',
    'J441': 'COPD with acute exacerbation',
    'J449': 'COPD, unspecified',
    'J459': 'Asthma',
    'J690': 'Aspiration pneumonia',
    'J80': 'ARDS',
    'J81': 'Pulmonary edema',
    'J90': 'Pleural effusion',
    'J840': 'Alveolar and parietoalveolar conditions',
    'J841': 'Pulmonary fibrosis',
    'J849': 'Interstitial lung disease',
    'J939': 'Pneumothorax',
    'J958': 'Postprocedural respiratory disorder',
    'J960': 'Acute respiratory failure',
    'J961': 'Chronic respiratory failure',
    'J969': 'Respiratory failure, unspecified',
    'J982': 'Interstitial emphysema',
    'J984': 'Other disorders of lung',
    'J988': 'Other respiratory disorders',
    'J989': 'Respiratory disorder, unspecified',
    'K559': 'Vascular disorder of intestine',
    'K566': 'Intestinal obstruction',
    'K703': 'Alcoholic cirrhosis',
    'K729': 'Hepatic failure',
    'K746': 'Other cirrhosis of liver',
    'K922': 'GI hemorrhage',
    'L899': 'Pressure ulcer',
    'M069': 'Rheumatoid arthritis',
    'M869': 'Osteomyelitis',
    'N170': 'Acute kidney failure with necrosis',
    'N179': 'Acute kidney failure',
    'N183': 'CKD stage 3',
    'N184': 'CKD stage 4',
    'N185': 'CKD stage 5',
    'N19': 'Renal failure, unspecified',
    'N189': 'CKD, unspecified',
    'N288': 'Other kidney disorders',
    'N289': 'Kidney disorder, unspecified',
    'N390': 'UTI',
    'O268': 'Other pregnancy-related conditions',
    'O961': 'Death from direct obstetric cause',
    'O985': 'Other viral diseases in pregnancy',
    'Q909': 'Down syndrome',
    'R000': 'Tachycardia',
    'R060': 'Dyspnea',
    'R068': 'Other respiratory abnormalities',
    'R13': 'Dysphagia',
    'R090': 'Asphyxia',
    'R092': 'Respiratory arrest',
    'R418': 'Altered mental status',
    'R53': 'Malaise and fatigue',
    'R54': 'Age-related debility',
    'R568': 'Other convulsions',
    'R570': 'Cardiogenic shock',
    'R578': 'Other shock',
    'R579': 'Shock, unspecified',
    'R58': 'Hemorrhage NEC',
    'R628': 'Lack of expected development',
    'R64': 'Cachexia',
    'R688': 'Other specified symptoms',
    'R91': 'Abnormal findings lung imaging',
    'R95': 'SIDS',
    'R99': 'Ill-defined cause of mortality',
    'U071': 'COVID-19',
    'U099': 'Post COVID-19 condition',
    'W18': 'Other fall on same level',
    'W19': 'Unspecified fall',
    'W80': 'Inhalation of gastric contents',
    'X42': 'Accidental poisoning by narcotics',
    'X44': 'Accidental poisoning by drugs',
    'Y841': 'Surgical procedure complication',
}

def get_desc(code):
    """Return description for ICD-10 code, or empty string if not found."""
    return ICD10_DESC.get(code, '')

def main():
    # Parse command line arguments
    top_n = 50  # default
    csv_output = False
    
    for arg in sys.argv[1:]:
        if arg == '-c' or arg == '--csv':
            csv_output = True
        else:
            try:
                top_n = int(arg)
            except ValueError:
                sys.stderr.write("Usage: %s [N] [-c]\n" % sys.argv[0])
                sys.stderr.write("  N  = number of top ICD-10 codes to display (default: 50)\n")
                sys.stderr.write("  -c = output as CSV format\n")
                sys.exit(1)

    # Columns: Count,rADS,rUDS,eADS,eUDS,rUCOD,rec_axis,ent_axis,eUCOD
    # Indices:   0     1    2    3    4     5       6        7       8

    # Accumulators
    promotion_total = 0
    demotion_total = 0
    v_position_counts = defaultdict(int)
    no_v_count = 0
    promoted_letters = defaultdict(int)
    demoted_letters = defaultdict(int)
    promoted_codes = defaultdict(int)   # eUCOD codes replaced by U071
    demoted_codes = defaultdict(int)    # rUCOD codes that replaced U071
    promoted_code_letters = {}          # eUCOD -> disease letter mapping
    demoted_code_letters = {}           # rUCOD -> disease letter mapping

    reader = csv.reader(sys.stdin)
    header = next(reader, None)  # skip header

    # Data validation counters
    empty_eUCOD_fixed = 0
    empty_rUCOD_fixed = 0
    invalid_rows = 0

    for row in reader:
        if len(row) < 9:
            invalid_rows += 1
            continue
        try:
            count = int(row[0])
        except ValueError:
            invalid_rows += 1
            continue

        rADS = row[1]
        rUDS = row[2]
        eADS = row[3]
        eUDS = row[4]
        rUCOD = row[5]
        rec_axis = row[6]
        ent_axis = row[7]
        eUCOD = row[8]

        # Data validation: fix empty eUCOD by using first code from ent_axis
        if eUCOD == '' or eUCOD.isspace():
            codes = ent_axis.split()
            if codes:
                eUCOD = codes[0]
                empty_eUCOD_fixed += count

        # Data validation: fix empty rUCOD by using first code from rec_axis
        if rUCOD == '' or rUCOD.isspace():
            codes = rec_axis.split()
            if codes:
                rUCOD = codes[0]
                empty_rUCOD_fixed += count

        # Promotions: rUCOD=U071 and eUCOD!=U071
        if rUCOD == 'U071' and eUCOD != 'U071':
            promotion_total += count

            # Position of V in eUDS
            pos = eUDS.find('V')
            if pos >= 0:
                v_position_counts[pos + 1] += count  # 1-based
            else:
                no_v_count += count

            # Disease letter replaced (first char of eADS)
            if len(eADS) > 0:
                promoted_letters[eADS[0]] += count
                # Store disease letter for this code
                if eUCOD not in promoted_code_letters:
                    promoted_code_letters[eUCOD] = eADS[0]

            # ICD-10 code replaced
            promoted_codes[eUCOD] += count

        # Demotions: eUCOD=U071 and rUCOD!=U071
        if eUCOD == 'U071' and rUCOD != 'U071':
            demotion_total += count

            # Disease letter that replaced U071 (first char of rADS)
            if len(rADS) > 0:
                demoted_letters[rADS[0]] += count
                # Store disease letter for this code
                if rUCOD not in demoted_code_letters:
                    demoted_code_letters[rUCOD] = rADS[0]

            # ICD-10 code that replaced U071
            demoted_codes[rUCOD] += count

    # Output results
    if csv_output:
        # CSV output mode
        writer = csv.writer(sys.stdout)
        
        # SUMMARY table
        writer.writerow(['Table', 'SUMMARY'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Promotions (rUCOD=U071 eUCOD!=U071)', promotion_total])
        writer.writerow(['Demotions (eUCOD=U071 rUCOD!=U071)', demotion_total])
        if empty_eUCOD_fixed > 0:
            writer.writerow(['Empty eUCOD fixed from ent_axis', empty_eUCOD_fixed])
        if empty_rUCOD_fixed > 0:
            writer.writerow(['Empty rUCOD fixed from rec_axis', empty_rUCOD_fixed])
        if invalid_rows > 0:
            writer.writerow(['Invalid rows skipped', invalid_rows])
        writer.writerow([])
        
        # POSITION OF V table
        writer.writerow(['Table', 'POSITION_OF_V_IN_eUDS_FOR_PROMOTIONS'])
        writer.writerow(['Position', 'Count', 'CumPct'])
        total = sum(v_position_counts.values()) + no_v_count
        running = 0
        for pos in sorted(v_position_counts.keys()):
            cnt = v_position_counts[pos]
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            writer.writerow([pos, cnt, '%.1f' % pct])
        if no_v_count > 0:
            running += no_v_count
            pct = 100.0 * running / total if total > 0 else 0
            writer.writerow(['no_V', no_v_count, '%.1f' % pct])
        writer.writerow(['Total', total, ''])
        writer.writerow([])
        
        # PROMOTED LETTERS table
        writer.writerow(['Table', 'DISEASE_LETTERS_REPLACED_WHEN_U071_PROMOTED'])
        writer.writerow(['Letter', 'Count', 'CumPct'])
        total = sum(promoted_letters.values())
        running = 0
        for letter, cnt in sorted(promoted_letters.items(), key=lambda x: -x[1]):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            writer.writerow([letter, cnt, '%.1f' % pct])
        writer.writerow(['Total', total, ''])
        writer.writerow([])
        
        # DEMOTED LETTERS table
        writer.writerow(['Table', 'DISEASE_LETTERS_THAT_REPLACED_U071_WHEN_DEMOTED'])
        writer.writerow(['Letter', 'Count', 'CumPct'])
        total = sum(demoted_letters.values())
        running = 0
        for letter, cnt in sorted(demoted_letters.items(), key=lambda x: -x[1]):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            writer.writerow([letter, cnt, '%.1f' % pct])
        writer.writerow(['Total', total, ''])
        writer.writerow([])
        
        # PROMOTED CODES table
        writer.writerow(['Table', 'ICD10_CODES_REPLACED_WHEN_U071_PROMOTED_TOP_%d' % top_n])
        writer.writerow(['eUCOD', 'DL', 'Count', 'CumPct', 'Description'])
        total = sum(promoted_codes.values())
        running = 0
        for i, (code, cnt) in enumerate(sorted(promoted_codes.items(), key=lambda x: -x[1])):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            desc = get_desc(code)
            dl = promoted_code_letters.get(code, '')
            writer.writerow([code, dl, cnt, '%.1f' % pct, desc])
            if i >= top_n - 1:
                break
        writer.writerow(['Total', '', total, '', ''])
        writer.writerow([])
        
        # DEMOTED CODES table
        writer.writerow(['Table', 'ICD10_CODES_THAT_REPLACED_U071_WHEN_DEMOTED_TOP_%d' % top_n])
        writer.writerow(['rUCOD', 'DL', 'Count', 'CumPct', 'Description'])
        total = sum(demoted_codes.values())
        running = 0
        for i, (code, cnt) in enumerate(sorted(demoted_codes.items(), key=lambda x: -x[1])):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            desc = get_desc(code)
            dl = demoted_code_letters.get(code, '')
            writer.writerow([code, dl, cnt, '%.1f' % pct, desc])
            if i >= top_n - 1:
                break
        writer.writerow(['Total', '', total, '', ''])
    
    else:
        # Text output mode (original format)
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print("Promotions (rUCOD=U071, eUCOD!=U071): %d" % promotion_total)
        print("Demotions  (eUCOD=U071, rUCOD!=U071): %d" % demotion_total)

        # Data validation report
        if empty_eUCOD_fixed > 0 or empty_rUCOD_fixed > 0 or invalid_rows > 0:
            print("\n" + "=" * 60)
            print("DATA VALIDATION")
            print("=" * 60)
            if empty_eUCOD_fixed > 0:
                print("Empty eUCOD fixed from ent_axis: %d records" % empty_eUCOD_fixed)
            if empty_rUCOD_fixed > 0:
                print("Empty rUCOD fixed from rec_axis: %d records" % empty_rUCOD_fixed)
            if invalid_rows > 0:
                print("Invalid rows skipped: %d" % invalid_rows)

        print("\n" + "=" * 60)
        print("POSITION OF V IN eUDS FOR PROMOTIONS")
        print("=" * 60)
        print("Position   Count  Cum%")
        total = sum(v_position_counts.values()) + no_v_count
        running = 0
        for pos in sorted(v_position_counts.keys()):
            cnt = v_position_counts[pos]
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            print("%8d  %6d  %5.1f" % (pos, cnt, pct))
        if no_v_count > 0:
            running += no_v_count
            pct = 100.0 * running / total if total > 0 else 0
            print("   no V   %6d  %5.1f" % (no_v_count, pct))
        print("Total: %d" % total)

        print("\n" + "=" * 60)
        print("DISEASE LETTERS REPLACED WHEN U071 PROMOTED (eADS[0])")
        print("=" * 60)
        print("Letter   Count  Cum%")
        total = sum(promoted_letters.values())
        running = 0
        for letter, cnt in sorted(promoted_letters.items(), key=lambda x: -x[1]):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            print("%-6s  %6d  %5.1f" % (letter, cnt, pct))
        print("Total: %d" % total)

        print("\n" + "=" * 60)
        print("DISEASE LETTERS THAT REPLACED U071 WHEN DEMOTED (rADS[0])")
        print("=" * 60)
        print("Letter   Count  Cum%")
        total = sum(demoted_letters.values())
        running = 0
        for letter, cnt in sorted(demoted_letters.items(), key=lambda x: -x[1]):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            print("%-6s  %6d  %5.1f" % (letter, cnt, pct))
        print("Total: %d" % total)

        print("\n" + "=" * 60)
        print("ICD-10 CODES REPLACED WHEN U071 PROMOTED (eUCOD) - TOP %d" % top_n)
        print("=" * 60)
        print("eUCOD     DL  Count  Cum%  Description")
        total = sum(promoted_codes.values())
        running = 0
        for i, (code, cnt) in enumerate(sorted(promoted_codes.items(), key=lambda x: -x[1])):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            desc = get_desc(code)
            dl = promoted_code_letters.get(code, '')
            print("%-8s  %s  %6d  %5.1f  %s" % (code, dl, cnt, pct, desc))
            if i >= top_n - 1:
                break
        print("Total: %d" % total)

        print("\n" + "=" * 60)
        print("ICD-10 CODES THAT REPLACED U071 WHEN DEMOTED (rUCOD) - TOP %d" % top_n)
        print("=" * 60)
        print("rUCOD     DL  Count  Cum%  Description")
        total = sum(demoted_codes.values())
        running = 0
        for i, (code, cnt) in enumerate(sorted(demoted_codes.items(), key=lambda x: -x[1])):
            running += cnt
            pct = 100.0 * running / total if total > 0 else 0
            desc = get_desc(code)
            dl = demoted_code_letters.get(code, '')
            print("%-8s  %s  %6d  %5.1f  %s" % (code, dl, cnt, pct, desc))
            if i >= top_n - 1:
                break
        print("Total: %d" % total)

if __name__ == '__main__':
    main()
