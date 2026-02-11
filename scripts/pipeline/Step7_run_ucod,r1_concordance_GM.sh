#!/bin/sh

# Input filename
INPUT_CSV="all_I=converted_with_original.csv"
# Intermediate filename
INTER_CSV="I_converted_with_original-f3,5,6,9,17,22,ag=sort=uniq-c=sort-rn.H.csv"

# 1. Extract, Calculate Age Group, Count Frequencies

if true; then
( echo "Count,rUDS,eUDS,rUCOD,eUCOD,year,age_group" ; \
  grep -v rUDS "$INPUT_CSV" | \
  cut -d',' -f3,5,6,9,17,22 | \
  perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[4],$F[5],$F[2],$m"' | \
  sort | uniq -c | sort -rn | \
  grep -v rUDS | sed 's/^ *//' | \
  sed 's/ /,/' ) > "$INTER_CSV"
fi

# 2. Concordance: r1e1 (First char of UDS), Excluding Covid (U071)
egrep -v 'Count|U071' "$INTER_CSV" | \
perl -anlF, -e '$ca{$F[5]}+=$F[0]; if(substr($F[1],0,1) eq substr($F[2],0,1)){$ce{$F[5]}+=$F[0]} END{{foreach $x (keys(%ca)){printf("%s %d %d %.1f\n",$x,$ca{$x},$ce{$x},100*$ce{$x}/$ca{$x})}}}' | \
sort -n | stc.pl > "r1e1_no_U071_concordance_yearly.csv"

# 3. Concordance: ICD10 (Full Code), Excluding Covid (U071)
egrep -v 'Count|U071' "$INTER_CSV" | \
perl -anlF, -e '$ca{$F[5]}+=$F[0]; if($F[3] eq $F[4]){$ce{$F[5]}+=$F[0]} END{{foreach $x (keys(%ca)){printf("%s %d %d %.1f\n",$x,$ca{$x},$ce{$x},100*$ce{$x}/$ca{$x})}}}' | \
sort -n | stc.pl > "icd10_no_U071_concordance_yearly.csv"

# 4. Concordance: r1e1, Including All
grep -v Count "$INTER_CSV" | \
perl -anlF, -e '$ca{$F[5]}+=$F[0]; if(substr($F[1],0,1) eq substr($F[2],0,1)){$ce{$F[5]}+=$F[0]} END{{foreach $x (keys(%ca)){printf("%s %d %d %.1f\n",$x,$ca{$x},$ce{$x},100*$ce{$x}/$ca{$x})}}}' | \
sort -n | stc.pl > "r1e1_concordance_yearly.csv"

# 5. Concordance: ICD10, Including All
grep -v Count "$INTER_CSV" | \
perl -anlF, -e '$ca{$F[5]}+=$F[0]; if($F[3] eq $F[4]){$ce{$F[5]}+=$F[0]} END{{foreach $x (keys(%ca)){printf("%s %d %d %.1f\n",$x,$ca{$x},$ce{$x},100*$ce{$x}/$ca{$x})}}}' | \
sort -n | stc.pl > "icd10_concordance_yearly.csv"

exit
