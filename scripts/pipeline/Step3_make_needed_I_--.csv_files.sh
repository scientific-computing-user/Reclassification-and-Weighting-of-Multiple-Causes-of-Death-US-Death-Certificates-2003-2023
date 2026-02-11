#!/bin/sh

ln -sf converted_with_original_use-last_eUCOD.csv converted_with_original.csv

echo "grep '^I' converted_with_original.csv > all_I=converted_with_original.csv"
grep '^I' converted_with_original.csv > all_I=converted_with_original.csv


echo "( echo \"Count,rUDS,eUDS,year,age_group\" ;           grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,6,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$F[2],$m\"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,6,9,ag=sort=uniq-c=sort-rn.H.csv"

( echo "Count,rUDS,rADS,year,age_group" ;           grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2,3,6,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[2],$m"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv


echo "( echo \"Count,rADS,rUDS,rUCOD,rec_axis,age-group\" ;                grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-3,9,17-18 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$F[3],$F[4],$m\"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-3,11,19-20,ag=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rADS,rUDS,rUCOD,rec_axis,age-group" ;                grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-3,9,17-18 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[3],$F[4],$m"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-3,11,19-20,ag=sort=uniq-c=sort-rn.H.csv &


echo "( echo \"Count,rUDS,eUDS,rUCOD,eUCOD\" ;                             grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,17,22         | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rUDS,eUDS,rUCOD,eUCOD" ;                             grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,17,22         | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv &


echo "( echo \"Count,rUDS,rADS,eUDS,rUDS,year,month,age_group,sex,race\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-7,9,10,11  | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-7,9,10,11=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rUDS,rADS,eUDS,rUDS,year,month,age_group,sex,race" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-7,9,10,11  | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-7,9,10,11=sort=uniq-c=sort-rn.H.csv &


echo "( echo \"Count,rUDS,rADS,year,age_group\" ;           grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2,3,6,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$F[2],$m\"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rUDS,rADS,year,age_group" ;           grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2,3,6,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[2],$m"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2,3,6,9,ag=sort=uniq-c=sort-rn.H.csv &


# Need for icd10 codes
echo "( echo \"Count,rADS,rUDS,rUCOD,rec_axis,age-group\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-3,9,17-18 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$F[3],$F[4],$m\"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rADS,rUDS,rUCOD,rec_axis,age-group" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-3,9,17-18 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[3],$F[4],$m"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv &


echo "( echo \"Count,rUDS,eUDS,rUCOD,eUCOD,year,age_group\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,6,9,17,22 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$F[4],$F[5],$F[2],$m\"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,6,9,17,22,ag=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rUDS,eUDS,rUCOD,eUCOD,year,age_group" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,6,9,17,22 | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[3]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$F[4],$F[5],$F[2],$m"' | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,6,9,17,22,ag=sort=uniq-c=sort-rn.H.csv &


# Used in 'cat grep=U071=I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv | python /Users/levitt/levitt/NewProjects25/USDeathCertificates/code/analyze_U071_ucod_v4.py 100 -c > analyze_U071_ucod_v3.py=100-c.csv'
echo "( echo \"Count,rADS,rUDS,eADS,eUDS,rUCOD,rec_axis,ent_axis,eUCOD\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-5,17-19,22 | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv"

( echo "Count,rADS,rUDS,eADS,eUDS,rUCOD,rec_axis,ent_axis,eUCOD" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f2-5,17-19,22 | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f2-5,17-19,22=sort=uniq-c=sort-rn.H.csv


# Select eCOD2
echo "( echo \"Count,rUDS,eUDS,rUCOD,eUCOD2\" ;                            grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,17,22,23 | perl -anlF, -e 'if(defined($F[4])){$F[3]=$F[4]};print\"$F[0],$F[1],$F[2],$F[3]\"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,17,22,23,eUCOD2=sort=uniq-c=sort-rn.H.csv &"

( echo "Count,rUDS,eUDS,rUCOD,eUCOD2" ;                            grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,17,22,23 | perl -anlF, -e 'if(defined($F[4])){$F[3]=$F[4]};print"$F[0],$F[1],$F[2],$F[3]"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,17,22,23,eUCOD2=sort=uniq-c=sort-rn.H.csv &


# New
echo "( echo \"Count,r1,rUCOD\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,17 | perl -anlF, -e '$r1=substr($F[0],0,1);print\"$r1,$F[1]\"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,17,r1=sort=uniq-c=sort-rn.H.csv"

( echo "Count,r1,rUCOD" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,17 | perl -anlF, -e '$r1=substr($F[0],0,1);print"$r1,$F[1]"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,17,r1=sort=uniq-c=sort-rn.H.csv


echo "( echo \"Count,rUDS,eUDS,age_group\" ;  grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m=\"GE65\"}else{$m=\"LT65\"};print\"$F[0],$F[1],$m\"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv"

( echo "Count,rUDS,eUDS,age_group" ;  grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f3,5,9     | perl -anlF, -e '($a,$b)=split(/[-+]/,$F[2]);if($a>=65){$m="GE65"}else{$m="LT65"};print"$F[0],$F[1],$m"' | sort | uniq -c | sort -rn | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f3,5,9,ag=sort=uniq-c=sort-rn.H.csv


echo "( echo \"Count,rec_axis\" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f18 | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv"
( echo "Count,rec_axis" ; grep -v rUDS all_I=converted_with_original.csv | cut -d',' -f18 | sort | uniq -c | sort -rn | grep -v rUDS | sed 's/^ *//' | sed 's/ /,/' ) > I_converted_with_original-f18=sort=uniq-c=sort-rn.H.csv


echo "rm I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz; gzip -k I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv"
rm I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv.gz; gzip -k I_converted_with_original-f3,5,17,22=sort=uniq-c=sort-rn.H.csv

wait

echo "ls -al I_*sort*csv"
ls -al I_*sort*csv


echo "ls -1 I_*sort*.csv | do_all 'head -3 %% | sed \"s/$/ %%/\"' | ttL.pl | A+add_empty_line_after_diff-c.pl -c1 > I_--sort--.csv.headers"
ls -1 I_*sort*.csv | do_all 'head -3 %% | sed "s/$/ %%/"' | ttL.pl | A+add_empty_line_after_diff-c.pl -c1 > I_--sort--.csv.headers


echo "ls -1 I_*.H.csv | do_all \"wc -l %%; head -2 %%\" | join_nth.pl -n 3 | ttL.pl > I_--.csv_file.wc.2.headers"
ls -1 I_*.H.csv | do_all "wc -l %%; head -2 %%" | join_nth.pl -n 3 | ttL.pl > I_--.csv_file.wc.2.headers

exit
