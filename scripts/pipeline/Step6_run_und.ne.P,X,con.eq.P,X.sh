#!/bin/sh

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|LT65' | perl -anlF, -e 'if(substr($F[1],0,1) ne "P"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "P"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > LT65_und.ne.P,con.eq.P_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|LT65' | perl -anlF, -e 'if(substr($F[1],0,1) ne "X"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "X"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > LT65_und.ne.X,con.eq.X_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|GE65' | perl -anlF, -e 'if(substr($F[1],0,1) ne "P"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "P"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > GE65_und.ne.P,con.eq.P_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|GE65' | perl -anlF, -e 'if(substr($F[1],0,1) ne "X"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "X"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > GE65_und.ne.X,con.eq.X_icd10.csv


cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|LT65' | perl -anlF, -e 'if(substr($F[1],0,1) eq "P"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "P"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > LT65_und.eq.P,con.eq.P_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|LT65' | perl -anlF, -e 'if(substr($F[1],0,1) eq "X"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "X"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > LT65_und.eq.X,con.eq.X_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|GE65' | perl -anlF, -e 'if(substr($F[1],0,1) eq "P"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "P"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > GE65_und.eq.P,con.eq.P_icd10.csv

cat I_converted_with_original-f2-3,9,17-18,ag=sort=uniq-c=sort-rn.H.csv | egrep 'Count|GE65' | perl -anlF, -e 'if(substr($F[1],0,1) eq "X"){$la=length($F[1]);for($i=1;$i<$la;$i++){$r=substr($F[1],$i,1);if($r eq "X"){@ra=split(/ /,$F[4]);$ru=$ra[$i-1];print"$_ ra=@ra use:,$F[0],$la,$i,$r=$ru\nuse:,$F[0],$la,$i,$r=$ru"}}}' | grep '^use:' | cut -d'=' -f2 | sort | uniq -c | sort -rn | stc.pl > GE65_und.eq.X,con.eq.X_icd10.csv

exit
