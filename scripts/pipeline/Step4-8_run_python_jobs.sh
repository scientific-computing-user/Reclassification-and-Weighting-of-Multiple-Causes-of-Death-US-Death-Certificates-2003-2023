#!/bin/sh


../code/Step4_run_python_jobs_Part1.sh

../code/Step5_run_python_jobs_Part2.sh  &

../code/Step6_run_und.ne.P,X,con.eq.P,X.sh  &

../code/Step7_run_ucod,r1_concordance_GM.sh  &

../code/Step8_Analyize_U071.sh  &


exit
