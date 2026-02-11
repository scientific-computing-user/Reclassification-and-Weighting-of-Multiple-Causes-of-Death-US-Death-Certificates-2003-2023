#!/bin/sh  

../code/Step1_Convert_mcd_usa-deaths-masr-icd10_to_big_death_input_v3.sh > Step1_Convert_mcd_usa-deaths-masr-icd10_to_big_death_input_v3.sh=.log

../code/Step2_Convert_icd10_to_ADS_use-last_eUCOD.sh > Step2_Convert_icd10_to_ADS_use-last_eUCOD.sh=log

../code/Step3_make_needed_I_--.csv_files.sh > Step3_make_needed_I_--.csv_files.sh=log

../code/Step4-8_run_python_jobs.sh > Step4-8_run_python_jobs.sh=log

../code/Step9_run_Tables.sh > Step9_run_Tables.sh.log

exit
