#!/bin/bash

#put this file in /home/lcross/Bundle_Value/code/preprocessing

#subject directory CHANGE FOR EVERY SUBJECT
subj_dir_prefix=/home/lcross/Bundle_Value/rawdata/dicom/114/
day_num=3

#what is the raw scans directory CHANGE FOR EVERY SUBJECT
scans_dir=/home/shared/xnat_data/archive/Bundle/arc001/JOD-Bundle-114-${day_num}/SCANS/

#make the directory 
mkdir ${subj_dir_prefix}
subj_dir=${subj_dir_prefix}day${day_num}
mkdir ${subj_dir}


# # #DAY 1 with anatomicals

# for run in 32 33
# do
# 	cp -r ${scans_dir}${run}/DICOM/*.dcm ${subj_dir}

# done

#DAY 2-3 without anatomicals

for run in 35 36
do
	cp -r ${scans_dir}${run}/DICOM/*.dcm ${subj_dir}

done