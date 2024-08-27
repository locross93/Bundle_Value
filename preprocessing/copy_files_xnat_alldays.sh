#!/bin/bash

#put this file in /home/lcross/Bundle_Value/code/preprocessing

#subject directory CHANGE FOR EVERY SUBJECT
subj=114
subj_dir_prefix=/home/lcross/Bundle_Value/rawdata/dicom/${subj}/

# #ALL DAYS
for day_num in 1 2 3
	do

	#what is the raw scans directory CHANGE FOR EVERY SUBJECT
	scans_dir=/home/shared/xnat_data/archive/Bundle/arc001/JOD-Bundle-${subj}-${day_num}/SCANS/

	#make the directory 
	mkdir ${subj_dir_prefix}
	subj_dir=${subj_dir_prefix}day${day_num}
	mkdir ${subj_dir}

	for run in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35
	do
		cp -r ${scans_dir}${run}/DICOM/*.dcm ${subj_dir}

	done
done

