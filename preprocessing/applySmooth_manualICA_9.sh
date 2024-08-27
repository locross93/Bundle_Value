#!/bin/bash

# pull in the subject we should be working on
subjectID=$1
sessionID=$2

echo "Subject ${subjectID} Session ${sessionID}"

# Set directory containing fsl scripts and templates
# directory with input files
input_dir=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${subjectID}/Session${sessionID}.ica/
# Set output directory for preprocessed files
output_dir=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${subjectID}/Session${sessionID}.ica/
analysis_dir1=/home/lcross/Bundle_Value/analysis/sub${subjectID:0:3}/
analysis_dir2=${analysis_dir1}day${subjectID:4:5}/manualICA/
sample_dir=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/samples/sub${subjectID}/

# Smooth the data at 5mm (2.12 half gaus) and limit data within the mask
echo "Smoothing Subject ${subjectID} Session ${sessionID} at $(date +"%T")"
fslmaths ${input_dir}filtered_func_data_clean_unwarped_new_ANTsReg -thr 1000 -bin ${input_dir}filtered_func_data_clean_unwarped_new_ANTsReg_masked
fslmaths ${input_dir}filtered_func_data_clean_unwarped_new_ANTsReg -s 2.12 -mas ${input_dir}filtered_func_data_clean_unwarped_new_ANTsReg_masked ${output_dir}filtered_func_data_full_pipeline
#create a sample to check BET quickly on computer
mkdir $sample_dir
fslroi ${output_dir}filtered_func_data_full_pipeline ${sample_dir}sub${subjectID}_run${sessionID}_sample_smooth 0 1

#move and gunzip file in analysis dir ## ADD DELETE .GZ VERSION
mkdir $analysis_dir1
mkdir $analysis_dir2
mv ${output_dir}filtered_func_data_full_pipeline.nii.gz ${analysis_dir2}run${sessionID}.nii.gz
gunzip ${analysis_dir2}run${sessionID}.nii.gz
echo "Done for Subject ${subjectID} Session ${sessionID} at $(date +"%T")"