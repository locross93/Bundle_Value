#!/bin/bash

subj=109
analysis_path=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/
mask_name=/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub${subj}/anatomy/T2_reoriented_brain_ANTsCoreg.nii.gz
 
for analysis_prefix in cross_decoding_rel_value_s2s cross_decoding_rel_value_s2b cross_decoding_rel_value_b2b cross_decoding_rel_value_b2s
	do
		fslmaths ${analysis_path}${analysis_prefix} -s 1.5 -mas $mask_name ${analysis_path}${analysis_prefix}_smooth
done