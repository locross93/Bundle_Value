#!/bin/bash

bundle_path=/Users/logancross/Documents/Bundle_Value/
analysis_prefix=decode_choice_minus_chance

for subj in 106 112
    do

	#/usr/local/fsl/bin/convert_xfm -omat ${bundle_path}mvpa/OpenfMRI_Format/sub${subj}/anatomy/T12CIT.mat -inverse ${bundle_path}mvpa/OpenfMRI_Format/sub${subj}/anatomy/CIT2T1.mat

	/usr/local/fsl/bin/flirt -in ${bundle_path}mvpa/analyses/sub${subj}/${analysis_prefix}.nii.gz \
	-applyxfm -init ${bundle_path}mvpa/OpenfMRI_Format/sub${subj}/anatomy/T12CIT.mat -out ${bundle_path}mvpa/analyses/sub${subj}/${analysis_prefix}_highres.nii.gz \
	-paddingsize 0.0 -interp trilinear -ref ${bundle_path}fmri/CIT_brains/CIT168_T1w_MNI.nii.gz

	fslmaths ${bundle_path}mvpa/analyses/sub${subj}/${analysis_prefix}_highres.nii.gz -s 2.12 -mas ${bundle_path}fmri/CIT_brains/CIT168_T1w_MNI.nii.gz ${bundle_path}mvpa/analyses/sub${subj}/${analysis_prefix}_highres_smooth.nii.gz

	gunzip -k ${bundle_path}mvpa/analyses/sub${subj}/${analysis_prefix}_highres_smooth.nii.gz
done