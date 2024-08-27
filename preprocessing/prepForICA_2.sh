#!/bin/bash

#modify for every subject if needed
anat_bet_param=0.4
fm_bet_param=0.35
ref_bet_param=0.35

# pull in the subject we should be working on
subjectID=$1
echo "Preparing subject ${subjectID} for ICA"

# directory containing fsl scripts and templates
code_dir=/home/lcross/Bundle_Value/code/preprocessing/
# Directory containing nifti data
subID_short=${subjectID:0:3}
day=${subjectID:4:5}
data_dir=/home/lcross/Bundle_Value/preprocessing/bidsdata/sub-${subID_short}/ses-day${day}/
anat_data_dir=/home/lcross/Bundle_Value/preprocessing/bidsdata/sub-${subID_short}/ses-day1/
# Output directory for preprocessed files
output_dir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/
mkdir ${output_dir}
sample_dir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sample_bet/

# ###################
# T1 SCAN: Reorient and extract brain
# Expects a file called T1 in the source directory
echo "Started working on T1 scan at $(date +"%T")"
#run-01 is bias corrected scan
T1scan=${anat_data_dir}anat/sub-${subID_short}_ses-day1_T1.nii.gz
T1_reoriented=${output_dir}T1_reoriented
T1_reoriented_brain=${T1_reoriented}_brain
# Reorient T1 scan to standard, and extract brain
fslreorient2std $T1scan $T1_reoriented
bet $T1_reoriented $T1_reoriented_brain -f $anat_bet_param -R
#copy to sample bet to check quickly on computer
cp ${T1_reoriented_brain}.nii.gz ${sample_dir}sub${subjectID}_T1_sample_bet.nii.gz
echo "Done working on T1 scan at $(date +"%T")"
	
# ###################
# T2 SCAN: Reorient and extract brain
# Expects a file called T2 in the source directory
echo "Started working on T2 scan at $(date +"%T")"
#run-01 is bias corrected scan
T2scan=${anat_data_dir}anat/sub-${subID_short}_ses-day1_T2.nii.gz
T2_reoriented=${output_dir}T2_reoriented
T2_reoriented_brain=${T2_reoriented}_brain
# Reorient T2 scan to standard, and extract brain
fslreorient2std $T2scan $T2_reoriented	
bet $T2_reoriented $T2_reoriented_brain -f $anat_bet_param -R
#copy to sample bet to check quickly on computer
cp ${T2_reoriented_brain}.nii.gz ${sample_dir}sub${subjectID}_T2_sample_bet.nii.gz
# echo "Done working on T2 scan at $(date +"%T")"

# run through each session
for sessionID in 1 2 3 4 5
do
	# single session output directories
	outSession_dir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/Session${sessionID}/
	mkdir ${outSession_dir}
	###################
	# Field map generation -  takes 5-10 min
	# Expects files named pos_Session<X> and neg_Session<X>
	mergedFM=${outSession_dir}FM_Session${sessionID}_reoriented_merged
	# reorient the fieldmaps, and merge into a single image
	fslreorient2std ${data_dir}fmap/sub-${subID_short}_ses-day${day}_run-0${sessionID}_dir-AP_epi.nii.gz ${outSession_dir}pos_Session${sessionID}_reoriented
	fslreorient2std ${data_dir}fmap/sub-${subID_short}_ses-day${day}_run-0${sessionID}_dir-PA_epi.nii.gz ${outSession_dir}neg_Session${sessionID}_reoriented
	fslmerge -t $mergedFM ${outSession_dir}pos_Session${sessionID}_reoriented ${outSession_dir}neg_Session${sessionID}_reoriented

	# Run TOPUP to generate fieldmap
	echo "started building fieldmaps at $(date +"%T")"
	topup --imain=$mergedFM --datain=${code_dir}fieldmaps_datain_bundle.txt --config=b02b0.cnf --fout=${outSession_dir}FM_Session${sessionID} --iout=${mergedFM}unwarped
	# Scale fieldmap from Hz to rad/s
	fslmaths ${outSession_dir}FM_Session${sessionID} -mul 6.28 ${outSession_dir}Session${sessionID}_rad
	# extract field magnitude, and extract brain
	fslmaths ${mergedFM}unwarped -Tmean ${outSession_dir}Session${sessionID}_mag
	bet ${outSession_dir}Session${sessionID}_mag ${outSession_dir}Session${sessionID}_mag_brain -f $fm_bet_param -R
	echo "finished building fieldmaps at $(date +"%T")"

	###################
	# extract brain from functional multi-band reference scan
	echo "Started on reference scan for run ${sessionID} at $(date +"%T")"
	refScan_reorient=${outSession_dir}ref_Session${sessionID}_reoriented
	refScan_brain=${refScan_reorient}_brain
	# reorient and extract the brain image
	fslreorient2std ${data_dir}ref/sub-${subID_short}_ses-day${day}_run-0${sessionID}_SBRef.nii.gz $refScan_reorient
	bet $refScan_reorient $refScan_brain -f $ref_bet_param -R -m 
	# segment and bias correct the extracted brain
	fast -n 4 -t 2 -B --out=$refScan_brain $refScan_brain
	echo "Done reference scan for run ${sessionID} at $(date +"%T")"
	# Delete unnecessary files
	rm "$outSession_dir"*_seg*
	rm "$outSession_dir"*_pve*
	rm "$outSession_dir"*_mixeltype*

	###################
	# reorient functional multi-band scans
	echo "Started reorientation on functional scans for run ${sessionID} at $(date +"%T")"
	# Reorient MB scans to standard
	fslreorient2std ${data_dir}func/sub-${subID_short}_ses-day${day}_run-0${sessionID}_bold.nii.gz ${outSession_dir}Session${sessionID}_reoriented
	# echo "Done reorientation on functional scans for run ${run} at $(date +"%T")"

	###########################
	#Mask functionals with mask generated from reference
	refScan_mask=${refScan_brain}_mask
	func_scan=${outSession_dir}Session${sessionID}_reoriented
	fslmaths ${func_scan} -mas ${refScan_mask} ${func_scan}_brain 
	#create a sample to check BET quickly on computer
	fslroi ${func_scan}_brain ${sample_dir}sub${subjectID}_run${sessionID}_sample_bet 0 1

done