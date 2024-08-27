#!/bin/bash

# pull in the subject we should be working on
subjectID=$1
runID=$2


echo "Preparing subject ${subjectID} session ${runID}"
# the subject directory
subT2=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/T2_reoriented_brain.nii.gz
# Directory containing functionals, high-res reference scans, and field-maps
funcDirOg=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${subjectID}/Session${runID}.ica/
funcDir=${funcDirOg}unclean/
mkdir $funcDir
# Directory with run-specific files
runDir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/Session${runID}/
# the dwell time for fugue unwarping
#dwellTime=0.00054
dwellTime=0.000309


#######################
#  put EPI and fieldmaps in the same voxel space
echo " putting fieldmaps and EPI in the same space for ${subjectID} Session ${runID} at $(date +"%T")"

funcScanOg=${funcDirOg}filtered_func_data
funcScan=${funcDir}filtered_func_data
mapImage=${runDir}Session${runID}_rad
magImage=${runDir}Session${runID}_mag_brain

# we need the warped magnitude image since is more similar to the functional to realin
fugue -i ${magImage} --dwell=${dwellTime} --loadfmap=${mapImage} --unwarpdir=y --nokspace -s 0.5 -w ${magImage}_warped

# extract mean of the functional for realignement
fslmaths ${funcScanOg} -Tmean ${funcScan}_mean

# # perform bias correction on the mean functional
# /usr/local/ANTs/build/bin/N4BiasFieldCorrection -i ${funcScan}_mean.nii.gz -o ${funcScan}_mean_bias.nii.gz --convergence [100x100x100x100,0.0] -d 3 -s 3 -b [300]

# # perform bias correction on the magnitude image
# /usr/local/ANTs/build/bin/N4BiasFieldCorrection -i ${magImage}_warped.nii.gz -o ${magImage}_warped_bias.nii.gz --convergence [100x100x100x100,0.0] -d 3 -s 3 -b [300]

# # register the magnitude image of the fieldmap acquisition to EPI to get transformation matrix by using the mag as a reference since it's better quality
# flirt -in ${funcScan}_mean_bias -ref ${magImage}_warped_bias -dof 6 -cost normcorr -out ${magImage}_EPIalign -omat ${magImage}_EPIalign.mat

# # since we used the magnitude as a reference we need to invert the transformation matrix before applying it
# convert_xfm -omat ${magImage}_EPIalign_inverted.mat -inverse ${magImage}_EPIalign.mat

# # apply transformation matrix to the rad image
# flirt -in ${mapImage} -ref ${funcScan}_mean_bias -init ${magImage}_EPIalign_inverted.mat -applyxfm -out ${mapImage}_EPIalign

#######################
#  unwarp the functionals
echo "Unwarping functionals for Subject ${subjectID} Session ${runID} at $(date +"%T")"

fugue -i ${funcScanOg} --dwell=${dwellTime} --loadfmap=${mapImage}_EPIalign --unwarpdir=y -u ${funcScan}_unwarped_unclean

# extract a single volume for warped/unwarped comparison to the original T2
fslroi ${funcScanOg} ${funcScan}_sample 0 1
fslroi ${funcScan}_unwarped_unclean ${funcScan}_unwarped_sample 0 1

flirt -in ${funcScan}_sample -ref ${subT2} -dof 6 -out ${funcScan}_sample_alignT2 -omat ${funcScan}_sample_alignT2.mat
flirt -in ${funcScan}_unwarped_sample -ref ${subT2} -init ${funcScan}_sample_alignT2.mat -applyxfm -out ${funcScan}_unwarped_sample_alignT2
echo "done unwarping functional scan for Subject ${subjectID} Session ${runID} at $(date +"%T")"


# #######################
# #  unwarp the reference image

# echo "Unwarping reference scan for Subject ${subjectID} Session ${runID} at $(date +"%T")"
# refScan=${runDir}ref_Session${runID}_reoriented_brain_restore
# mapImage=${runDir}Session${runID}_rad
# fugue -i ${refScan} --dwell=${dwellTime} --loadfmap=${mapImage}_EPIalign --unwarpdir=y -u ${refScan}_unwarped_new

# # save T2-aligned warped & unwarped images for comparison
# flirt -in ${refScan} -ref ${subT2} -dof 6 -out ${refScan}_alignT2 -omat ${refScan}_alignT2.mat
# flirt -in ${refScan}_unwarped_new -ref ${subT2} -init ${refScan}_alignT2.mat -applyxfm -out ${refScan}_unwarped_alignT2
# echo "Done Unwarping reference for Subject ${subjectID} Session ${runID} at $(date +"%T")"