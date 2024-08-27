#!/bin/bash

# pull in the subject we should be working on
subjectID=$1
sessionID=$2
echo "Preparing subject ${subjectID} session ${sessionID}"

# directory containing fsl scripts and templates
code_dir=/home/lcross/Bundle_Value/code/preprocessing/
# Directory containing nifti data
data_dir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/Session${sessionID}/
# Output directory for preprocessed files
output_dir=/home/lcross/Bundle_Value/preprocessing/prepForICA/sub${subjectID}/Session${sessionID}/

# Run MELODIC template for current subject and current run
echo "Started MELODIC for run ${sessionID} at $(date +"%T")"
melodicTempplate=${output_dir}ICA.fsf
cp ${code_dir}ICA.fsf $melodicTempplate
sed -i -e 's/subXXX/'sub$subjectID'/g' $melodicTempplate
sed -i -e 's/SessionYYY/'Session$sessionID'/g' $melodicTempplate
# correct the number of volumes if necessary
nvols=`fslnvols ${data_dir}Session${sessionID}_reoriented`
if [ $nvols -lt 558 ]
then
	echo "ERROR: LESS THAN 558 VOLUMES"
fi
echo "Number of volumes ${nvols}"
sed -i -e 's/ZZZ/'$nvols'/' $melodicTempplate
feat $melodicTempplate
echo "Finished MELODIC for run ${sessionID} at $(date +"%T")"