#!/bin/bash

subScript=/home/lcross/Bundle_Value/code/preprocessing/prepForICA_2.sh

# Loop over subjects
# e.g: for subj in 001 002
for subj in 113-1
do
	qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=4:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N prepICA_Subject_${subj} -F "${subj}" ${subScript}
done