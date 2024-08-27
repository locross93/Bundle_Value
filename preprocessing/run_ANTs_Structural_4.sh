#!/bin/bash

subScript=/home/lcross/Bundle_Value/code/preprocessing/ANTsAnatomicalWarp_4.sh

# Loop over subjects
# e.g: for subj in 001 002
for subj in 111-1 112-1 113-1 114-1
do
	qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=8:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N ANTS_Struct_Subject_${subj} -F "${subj}" ${subScript}
done