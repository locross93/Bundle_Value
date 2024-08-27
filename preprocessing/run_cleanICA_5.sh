#!/bin/bash

# session level script
sessionScript=/home/lcross/Bundle_Value/code/preprocessing/cleanICA_5.sh

for subj in 111-2
do
	# Loop over runs, prep fieldmaps and reorient
	for run in 4
		do
			# spawn session jobs to the cluster after the subject level work is complete
			qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=2:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N cleanICA_Sub_${subj}_Ses_${run} -F "${subj} ${run}" ${sessionScript}

	done

done