#!/bin/bash

# session level script
sessionScript=/home/lcross/Bundle_Value/code/preprocessing/applyANTsToFunc_8.sh

for subj in 111-1 111-2 111-3 112-1 112-2 112-3 113-1 113-2 113-3 114-1 114-2 114-3
do
	# Loop over runs, prep fieldmaps and reorient
	for run in 1 2 3 4 5
		do
			# spawn session jobs to the cluster after the subject level work is complete
			qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=4:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N ANTsFunc_Sub_${subj}_Ses_${run} -F "${subj} ${run}" ${sessionScript}

	done

done

