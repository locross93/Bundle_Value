#!/bin/bash

# session level script
sessionScript=/home/lcross/Bundle_Value/code/preprocessing/applySmooth_manualICA_9.sh

for subj in 105-1 105-2 105-3
do
	# Loop over runs
	for run in 1 2 3 4 5
		do
			# spawn session jobs to the cluster after the subject level work is complete
			qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=2:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N Smooth_${subj}_${run} -F "${subj} ${run}" ${sessionScript}

	done

done