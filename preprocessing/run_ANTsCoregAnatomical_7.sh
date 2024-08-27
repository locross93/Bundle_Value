#!/bin/bash

# session level script
sessionScript=/home/lcross/Bundle_Value/code/preprocessing/ANTsCoregAnatomical_7.sh

for subj in 111-1 112-1 113-1 114-1
do

	# spawn session jobs to the cluster after the subject level work is complete
	qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=4:00:00 -M lcross@caltech.edu -m e -l nodes=1 -q batch -N ANTsCoregAnat_Sub_${subj}_Ses_${run} -F "${subj}" ${sessionScript}

done