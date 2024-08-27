#!/bin/bash

#sessionScript=/home/lcross/Bundle_Value/mvpa/tolman_rsa_sl.py
sessionScript=/home/lcross/Bundle_Value/mvpa/tolman_xdecode_sl.py
analysis_name=MVPA_xdecode_rel_value
num_parts=10

for subj in 107
	do
	# Break up analysis in parts
	for part in 1 2 3 4 5 6 7 8 9 10
		do
			qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=10:00:00:00 -M lcross@caltech.edu -m e -l nodes=master -q batch -N ${analysis_name}_sub${subj}_part${part} -F "${part} ${num_parts} ${subj}" ${sessionScript}
	done
done