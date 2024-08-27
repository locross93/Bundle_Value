#!/bin/bash

#sessionScript=/home/lcross/Bundle_Value/mvpa/tolman_rsa_sl.py
sessionScript=/home/lcross/Bundle_Value/mvpa/tolman_xdecode_sl.py
analysis_name=MVPA_xdecode_rel_value

for subj in 114
	do
		qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=5:00:00:00 -M lcross@caltech.edu -m e -l nodes=master -q batch -N ${analysis_name}_sub${subj} -F "${subj}" ${sessionScript}

done