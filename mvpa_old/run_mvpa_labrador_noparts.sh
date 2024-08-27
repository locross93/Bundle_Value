#!/bin/bash

sessionScript=/state/partition1/home/lcross/Bundle_Value/mvpa/run_python_labrador.sh
#analysis_type=rsa_normalized_codes
#analysis_type=xdecode_abs_rel
analysis_type=MVPA_xdecode_roi
output_dir=/state/partition1/home/lcross/Bundle_Value/ClusterOutput/

for sub in 101 102 103
    do
 	job_name=${analysis_type}_sub${sub}
	qsub -q all.q -w e -N $job_name -l h_vmem=20G -l mem_free=10G -pe smp 1 -o ${output_dir} -e ${output_dir} -v sub=$sub, $sessionScript
	#qsub -q all.q -w e -N $job_name -l h_vmem=40G -l mem_free=30G -pe smp 1 -o ${output_dir} -e ${output_dir} -v sub=$sub, $sessionScript
	#qsub -q all.q -w e -N $job_name -l h_vmem=20G -l mem_free=10G -pe smp 1 -o ${output_dir} -e ${output_dir} $sessionScript
done