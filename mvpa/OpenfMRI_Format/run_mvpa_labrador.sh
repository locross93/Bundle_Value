#!/bin/bash

sessionScript=/state/partition1/home/lcross/Bundle_Value/mvpa/run_python_labrador.sh
analysis_type=MVPA_decode_choice
num_parts=5
output_dir=/state/partition1/home/lcross/Bundle_Value/ClusterOutput/

for sub in 105
    do
    # Break up analysis in parts
    for part in 1 2 3 4 5
    	do
     	job_name=${analysis_type}_sub${sub}_part${part}
		#qsub -q all.q -w e -N $job_name -l h_vmem=20G -pe smp 1 -o ${output_dir}${job_name}_out -e ${output_dir}${job_name}_err -v part=$part,num_parts=$num_parts,sub=$sub,game=$game, $sessionScript
    	qsub -q all.q -w e -N $job_name -l h_vmem=20G -pe smp 1 -o ${output_dir} -e ${output_dir} -v part=$part,num_parts=$num_parts,sub=$sub, $sessionScript
    done
done