#!/bin/bash

sessionScript=/state/partition1/home/lcross/Bundle_Value/mvpa/run_python_labrador2.sh
analysis_type=concat_data
num_parts=5
output_dir=/state/partition1/home/lcross/Bundle_Value/ClusterOutput/

qsub -q all.q -w e -N $analysis_type -l h_vmem=20G -pe smp 1 -o ${output_dir} -e ${output_dir} $sessionScript