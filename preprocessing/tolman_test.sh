#!/bin/bash

sessionScript=/home/lcross/Bundle_Value/code/preprocessing/preprocessing_nipype_tolman.py

qsub -o ~/Bundle_Value/ClusterOutput -j oe -l walltime=20:00:00 -M lcross@caltech.edu -m e -l nodes=1:ppn=5 -q batch -v PATH=$PATH -N BundleValue_preproc ${sessionScript}