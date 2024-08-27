#!/bin/bash

# pull in the subject we should be working on
subjectID=$1
sessionID=$2

echo "Preparing subject ${subjectID} session ${sessionID}"

# Directory containing functionals
func_dir=/home/lcross/Bundle_Value/preprocessing/FSL_ICA_BET/sub${subjectID}/Session${sessionID}.ica/
# point to of FIX code
fix_dir=/usr/local/fix/
# the classifier path
classifier_path=/home/shared/fix/classifiers/FIX_giovanniCoins_jeffCasinoUSA.RData
# the threshold
threshold=20

echo "started classification at $(date +"%T")"
# classify the components
${fix_dir}fix -c ${func_dir} ${classifier_path} ${threshold}
# remove bad ones
${fix_dir}fix -a ${func_dir}fix4melview_FIX_giovanniCoins_jeffCasinoUSA_thr${threshold}.txt -m
echo "finished cleanup at $(date +"%T")"