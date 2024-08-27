#!/bin/bash

subj=109
suffix=b2s
thr=0.19296295428393861

cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cross_decoding_rel_value_${suffix}.nii.gz \
--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cross_decoding_rel_value_${suffix}_cluster.nii.gz \
--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_${suffix}_stats.txt \
--thresh=${thr} > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_$suffix.txt

cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen.nii.gz \
--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_cluster.nii.gz \
--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_sig_gen_stats.txt \
--thresh=${thr} > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_sig_gen.txt

fslmaths /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub109/xdecode_sig_gen_cluster.nii.gz -thr 426 -bin /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub109/xdecode_sig_gen_cluster_ethr10.nii.gz