#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 16:07:11 2021

@author: logancross
"""
from mvpa2.suite import *
import pandas as pd
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 

subj = '107'

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

analysis_name = 'cross_decoding_rel_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

relative_value = True

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value)

#load xdecode scores to get voxels with signal
s2s_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
s2b_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b'
b2b_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b'
b2s_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s'

scores_s2s = h5load(s2s_file)
scores_s2b = h5load(s2b_file)
scores_b2b = h5load(b2b_file)
scores_b2s = h5load(b2s_file)

#load perm scores
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2s.npy'
perm_scores_s2s = np.load(vector_file)
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2b.npy'
perm_scores_s2b = np.load(vector_file)
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2b.npy'
perm_scores_b2b = np.load(vector_file)
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2s.npy'
perm_scores_b2s = np.load(vector_file)

sig_thr_s2s = np.percentile(perm_scores_s2s, 95)
sig_thr_s2b = np.percentile(perm_scores_s2b, 95)
sig_thr_b2b = np.percentile(perm_scores_b2b, 95)
sig_thr_b2s = np.percentile(perm_scores_b2s, 95)

sig_voxs_s2s = np.where(scores_s2s > sig_thr_s2s)[0]
sig_voxs_s2b = np.where(scores_s2b > sig_thr_s2b)[0]
sig_voxs_b2b = np.where(scores_b2b > sig_thr_b2b)[0]
sig_voxs_b2s = np.where(scores_b2s > sig_thr_b2s)[0]

#find voxels that are significant at generalizing in both categories at < 0.05
gen_intersect = np.intersect1d(sig_voxs_s2b, sig_voxs_b2s)
print 'Number of general value voxels',gen_intersect.shape[0]
scores_gen = np.zeros(len(scores_s2s))
scores_gen[gen_intersect] = 1

vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_sig_gen_p05'
h5save(vector_file,scores_gen)
nimg0 = map2nifti(fds, scores_gen)
nii_file0 = vector_file+'.nii.gz'
nimg0.to_filename(nii_file0)

#get a map of the clusters on this sig gen map
subj=107
cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p05.nii.gz \
--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p05_cluster.nii.gz \
--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_sig_gen_stats.txt \
--thresh=0.5 > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_sig_gen.txt

#subj=104
#cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p01.nii.gz \
#--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p01_cluster.nii.gz \
#--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_sig_gen_stats.txt \
#--thresh=0.5 > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_sig_gen.txt

#find the cluster index of the smallest cluster that meets the extent threshold
cluster_info_sig_gen = pd.read_table('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/cluster_info_sig_gen.txt')
cls_ext_thr = 10
thr_inds = np.where(cluster_info_sig_gen['Voxels'] >= cls_ext_thr)[0]
lowest_cluster = cluster_info_sig_gen['Cluster Index'][thr_inds[-1]]
print lowest_cluster

#threshold and binarize the cluster image based on this cluster number
#type out thr number in bash
subj=107
lowest_cluster=521
fslmaths /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p05_cluster.nii.gz -thr $lowest_cluster -bin /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_gen_p05_cluster_ethr10.nii.gz

#################################################################
#find voxels that are only significant within category at < 0.001 
sig_thr_s2s = np.percentile(perm_scores_s2s, 99.9)
sig_thr_s2b = np.percentile(perm_scores_s2b, 99.9)
sig_thr_b2b = np.percentile(perm_scores_b2b, 99.9)
sig_thr_b2s = np.percentile(perm_scores_b2s, 99.9)

sig_voxs_s2s = np.where(scores_s2s > sig_thr_s2s)[0]
sig_voxs_s2b = np.where(scores_s2b > sig_thr_s2b)[0]
sig_voxs_b2b = np.where(scores_b2b > sig_thr_b2b)[0]
sig_voxs_b2s = np.where(scores_b2s > sig_thr_b2s)[0]

sig_voxs_not_s2s = np.concatenate((sig_voxs_s2b, sig_voxs_b2b, sig_voxs_b2s), axis=None)
sig_s2s_only = np.array([ind for ind in sig_voxs_s2s if ind not in sig_voxs_not_s2s])

sig_voxs_not_b2b = np.concatenate((sig_voxs_s2s, sig_voxs_s2b, sig_voxs_b2s), axis=None)
sig_b2b_only = np.array([ind for ind in sig_voxs_b2b if ind not in sig_voxs_not_b2b])

scores_sig_s2s_only = np.zeros(len(scores_s2s))
scores_sig_s2s_only[sig_s2s_only] = 1

scores_sig_b2b_only = np.zeros(len(scores_b2b))
scores_sig_b2b_only[sig_b2b_only] = 1

vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_sig_s2s_only_p001'
h5save(vector_file,scores_sig_s2s_only)
nimg0 = map2nifti(fds, scores_sig_s2s_only)
nii_file0 = vector_file+'.nii.gz'
nimg0.to_filename(nii_file0)

vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_sig_b2b_only_p001'
h5save(vector_file,scores_sig_b2b_only)
nimg0 = map2nifti(fds, scores_sig_b2b_only)
nii_file0 = vector_file+'.nii.gz'
nimg0.to_filename(nii_file0)

#S2S
#get a map of the clusters on this sig s2s only map
subj=107
cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_s2s_only_p001.nii.gz \
--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_s2s_only_p001_cluster.nii.gz \
--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_sig_s2s_only_stats.txt \
--thresh=0.5 > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_sig_s2s_only.txt

#find the cluster index of the smallest cluster that meets the extent threshold
cluster_info_sig_s2s_only = pd.read_table('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/cluster_info_sig_s2s_only.txt')
cls_ext_thr = 10
thr_inds = np.where(cluster_info_sig_s2s_only['Voxels'] >= cls_ext_thr)[0]
lowest_cluster = cluster_info_sig_s2s_only['Cluster Index'][thr_inds[-1]]
print lowest_cluster

#threshold and binarize the cluster image based on this cluster number
#type out thr number in bash
subj=107
lowest_cluster=456
fslmaths /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_s2s_only_p001_cluster.nii.gz \
-thr $lowest_cluster -bin /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_s2s_only_p001_cluster_ethr10.nii.gz

#B2B
#get a map of the clusters on this sig b2b only map
subj=107
cluster --in=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_b2b_only_p001.nii.gz \
--oindex=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_b2b_only_p001_cluster.nii.gz \
--olmax=/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdec_sig_b2b_only_stats.txt \
--thresh=0.5 > /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/cluster_info_sig_b2b_only.txt

#find the cluster index of the smallest cluster that meets the extent threshold
cluster_info_sig_b2b_only = pd.read_table('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/cluster_info_sig_b2b_only.txt')
cls_ext_thr = 10
thr_inds = np.where(cluster_info_sig_b2b_only['Voxels'] >= cls_ext_thr)[0]
lowest_cluster = cluster_info_sig_b2b_only['Cluster Index'][thr_inds[-1]]
print lowest_cluster

#threshold and binarize the cluster image based on this cluster number
#type out thr number in bash
subj=107
lowest_cluster=464
fslmaths /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_b2b_only_p001_cluster.nii.gz \
-thr $lowest_cluster -bin /Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub${subj}/xdecode_sig_b2b_only_p001_cluster_ethr10.nii.gz

