#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 13:45:51 2022

@author: logancross
"""


from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import seaborn as sns

#need to re do this with 100 full maps per subject to preserve the spatial smoothness

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 

relative_value = True

analysis_name = 'cross_decoding_rel_value'

cit_file = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_brains/CIT168_T1w_MNI_lowres.nii.gz'
cit_ref = fmri_dataset(samples=cit_file, mask=cit_file)
cit_voxel_inds = cit_ref.fa.voxel_indices

cit_set = set([tuple(x) for x in cit_voxel_inds])
num_vox = cit_ref.shape[1]
num_subs = len(subj_list)

#mean_map_file_s2s = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2S/mean_map_s2s.nii.gz'
#mean_map_s2s = fmri_dataset(samples=mean_map_file_s2s)
mean_map_file_s2b = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/mean_map2_s2b.nii.gz'
mean_map_s2b = fmri_dataset(samples=mean_map_file_s2b)
#mean_map_file_b2b = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2B/mean_map_b2b.nii.gz'
#mean_map_b2b = fmri_dataset(samples=mean_map_file_b2b)
#mean_map_file_b2s = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/B2S/mean_map_b2s.nii.gz'
#mean_map_b2s = fmri_dataset(samples=mean_map_file_b2s)

num_perms_per_sub = 100
#perms_all_subs_s2s = np.zeros([num_perms_per_sub*num_subs,num_vox])
perms_all_subs_s2b = np.zeros([num_perms_per_sub*num_subs,num_vox])
#perms_all_subs_b2b = np.zeros([num_perms_per_sub*num_subs,num_vox])
#perms_all_subs_b2s = np.zeros([num_perms_per_sub*num_subs,num_vox])
perm_chunks = np.zeros([num_perms_per_sub*num_subs])

count = -1
for sub_num,subj in enumerate(subj_list):
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
    sub_voxel_inds = fds.fa.voxel_indices
    
    sub_set = set([tuple(x) for x in sub_voxel_inds])
    sub_inter_inds = [i for i in range(len(cit_set)) if tuple(cit_voxel_inds[i,:]) in sub_set]
    num_sub_voxs = len(sub_inter_inds)
    
#    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2s.npy'
#    perm_scores_s2s = np.load(vector_file)
    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2b.npy'
    perm_scores_s2b = np.load(vector_file)
#    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2b.npy'
#    perm_scores_b2b = np.load(vector_file)
#    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2s.npy'
#    perm_scores_b2s = np.load(vector_file)
    
    for perm_num in range(num_perms_per_sub):
        count+=1
#        temp_map = np.random.choice(perm_scores_s2s, num_sub_voxs, replace=False)
#        perms_all_subs_s2s[count,sub_inter_inds] = temp_map
        
        temp_map = np.random.choice(perm_scores_s2b, num_sub_voxs, replace=False)
        perms_all_subs_s2b[count,sub_inter_inds] = temp_map
        
#        temp_map = np.random.choice(perm_scores_b2b, num_sub_voxs, replace=False)
#        perms_all_subs_b2b[count,sub_inter_inds] = temp_map
#        
#        temp_map = np.random.choice(perm_scores_b2s, num_sub_voxs, replace=False)
#        perms_all_subs_b2s[count,sub_inter_inds] = temp_map
        
        perm_chunks[count] = sub_num
        
np.save('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/perms_all_subs_s2b_2',perms_all_subs_s2b)
np.save('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/perm_chunks',perm_chunks)
        
#sa_dict = {'chunks': perm_chunks}
#fa_dict = {'voxel_indices': cit_voxel_inds}   
#perm_map_s2b = dataset_wizard(perms_all_subs_s2b, sa=sa_dict, fa=fa_dict)
#
#clthr = GroupClusterThreshold(feature_thresh_prob=.005,n_bootstrap=100000,fwe_rate=.05)
#clthr.train(perm_map_s2b)

mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/lowres/pfc.nii.gz'
#mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/AAL_lowres/Frontal_Med_Orb.nii.gz'
brain_mask = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_brains/CIT168_T1w_MNI_lowres.nii.gz'
masked = fmri_dataset(mask_name, mask=brain_mask)
reshape_masked=masked.samples.reshape(cit_ref.shape[1])
reshape_masked=reshape_masked.astype(bool)
mask_map = mask_mapper(mask=reshape_masked)

mask_slice = mask_map[1].slicearg
mask_inds = np.where(mask_slice == 1)[0]

sa_dict = {'chunks': perm_chunks}
fa_dict = {'voxel_indices': cit_voxel_inds[mask_inds]}   
perm_map_s2b = AttrDataset(perms_all_subs_s2b[:,mask_inds], sa=sa_dict, fa=fa_dict)

clthr = GroupClusterThreshold(feature_thresh_prob=.0005,n_bootstrap=10000,fwe_rate=.05)
clthr.train(perm_map_s2b)

mask_map_s2b = mean_map_s2b[:,mask_inds]
res_s2b = clthr(mask_map_s2b)

res_wholebrain = np.zeros(cit_ref.shape)
res_wholebrain[:,mask_inds] = res_s2b.fa.clusters_fwe_thresh

nimg = map2nifti(cit_ref, res_wholebrain)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/clusterthr_s2b.nii.gz'
nimg.to_filename(nii_file)

res_wholebrain2 = np.zeros(cit_ref.shape)
mask_map_s2b = mean_map_s2b[:,mask_inds]
res_wholebrain2[:,mask_inds] = mask_map_s2b

nimg = map2nifti(cit_ref, res_wholebrain2)
nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/Group/Cross_decoding/S2B/temp_s2b.nii.gz'
nimg.to_filename(nii_file)

