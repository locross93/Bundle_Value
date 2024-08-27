#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 15:22:56 2021

@author: logancross
"""

from mvpa2.suite import *
import mvpa_utils 
import os

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj = '104'

#Save?
save = True
save_nifti=True

delete_parts=True

parts = 5

analysis_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
#analysis_prefices = ['cross_decoding_rel_value_s2s','cross_decoding_rel_value_s2b','cross_decoding_rel_value_b2b','cross_decoding_rel_value_b2s']
analysis_prefices = ['cross_decoding_rel_value_s2s']

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
#brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
relative_value = True
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
    
##SCRIPT ARGUMENTS END

for analysis_prefix in analysis_prefices:
    print analysis_prefix
    scores_per_voxel = np.array([])
    for i in range(parts):
        i = i + 1
        temp_array = h5load(analysis_path+analysis_prefix+'_part'+str(i))
        scores_per_voxel = np.append(scores_per_voxel, temp_array)
    #get rid of NaNs
    nan_inds = np.where(np.isnan(scores_per_voxel))[0]
    scores_per_voxel[nan_inds] = 0
    
    assert scores_per_voxel.shape[0] == fds.shape[1]
        
    if save:
        vector_file = analysis_path+analysis_prefix
        h5save(vector_file,scores_per_voxel)
        if save_nifti:
            nimg = map2nifti(fds, scores_per_voxel)
            nii_file = vector_file+'.nii.gz'
            nimg.to_filename(nii_file)
            
    if delete_parts:
        for i in range(parts):
            i = i + 1
            os.remove(analysis_path+analysis_prefix+'_part'+str(i))