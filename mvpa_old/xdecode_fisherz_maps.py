#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 10:48:33 2022

@author: logancross
"""

from mvpa2.suite import *
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['105','106','107','108','109','110','111','112','113','114']
subj_list = ['101']

save = True
save_nifti = True

for subj in subj_list:
    print subj
    
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    relative_value = True
    conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    analysis_name = '                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      '

    s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
    s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b'
    b2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b'
    b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s'
    
    scores_s2s = h5load(s2s_file)
    scores_s2b = h5load(s2b_file)
    scores_b2b = h5load(b2b_file)
    scores_b2s = h5load(b2s_file)
    
    r_z_s2s = np.arctanh(scores_s2s)
    r_z_s2b = np.arctanh(scores_s2b)
    r_z_b2b = np.arctanh(scores_b2b)
    r_z_b2s = np.arctanh(scores_b2s)
    
    if save:
        vector_file_s2s = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s_fisherz'
        h5save(vector_file_s2s,r_z_s2s)
        vector_file_s2b = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b_fisherz'
        h5save(vector_file_s2b,r_z_s2b)
        vector_file_b2b = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b_fisherz'
        h5save(vector_file_b2b,r_z_b2b)
        vector_file_b2s = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s_fisherz'
        h5save(vector_file_b2s,r_z_b2s)
        if save_nifti:
            nimg = map2nifti(fds, r_z_s2s)
            nii_file = vector_file_s2s+'.nii.gz'
            nimg.to_filename(nii_file)
            nimg = map2nifti(fds, r_z_s2b)
            nii_file = vector_file_s2b+'.nii.gz'
            nimg.to_filename(nii_file)
            nimg = map2nifti(fds, r_z_b2b)
            nii_file = vector_file_b2b+'.nii.gz'
            nimg.to_filename(nii_file)
            nimg = map2nifti(fds, r_z_b2s)
            nii_file = vector_file_b2s+'.nii.gz'
            nimg.to_filename(nii_file)