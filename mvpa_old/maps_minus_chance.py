#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 12:27:43 2021

@author: logancross
"""

from mvpa2.suite import *
import mvpa_utils 
import os

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['106','112']

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
    
    analysis_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/decode_choice'
    decode_scores = h5load(analysis_file)
    
    decode_minus_chance = decode_scores - 0.50
    
    if save:
        vector_file = analysis_file+'_minus_chance'
        h5save(vector_file,decode_minus_chance)
        if save_nifti:
            nimg = map2nifti(fds, decode_minus_chance)
            nii_file = vector_file+'.nii.gz'
            nimg.to_filename(nii_file)