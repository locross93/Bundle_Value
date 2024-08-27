#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:48:03 2021

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

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

analysis_name = 'abs_value'

subj_list = ['104','105','107','108','109','110','111','113','114']
for subj in subj_list:
    print subj
    vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/ridge_sl_'+analysis_name
    scores_per_voxel = h5load(vector_file)
    
    #make fds to get shape
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    scores_per_voxel = np.mean(scores_per_voxel, axis=0)
    h5save(vector_file,scores_per_voxel)
    nimg = map2nifti(fds, scores_per_voxel)
    nii_file = vector_file+'.nii.gz'
    nimg.to_filename(nii_file)