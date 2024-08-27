#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 15:48:55 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj = '107'

#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','107','108','109','110','111','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Trinket item']

analysis_name = 'rel_value_wbrain'

for subj in subj_list:
    print subj
    
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/pfc_full_bin.nii.gz'
    
    #fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, relative_value=True)
    
    #zscore targets
    fds.targets = scipy.stats.zscore(fds.targets)
    
    #define model
    #alp=3
    #sk_ridge = linear_model.Ridge(alpha=10*alp)
    #r_scorer = make_scorer(get_correlation)
    #run_num = 15
    
    #define model
    alp=3
    ridge = RidgeReg(lm=10**alp)
    cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
    
    if __debug__:
        debug.active += ["SLC"]
        
    sl_ridge = sphere_searchlight(cv, radius=3)
    
    start_time = time.time()
    print 'starting searchlight',time.time() - start_time
    sl_map = sl_ridge(fds)
    sl_time = time.time() - start_time
    print 'finished searchlight',sl_time
    
    # reverse map scores back into nifti format
    scores_per_voxel = np.mean(sl_map.samples, axis=0)
    vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/ridge_sl_'+analysis_name
    h5save(vector_file,scores_per_voxel)
    nimg = map2nifti(fds, scores_per_voxel)
    nii_file = vector_file+'.nii.gz'
    nimg.to_filename(nii_file)