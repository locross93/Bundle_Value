#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 14:56:29 2019

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import mvpa_utils 

##started 1/10/18
#analysis_list = [[101, 'abs_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
#                  [102, 'abs_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
#                  [103, 'abs_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
#                  [101, 'item_value', ['Food item', 'Trinket item']],
#                  [102, 'item_value', ['Food item', 'Trinket item']],
#                  [103, 'item_value', ['Food item', 'Trinket item']]
#        ]

#started 1/11/18
analysis_list = [[101, 'rel_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
                  [102, 'rel_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
                  [103, 'rel_value', ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']],
                  [101, 'bundle_value', ['Food bundle','Trinket bundle','Mixed bundle']],
                  [102, 'bundle_value', ['Food bundle','Trinket bundle','Mixed bundle']],
                  [103, 'bundle_value', ['Food bundle','Trinket bundle','Mixed bundle']],
                  [101, 'food_value', ['Food item']],
                  [102, 'food_value', ['Food item']],
                  [103, 'food_value', ['Food item']],
                  [101, 'trinket_value', ['Trinket item']],
                  [102, 'trinket_value', ['Trinket item']],
                  [103, 'trinket_value', ['Trinket item']]
        ]

for analysis in analysis_list:
    
    print 'Starting analysis ',analysis
    
    ###SCRIPT ARGUMENTS
    
    subj = analysis[0]
    
    analysis_name = analysis[1]
    
    #which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
    conditions = analysis[2] 
    
    #which ds to use and which mask to use
    glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    
    ###SCRIPT ARGUMENTS END
    
    #make targets with mvpa utils
    train_on='default'
    test_on='default'
    if analysis_name == 'rel_value':
        relative_value=True
    else:
        relative_value=False
    tolman=False
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on, test_on, relative_value, tolman)
    
    #feature selection - remove voxels with no variance
    voxel_vars = np.var(fds.samples, axis=0)
    nz_inds = np.where(voxel_vars != 0)[0]
    fds_fs = fds[:,nz_inds]
    
    #searchlight
    # enable debug output for searchlight call
    if __debug__:
        debug.active += ["SLC"]
        
    alp=3
    ridge = RidgeReg(lm=10**alp)
    cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
    
    ridge_sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
                                 postproc=mean_sample())
    
    start_time = time.time()
    print 'starting searchlight',time.time() - start_time
    res_sl = ridge_sl(fds_fs)
    print 'finished searchlight',time.time() - start_time
    
    #save map
    scores_per_voxel = np.zeros(fds.shape[1])
    scores_per_voxel[nz_inds] = res_sl.samples
    
    ## reverse map scores back into nifti format
    vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_'+analysis_name
    h5save(vector_file,scores_per_voxel)
    nimg = map2nifti(fds, scores_per_voxel)
    nii_file = vector_file+'.nii.gz'
    nimg.to_filename(nii_file)