#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:54:26 2018

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import time
import os
import sys

sys.path.insert(0, '/home/lcross/Bundle_Value/mvpa')
os.chdir('/home/lcross/Bundle_Value/mvpa')

import mvpa_utils

###SCRIPT ARGUMENTS

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

subj = int(sys.argv[1])

#which ds to use and which mask to use
#glm_ds_file = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/GLM_betas/all_trials_4D.nii.gz'
glm_ds_file = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/tstat_all_trials_4D.nii'
mask_name = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

###SCRIPT ARGUMENTS END

#make targets with mvpa utils
train_on='default'
test_on='default'
if analysis_name == 'rel_value':
    relative_value=True
else:
    relative_value=False
tolman=True
fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on, test_on, relative_value, tolman)

#feature selection - remove voxels with no variance
voxel_vars = np.var(fds.samples, axis=0)
nz_inds = np.where(voxel_vars != 0)[0]
fds_fs = fds[:,nz_inds]

ridge = RidgeReg(lm=10**3)
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)

#searchlight
# enable progress bar
#if __debug__:
#    debug.active += ["SLC"]
    
sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
                             postproc=mean_sample())


start_time = time.time()
print 'starting searchlight',time.time() - start_time
sl_map = sl(fds_fs)
sl_time = time.time() - start_time
print 'finished searchlight',sl_time

comp_speed = sl_time/fds_fs.shape[1]
print 'Analyzed at a speed of ',comp_speed,'  per voxel'

#save map
scores_per_voxel = np.zeros(fds.shape[1])
scores_per_voxel[nz_inds] = sl_map.samples

#reverse map scores back into nifti format
vector_file = '/home/lcross/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_'+analysis_name
h5save(vector_file,scores_per_voxel)
nimg = map2nifti(fds, scores_per_voxel)
nii_file = vector_file+'.nii.gz'
nimg.to_filename(nii_file)