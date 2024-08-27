#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:54:26 2018

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.searchlight import sphere_searchlight
import time
import os
import sys

sys.path.insert(0, '/home/lcross/Bundle_Value/mvpa')
os.chdir('/home/lcross/Bundle_Value/mvpa')

import mvpa_utils

###SCRIPT ARGUMENTS

analysis_name = 'rel_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

subj = int(sys.argv[1])

#which ds to use and which mask to use
#glm_ds_file = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/GLM_betas/all_trials_4D.nii.gz'
glm_ds_file = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/tstat_all_trials_4D.nii'
#mask_name = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
mask_name = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/mask.nii'

square_dsm_bool = False

remove_within_day = True

###SCRIPT ARGUMENTS END

#make targets with mvpa utils
if analysis_name == 'rel_value':
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=True, tolman=True)
else:
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=False, tolman=True)

target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, tolman=True)

res_stim_id = target_dsms['stim_id']
res_fvalue = target_dsms['fvalue']
res_tvalue = target_dsms['tvalue']
res_bvalue = target_dsms['bvalue']

if remove_within_day:
    res_day = target_dsms['day']
    btwn_run_inds = np.where(res_day.samples == 1)[0]

#feature selection - remove voxels with no variance
#voxel_vars = np.var(fds.samples, axis=0)
#nz_inds = np.where(voxel_vars != 0)[0]
#fds_fs = fds[:,nz_inds]

model_dsms = np.column_stack((res_stim_id, res_fvalue.samples.reshape(-1), \
                              res_tvalue.samples.reshape(-1), res_bvalue.samples.reshape(-1)))

rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation', keep_pairs=btwn_run_inds)

#searchlight
#enable progress bar
#if __debug__:
#    debug.active += ["SLC"]
    
sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3)

start_time = time.time()
print 'starting searchlight',time.time() - start_time
sl_map = sl_rsa_reg(fds)
sl_time = time.time() - start_time
print 'finished searchlight',sl_time

comp_speed = sl_time/fds.shape[1]
print 'Analyzed at a speed of ',comp_speed,'  per voxel'

#save
#scores_per_voxel = np.zeros([sl_map.shape[0]-1,fds.shape[1]])
#scores_per_voxel[:,nz_inds] = sl_map[:-1,:].samples
scores_per_voxel = sl_map[:-1,:].samples
vector_file = '/home/lcross/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/whole_brain_rsa_reg_btwn_day_rel_value'
h5save(vector_file,scores_per_voxel)

nimg0 = map2nifti(fds, scores_per_voxel[0,:])
nii_file0 = vector_file+'_stim_id.nii.gz'
nimg0.to_filename(nii_file0)

nimg1 = map2nifti(fds, scores_per_voxel[1,:])
nii_file1 = vector_file+'_fvalue.nii.gz'
nimg1.to_filename(nii_file1)

nimg2 = map2nifti(fds, scores_per_voxel[2,:])
nii_file2 = vector_file+'_tvalue.nii.gz'
nimg2.to_filename(nii_file2)

nimg3 = map2nifti(fds, scores_per_voxel[3,:])
nii_file3 = vector_file+'_bvalue.nii.gz'
nimg3.to_filename(nii_file3)




