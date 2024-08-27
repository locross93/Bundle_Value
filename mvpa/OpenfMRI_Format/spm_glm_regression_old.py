#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:30:09 2019

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
from os import listdir

subj = 101

onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model001/onsets/'

dir_onsets = listdir(onsets_folder)
if dir_onsets[0] == '.DS_Store':
    dir_onsets.remove('.DS_Store')

value_list = []
chunks_list = []
run_num = 0
for run in dir_onsets:
    temp_folder = onsets_folder+run
    cond001_onsets = np.genfromtxt(temp_folder+'/cond001.txt')
    cond002_onsets = np.genfromtxt(temp_folder+'/cond002.txt')
    timing = np.concatenate((cond001_onsets[:,0], cond002_onsets[:,0]))
    sort_time_inds = np.argsort(timing)
    value = np.concatenate((cond001_onsets[:,2], cond002_onsets[:,2]))
    value = value[sort_time_inds]
    value_list.append(value)
    chunks = run_num*np.ones([len(value)])
    chunks_list.append(chunks)
    run_num+=1
    
value_allruns = np.asarray([item for sublist in value_list for item in sublist]).astype(int)
chunks_allruns = np.asarray([item for sublist in chunks_list for item in sublist]).astype(int)  
    

glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial/test_4D_zip.nii.gz'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds_old = fmri_dataset(samples=glm_ds_file, targets=value_allruns, chunks=chunks_allruns, mask=mask_name)

#searchlight
# enable debug output for searchlight call
#if __debug__:
#    debug.active += ["SLC"]
#    
#alp=3
#ridge = RidgeReg(lm=10**alp)
#cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
#
#ridge_sl = sphere_searchlight(cv, radius=4, space='voxel_indices',
#                             postproc=mean_sample())
#
#start_time = time.time()
#print 'starting searchlight',time.time() - start_time
#res_sl = ridge_sl(fds)
#print 'finished searchlight',time.time() - start_time
#
#scores_per_voxel = res_sl.samples
#
## reverse map scores back into nifti format
#nimg = map2nifti(fds, scores_per_voxel)
#nii_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_value.nii.gz'
#nimg.to_filename(nii_file)