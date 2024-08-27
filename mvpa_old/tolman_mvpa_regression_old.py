#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 12:54:26 2018

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import time

subj = 101

evds = h5load('/home/lcross/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/glm_ds_pfc.hdf5')

evds_sub = evds[:,:100]

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
sl_map = sl(evds_sub)
sl_time = time.time() - start_time
print 'finished searchlight',sl_time

comp_speed = sl_time/evds.shape[1]
print 'Analyzed at a speed of ',comp_speed,'  per voxel'

##save map
#scores_per_voxel = sl_map.samples

# reverse map scores back into nifti format
#nimg = map2nifti(evds, scores_per_voxel)
#nii_file = '/home/lcross/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_value.nii.gz'
#nimg.to_filename(nii_file)