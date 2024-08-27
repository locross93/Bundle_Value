#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 14:36:31 2018

@author: logancross
"""

from mvpa2.suite import *
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

bundle_path='/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/'
bundle_handle = OpenFMRIDataset(bundle_path)

subj = 102
task = 1
run = 1
model = 1

#mask_fname='/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/T1_ANTScoreg_mask.nii.gz'
mask_fname='/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds = bundle_handle.get_bold_run_dataset(subj, task, run, mask=mask_fname)

run_datasets = []
for run_id in bundle_handle.get_task_bold_run_ids(task)[subj]:
#for run_id in range(1,2):
    print 'here '+str(run_id)
    # load design info for this run
    run_events = bundle_handle.get_bold_run_model(model, subj, run_id)
    # load BOLD data for this run (with masking); add 0-based chunk ID
    run_ds = bundle_handle.get_bold_run_dataset(subj, task, run_id,
                                               chunks=run_id -1,
                                               mask=mask_fname)
    # convert event info into a sample attribute and assign as 'targets'
    run_ds.sa['targets'] = events2sample_attr(run_events, run_ds.sa.time_coords, noinfolabel='rest')
    # additional time series pre  processing can go here
    run_datasets.append(run_ds)
fds = vstack(run_datasets, a=0) 

fds.save('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/raw_voxels_pfc.hdf5')