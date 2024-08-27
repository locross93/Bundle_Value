#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:35:36 2019

@author: logancross
"""
from mvpa2.suite import *
import matplotlib.pyplot as plt
from os import listdir

def make_targets(conditions, categories, train_on, test_on, relative_value=False, tolman=False):
    
    #create dict conditions name to number conversion
    cond_dict = {
		'Food item' : 1,
		'Trinket item' : 2,
		'Food bundle' : 3,
		'Trinket bundle' : 4,
		'Mixed bundle' : 5
	}
    
    cond_nums = [cond_dict[condition] for condition in conditions]
    
    subj = 101
    
    if tolman:
        onsets_folder = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    else:
        onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    
    trial_list = []
    chunks_list = []
    run_num = 0
    for run in dir_onsets:
        temp_folder = onsets_folder+run
        cond001_onsets = np.genfromtxt(temp_folder+'/cond001.txt')
        #add one column to signify condition
        cond001_onsets = np.column_stack([cond001_onsets, 1*np.ones(len(cond001_onsets))])
        cond002_onsets = np.genfromtxt(temp_folder+'/cond002.txt')
        cond002_onsets = np.column_stack([cond002_onsets, 2*np.ones(len(cond002_onsets))])
        cond003_onsets = np.genfromtxt(temp_folder+'/cond003.txt')
        cond003_onsets = np.column_stack([cond003_onsets, 3*np.ones(len(cond003_onsets))])
        cond004_onsets = np.genfromtxt(temp_folder+'/cond004.txt')
        cond004_onsets = np.column_stack([cond004_onsets, 4*np.ones(len(cond004_onsets))])
        cond005_onsets = np.genfromtxt(temp_folder+'/cond005.txt')
        cond005_onsets = np.column_stack([cond005_onsets, 5*np.ones(len(cond005_onsets))])
        timing = np.concatenate((cond001_onsets[:,0], cond002_onsets[:,0], cond003_onsets[:,0], cond004_onsets[:,0], cond005_onsets[:,0]))
        sort_time_inds = np.argsort(timing)
        all_trials = np.concatenate((cond001_onsets, cond002_onsets, cond003_onsets, cond004_onsets, cond005_onsets))
        all_trials = all_trials[sort_time_inds,:]
        trial_list.append(all_trials)
        chunks = run_num*np.ones([len(value)])
        chunks_list.append(chunks)
        run_num+=1
        
    trials_allruns = np.asarray([item for sublist in trial_list for item in sublist])
    chunks_allruns = np.asarray([item for sublist in chunks_list for item in sublist]).astype(int) 
    value_allruns = trials_allruns[:,2]
    
    if relative_value:
        #if relative value, z score across individual item trials and separately z score across bundle trials
        num_trials = len(trials_allruns)
        cond_by_trial = trials_allruns[:,3]
        item_inds = [c for c in range(num_trials) if cond_by_trial[c] in [1, 2]]
        bundle_inds = [c for c in range(num_trials) if cond_by_trial[c] in [3, 4, 5]]
        
        zitem_values = scipy.stats.zscore(value_allruns[item_inds])
        value_allruns[item_inds] = zitem_values
        zbundle_values = scipy.stats.zscore(value_allruns[bundle_inds])
        value_allruns[bundle_inds] = zbundle_values
    
    #load fmri dataset with these values as targets
    fds = fmri_dataset(samples=glm_ds_file, targets=value_allruns, chunks=chunks_allruns, mask=mask_name)
    
    #pick trials in conditions we want
    num_trials = len(trials_allruns)
    cond_by_trial = trials_allruns[:,3]
    inds_in_conds = [c for c in range(num_trials) if cond_by_trial[c] in cond_nums]
    
    fds_subset = fds[inds_in_conds,:]
    
    return fds_subset
    
    
    
    
        
    
    