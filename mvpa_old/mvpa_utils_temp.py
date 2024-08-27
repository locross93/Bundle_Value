#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:35:36 2019

@author: logancross
"""
from mvpa2.suite import *
import matplotlib.pyplot as plt
from os import listdir
import time

def make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=False, tolman=False):
    
    start_time = time.time()
    print 'Starting making targets',time.time() - start_time
    
    #create dict conditions name to number conversion
    cond_dict = {
		'Food item' : 1,
		'Trinket item' : 2,
		'Food bundle' : 3,
		'Trinket bundle' : 4,
		'Mixed bundle' : 5
	}
    
    cond_nums = [cond_dict[condition] for condition in conditions]
    
    if tolman:
        onsets_folder = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    else:
        onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    
    trial_list = []
    trial_categ_list = []
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
        
        #get timing for all conditions and sort by this timing
        timing = np.concatenate((cond001_onsets[:,0], cond002_onsets[:,0], cond003_onsets[:,0], cond004_onsets[:,0], cond005_onsets[:,0]))
        #add a list of trial category as a sample attribute
        trial_categ_unsort = np.concatenate((1*np.ones(len(cond001_onsets)), 2*np.ones(len(cond002_onsets)), 3*np.ones(len(cond003_onsets)), 4*np.ones(len(cond004_onsets)), 5*np.ones(len(cond005_onsets))))
        #sort by trial timing and append to lists
        sort_time_inds = np.argsort(timing)
        all_trials = np.concatenate((cond001_onsets, cond002_onsets, cond003_onsets, cond004_onsets, cond005_onsets))
        all_trials = all_trials[sort_time_inds,:]
        trial_list.append(all_trials)
        trial_categ = trial_categ_unsort[sort_time_inds]
        trial_categ_list.append(trial_categ)
        chunks = run_num*np.ones([len(all_trials)])
        chunks_list.append(chunks)
        run_num+=1
        
    trials_allruns = np.asarray([item for sublist in trial_list for item in sublist])
    trial_categ_allruns = np.asarray([item for sublist in trial_categ_list for item in sublist])
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
    
    #add trial category as a sample attribute
    fds_subset.sa.trial_categ = trial_categ_allruns[inds_in_conds]
    
    print 'Finished making targets',time.time() - start_time
    
    return fds_subset

def get_correlation_pval(r, df):
    from scipy.stats.stats import _betai
    
    t_squared = r*r * (df / ((1.0 - r) * (1.0 + r)))
    prob = _betai(0.5*df, 0.5, df / (df + t_squared))
    
    return prob


def get_fdr_r_threshold(r_array, df, alpha):
    from statsmodels.stats.multitest import multipletests
    
    pvals = np.array([get_correlation_pval(r, df) for r in r_array])
    fdr = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_inds = np.where(fdr[0])[0]
    threshold_ind = np.where(pvals == np.max(pvals[sig_inds]))[0]
    pearsonr_thr = r_array[threshold_ind]
    
    return pearsonr_thr

def get_fdr_r_threshold_onesided(r_array, df, alpha, pos_only=True):
    from statsmodels.stats.multitest import multipletests
    
    if pos_only:
        neg_inds = np.where(r_array < 0)[0]
        r_array[neg_inds] = 0
    
    pvals = np.array([get_correlation_pval(r, df) for r in r_array])
    if pos_only:
        #turned two tailed pvalue into one tailed
        pvals = pvals/2
    fdr = multipletests(pvals, alpha=alpha, method='fdr_bh')
    sig_inds = np.where(fdr[0])[0]
    threshold_ind = np.where(pvals == np.max(pvals[sig_inds]))[0]
    pearsonr_thr = r_array[threshold_ind]
    
    return pearsonr_thr

def get_uncorrected_threshold(r_array, df, alpha):
    
    pvals = np.array([get_correlation_pval(r, df) for r in r_array])
    sig_inds = np.where(pvals < alpha)[0]
    threshold_ind = np.where(pvals == np.max(pvals[sig_inds]))[0]
    pearsonr_thr = r_array[threshold_ind]
    
    return pearsonr_thr
    
def conjunction_and_diff_images(scores1, scores2, thr1, thr2):  
    
    assert len(scores1) == len(scores2)
    
    #conjunction
    sig_scores1 = np.where(scores1 > thr1)[0]
    sig_scores2 = np.where(scores2 > thr2)[0]
    conjunct_inds = np.intersect1d(sig_scores1, sig_scores2)
    scores_conjunct = np.zeros(len(scores1))
    scores_conjunct[conjunct_inds] = 1
    
    #unique maps
    diff_inds1 = np.setdiff1d(sig_scores1, conjunct_inds)
    diff_inds2 = np.setdiff1d(sig_scores2, conjunct_inds)
    scores_diff1 = np.zeros(len(scores1))
    scores_diff1[diff_inds1] = 1
    scores_diff2 = np.zeros(len(scores2))
    scores_diff2[diff_inds2] = 1
    
    return scores_conjunct, scores_diff1, scores_diff2

def plot_mtx(mtx, labels, title):
    # little helper function to plot dissimilarity matrices
    # if using correlation-distance, we use colorbar range of [0,2]
    pl.figure()
    pl.imshow(mtx, interpolation='nearest')
    pl.xticks(range(len(mtx)), labels, rotation=-45)
    pl.yticks(range(len(mtx)), labels)
    pl.title(title)
    pl.clim((0, 2))
    pl.colorbar()
    
def mask_dset(ds, mask):
    '''
    Returns masked dataset

    ds: pymvpa dataset
    mask: binary [0,1] mask file in nii format
    
    *currently temporarily reverts chain mapper to 2 mappers used to load fmri_dataset
    '''

    ds.a.mapper = ds.a.mapper[:2]
    mask = datasets.mri._load_anyimg(mask)[0]
    flatmask = ds.a.mapper.forward1(mask)
    return ds[:, flatmask != 0]
    
#    
#  
#subj = 101
#scores_foodval = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_food_value')
#scores_trinketval = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_trinket_value')
#scores_conjunct, scores_diff1, scores_diff2 = conjunction_and_diff_images(scores_foodval, scores_trinketval, 0.1)
#nimg_conjunct = map2nifti(fds, scores_conjunct)
#nimg_conjunct.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_food_trinket_conjunct_thr10.nii.gz')
#nimg_item = map2nifti(fds, scores_diff1)
#nimg_item.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_food_unique_thr10.nii.gz')
#nimg_bundle = map2nifti(fds, scores_diff2)
#nimg_bundle.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_trinket_unique_thr10.nii.gz')
#    
#
#z_coord = sub103_vox_inds[:,2]
#sort_z = np.sort(z_coord)
#arg_sort_z = np.argsort(z_coord)
#sort_z_itemval = scores_itemval[arg_sort_z]
#sort_z_bundleval = scores_bundleval[arg_sort_z]
#
#print scipy.stats.pearsonr(sort_z[:140000], sort_z_itemval[:140000])
#print scipy.stats.pearsonr(sort_z[:140000], sort_z_bundleval[:140000])
#
#plt.plot(sort_z,sort_z_itemval,'ro-', label="Item Value R")
#plt.plot(sort_z,sort_z_bundleval,'bo-', label="Bundle Value R")
#plt.xlabel('Z Coordinate')
#plt.ylabel('R')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
  
#scores_itemval = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_item_value')
#scores_bundleval = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_bundle_value') 
#thr1 = get_fdr_r_threshold(scores_itemval, 298, 0.005)
#thr2 = get_fdr_r_threshold(scores_bundleval, 598, 0.005)
#scores_conjunct, scores_diff1, scores_diff2 = conjunction_and_diff_images(scores_itemval, scores_bundleval, thr1, thr2)
#nimg_conjunct = map2nifti(fds, scores_conjunct)
#nimg_conjunct.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_item_bundle_conjunct_thrfdr005.nii.gz')
#nimg_item = map2nifti(fds, scores_diff1)
#nimg_item.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_item_unique_thrfdr005.nii.gz')
#nimg_bundle = map2nifti(fds, scores_diff2)
#nimg_bundle.to_filename('/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_bundle_unique_thrfdr005.nii.gz')

    
    