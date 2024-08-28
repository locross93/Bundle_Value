#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 16:35:36 2019

@author: logancross
"""
from mvpa2.suite import *
from mvpa2.datasets.base import Dataset
from mvpa2.measures.base import Measure
from scipy.stats import rankdata, pearsonr
from scipy.spatial.distance import squareform
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from os import listdir
import time

bundle_path = '/Users/locro/Documents/Bundle_Value/'

#def make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=False, tolman=False):
def make_targets(subj, glm_ds_file, mask_name, conditions, relative_value, system='mac'):
    
    start_time = time.time()
    #print 'Starting making targets',time.time() - start_time
    
    #create dict conditions name to number conversion
    cond_dict = {
		'Food item' : 1,
		'Trinket item' : 2,
		'Food bundle' : 3,
		'Trinket bundle' : 4,
		'Mixed bundle' : 5
	}
    
    cond_nums = [cond_dict[condition] for condition in conditions]
    
#    if system == 'tolman':
#        onsets_folder = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
#    elif system == 'labrador':
#        onsets_folder = '/state/partition1/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
#    elif system == 'mac':
#        onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    onsets_folder = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    
    trial_list = []
    trial_categ_list = []
    chunks_list = []
    run_num = 0
    #changed LC 4/25/19 tolman does this out of order
    #for run in dir_onsets:
    for run in range(1,len(dir_onsets)+1):
        #temp_folder = onsets_folder+run
        if run < 10:
            temp_folder = onsets_folder+'task001_run00'+str(run)
        else:
            temp_folder = onsets_folder+'task001_run0'+str(run)
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
    rts_allruns = trials_allruns[:,1]
    
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
    
    #add reaction time as a sample attribute
    fds_subset.sa.rt = rts_allruns[inds_in_conds]
    
    #print 'Finished making targets',time.time() - start_time
    
    return fds_subset

def make_targets_choice(subj, glm_ds_file, mask_name, conditions, system='mac'):
    
    start_time = time.time()
    #print 'Starting making targets',time.time() - start_time
    
    #create dict conditions name to number conversion
    cond_dict = {
		'Food item' : 1,
		'Trinket item' : 2,
		'Food bundle' : 3,
		'Trinket bundle' : 4,
		'Mixed bundle' : 5
	}
    
    cond_nums = [cond_dict[condition] for condition in conditions]
    
    if system == 'tolman':
        onsets_folder = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model004/onsets/'
    elif system == 'labrador':
        onsets_folder = '/state/partition1/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model004/onsets/'
    elif system == 'mac':
        onsets_folder = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model004/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    trial_list = []
    chunks_list = []
    run_num = 0
    for run in range(1,len(dir_onsets)+1):
        #temp_folder = onsets_folder+run
        if run < 10:
            temp_folder = onsets_folder+'task001_run00'+str(run)
        else:
            temp_folder = onsets_folder+'task001_run0'+str(run)
        choice_mat = np.genfromtxt(temp_folder+'/cond001.txt')
        trial_list.append(choice_mat)
        chunks = run_num*np.ones([len(choice_mat)])
        chunks_list.append(chunks)
        run_num+=1
    trials_allruns = np.vstack(trial_list)
    chunks_allruns = np.asarray([item for sublist in chunks_list for item in sublist]).astype(int) 
    choice_allruns = trials_allruns[:,2]
    
    #load fmri dataset with these values as targets
    fds = fmri_dataset(samples=glm_ds_file, targets=choice_allruns, chunks=chunks_allruns, mask=mask_name)
    
    #pick trials in conditions we want
    num_trials = len(trials_allruns)
    cond_by_trial = trials_allruns[:,3]
    inds_in_conds = [c for c in range(num_trials) if cond_by_trial[c] in cond_nums]
    
    fds_subset = fds[inds_in_conds,:]
    
    #add trial category as a sample attribute
    fds_subset.sa.trial_categ = trials_allruns[inds_in_conds,3]
    
    return fds_subset

def get_all_values(subj):
    onsets_folder = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model002/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    
    trial_list = []
    trial_categ_list = []
    run_num = 0
    #changed LC 4/25/19 tolman does this out of order
    #for run in dir_onsets:
    for run in range(1,len(dir_onsets)+1):
        #temp_folder = onsets_folder+run
        if run < 10:
            temp_folder = onsets_folder+'task001_run00'+str(run)
        else:
            temp_folder = onsets_folder+'task001_run0'+str(run)
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
        run_num+=1
    trials_allruns = np.asarray([item for sublist in trial_list for item in sublist])
    value_allruns = trials_allruns[:,2]
    cond_by_trial = trials_allruns[:,3]
    
    return value_allruns, cond_by_trial
        
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

def plot_mtx(mtx, fig_max=None, skip=5, labels=None, title=''):
    #if matrix not square make it square
    if len(mtx.shape) == 1 or mtx.shape[1] == 1:
        mtx = squareform(mtx)
    
    if labels is None:
        labels = np.arange(len(mtx))
        
    if fig_max==None:
        max_dist = np.max(mtx)
    else:
        max_dist = fig_max
        
    pl.figure(figsize=(8,8))
    pl.imshow(mtx, interpolation='nearest', cmap='jet')
    pl.xticks(range(len(mtx))[::skip], labels[::skip], rotation=90)
    pl.yticks(range(len(mtx))[::skip], labels[::skip])
    pl.title(title)
    pl.clim((0, max_dist))
    pl.colorbar()
    plt.show()
    
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
    
def get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked, tolman=False):
    
    target_dsms = {}
    
    #control dsm for day
    num_trials = fds.shape[0]    
    day_array = np.zeros(num_trials)
    run_array = fds.chunks
    day2_inds = np.intersect1d(np.where(run_array > 4)[0],np.where(run_array < 10)[0])
    day_array[day2_inds] = 1
    day3_inds = np.where(run_array >= 10)[0]
    day_array[day3_inds] = 2
    ds_day = dataset_wizard(day_array, targets=np.zeros(num_trials))
    
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_day = dsm(ds_day)
    if ranked:
        res_day = rankdata(res_day)
    else:
        res_day = res_day.samples.reshape(-1)
    target_dsms['day'] = res_day
    
    #control dsm for run
    ds_run = dataset_wizard(run_array, targets=np.zeros(num_trials))
    
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_run = dsm(ds_run)
    if ranked:
        res_run = rankdata(res_run)
    else:
        res_run = res_run.samples.reshape(-1)
    target_dsms['run'] = res_run
    
    #stimulus identity
    item_list = np.genfromtxt(bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
    
    #if only individual item trials, only include these trials
    if conditions == ['Food item', 'Trinket item']:
        inds_in_conds = np.where(item_list[:,1] == -1)[0]
        item_list = item_list[inds_in_conds, :]
    elif conditions == ['Food bundle','Trinket bundle','Mixed bundle']:
        inds_in_conds = np.where(item_list[:,1] != -1)[0]
        item_list = item_list[inds_in_conds, :]
    
    num_items = len(item_list)
    square_rdm = np.ones([num_items, num_items])
    
    for i in range(num_items):
        for j in range(num_items):
            items_row = item_list[i,:]
            items_col = item_list[j,:]
            if items_row[0] in items_col:
                square_rdm[i,j] = 0
            elif items_row[1] > 0 and items_row[1] in items_col:
                square_rdm[i,j] = 0
    
    if square_dsm_bool:
        res_stim_id = square_rdm
    else:
        res_stim_id = squareform(square_rdm)
    if ranked:
        res_stim_id = rankdata(res_stim_id)
    target_dsms['stim_id'] = res_stim_id
        
    #value
    value = fds.targets
    #value_norm = zscore(value)
    scaler = MinMaxScaler()
    #scaler = StandardScaler()
    value_norm = scaler.fit_transform(value.reshape(-1,1)).reshape(-1)
    ds_value = dataset_wizard(value_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_value = dsm(ds_value)
    if ranked:
        res_value = rankdata(res_value)
    else:
        res_value = res_value.samples.reshape(-1)
    target_dsms['value'] = res_value
    
    trial_categ = fds.sa.trial_categ
    #sometimes fds.sa.trial_categ acts weird
    assert(np.unique(trial_categ).shape[0] == len(conditions))
    #food value
    food_inds = np.where(trial_categ == 1)[0]
    food_value = value[food_inds]
    #make the food value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
    mean_value = np.mean(food_value)
    food_value_norm = mean_value*np.ones(num_trials)
    food_value_norm[food_inds] = food_value
    food_value_norm = scaler.fit_transform(food_value_norm.reshape(-1,1))
    ds_fvalue = dataset_wizard(food_value_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_fvalue = dsm(ds_fvalue)
    if ranked:
        res_fvalue = rankdata(res_fvalue)
    else:
        res_fvalue = res_fvalue.samples.reshape(-1)
    target_dsms['fvalue'] = res_fvalue
    
    #trinket value
    trinket_inds = np.where(trial_categ == 2)[0]
    trinket_value = value[trinket_inds]
    #make the trinket value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
    mean_value = np.mean(trinket_value)
    trinket_value_norm = mean_value*np.ones(num_trials)
    trinket_value_norm[trinket_inds] = trinket_value
    trinket_value_norm = scaler.fit_transform(trinket_value_norm.reshape(-1,1))
    ds_tvalue = dataset_wizard(trinket_value_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_tvalue = dsm(ds_tvalue)
    if ranked:
        res_tvalue = rankdata(res_tvalue)
    else:
        res_tvalue = res_tvalue.samples.reshape(-1)
    target_dsms['tvalue'] = res_tvalue
    
    #single item trials value
    ind_item_inds = np.where(trial_categ < 3)[0]
    ind_item_value = value[ind_item_inds]
    #make the ind item value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
    mean_value = np.mean(ind_item_value)
    mean_value = np.median(ind_item_value)
    ind_item_value_norm = mean_value*np.ones(num_trials)
    ind_item_value_norm[ind_item_inds] = ind_item_value
    ind_item_value_norm = scaler.fit_transform(ind_item_value_norm.reshape(-1,1))
    ds_ivalue = dataset_wizard(ind_item_value_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_ivalue = dsm(ds_ivalue)
    if ranked:
        res_ivalue = rankdata(res_ivalue)
    else:
        res_ivalue = res_ivalue.samples.reshape(-1)
    target_dsms['ivalue'] = res_ivalue
    
    #bundle value
    bundle_inds = np.where(trial_categ > 2)[0]
    bundle_value = value[bundle_inds]
    #make the bundle value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
    mean_value = np.mean(bundle_value)
    mean_value = np.median(bundle_value)
    bundle_value_norm = mean_value*np.ones(num_trials)
    bundle_value_norm[bundle_inds] = bundle_value
    bundle_value_norm = scaler.fit_transform(bundle_value_norm.reshape(-1,1))
    ds_bvalue = dataset_wizard(bundle_value_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_bvalue = dsm(ds_bvalue)
    if ranked:
        res_bvalue = rankdata(res_bvalue)
    else:
        res_bvalue = res_bvalue.samples.reshape(-1)
    target_dsms['bvalue'] = res_bvalue
    
    #binned value
    num_bins = 4
    step = 1.0/num_bins
    percentiles = np.arange(0,1+step,step)
    bin_edges = scipy.stats.mstats.mquantiles(value, percentiles)
    #add 1 to last bin edge to make it inclusive
    bin_edges[-1] = bin_edges[-1] + 1
    value_binned = np.digitize(value, bin_edges)
    ds_value_bin = dataset_wizard(value_binned, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_value_bin = dsm(ds_value_bin)
    if ranked:
        res_value_bin = rankdata(res_value_bin)
    else:
        res_value_bin = res_value_bin.samples.reshape(-1)
    target_dsms['value_bin'] = res_value_bin
    
    #choice difficulty
    choice_diff = np.zeros(num_trials)
    median_val_ind_item = np.median(ind_item_value)
    choice_diff[ind_item_inds] = np.absolute(value[ind_item_inds] - median_val_ind_item)
    median_val_bundle = np.median(bundle_value)
    choice_diff[bundle_inds] = np.absolute(value[bundle_inds] - median_val_bundle)
    ds_choice_diff = dataset_wizard(choice_diff, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_cdiff = dsm(ds_choice_diff)
    if ranked:
        res_cdiff = rankdata(res_cdiff)
    else:
        res_cdiff = res_cdiff.samples.reshape(-1)
    target_dsms['choice_diff'] = res_cdiff
    
    #choice 
    choice = get_fmri_choices(subj, conditions)
    ds_choice = dataset_wizard(choice, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_choice = dsm(ds_choice)
    if ranked:
        res_choice = rankdata(res_choice)
    else:
        res_choice = res_choice.samples.reshape(-1)
    target_dsms['choice'] = res_choice
    
    #left vs right choice
    lr_choice_list = np.genfromtxt(bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/lr_choice.txt')
    lr_choice = lr_choice_list[:,1]
    #if only individual item trials, only include these trials
    if len(conditions) < 5:
        lr_choice = lr_choice[inds_in_conds]
    ds_lr_choice = dataset_wizard(lr_choice, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_lr_choice = dsm(ds_lr_choice)
    if ranked:
        res_lr_choice = rankdata(res_lr_choice)
    else:
        res_lr_choice = res_lr_choice.samples.reshape(-1)
    target_dsms['lr_choice'] = res_lr_choice
    
    #item or bundle?
    if len(conditions) == 5:
        item_or_bundle = fds.sa.trial_categ
        item_inds = np.where(item_or_bundle < 3)[0]
        bundle_inds = np.where(item_or_bundle > 2)[0]
        item_or_bundle[item_inds] = 0
        item_or_bundle[bundle_inds] = 1
        assert np.max(item_or_bundle) == 1
        
        ds_trial_cat = dataset_wizard(item_or_bundle, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_trial_cat = dsm(ds_trial_cat)
        if ranked:
            res_trial_cat = rankdata(res_trial_cat)
        else:
            res_trial_cat = res_trial_cat.samples.reshape(-1)
        target_dsms['item_or_bundle'] = res_trial_cat
        
    #reaction time
    rt = fds.sa.rt
    rt_norm = scaler.fit_transform(rt.reshape(-1,1)).reshape(-1)
    ds_rt = dataset_wizard(rt_norm, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_rt = dsm(ds_rt)
    if ranked:
        res_rt = rankdata(res_rt)
    else:
        res_rt = res_rt.samples.reshape(-1)
    target_dsms['rt'] = res_rt
    
    return target_dsms

def get_fmri_choices(subj, conditions):
    #create dict conditions name to number conversion
    cond_dict = {
		'Food item' : 1,
		'Trinket item' : 2,
		'Food bundle' : 3,
		'Trinket bundle' : 4,
		'Mixed bundle' : 5
	}
    
    cond_nums = [cond_dict[condition] for condition in conditions]
    
    onsets_folder = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/model004/onsets/'
    
    dir_onsets = listdir(onsets_folder)
    if dir_onsets[0] == '.DS_Store':
        dir_onsets.remove('.DS_Store')
    trial_list = []
    for run in range(1,len(dir_onsets)+1):
        #temp_folder = onsets_folder+run
        if run < 10:
            temp_folder = onsets_folder+'task001_run00'+str(run)
        else:
            temp_folder = onsets_folder+'task001_run0'+str(run)
        choice_mat = np.genfromtxt(temp_folder+'/cond001.txt')
        trial_list.append(choice_mat)
    trials_allruns = np.vstack(trial_list)
    choice_allruns = trials_allruns[:,2]
    
    #pick trials in conditions we want
    num_trials = len(trials_allruns)
    cond_by_trial = trials_allruns[:,3]
    inds_in_conds = [c for c in range(num_trials) if cond_by_trial[c] in cond_nums]
    
    choice = choice_allruns[inds_in_conds]
    
    return choice

class rsa_custom(Measure):
    
    is_trained = True
    """Indicate that this measure is always trained."""
    
    def __init__(self, target_dsm, num_trials, chunks, square_dsm_bool, remove_within_day, pairwise_metric='correlation', 
                 comparison_metric='pearson', **kwargs):
        
        # init base classes first
        Measure.__init__(self, **kwargs)
        if comparison_metric not in ['spearman','pearson']:
            raise Exception("comparison_metric %s is not in "
                            "['spearman','pearson']" % comparison_metric)
        self.target_dsm = target_dsm
        if comparison_metric == 'spearman':
            self.target_dsm = rankdata(target_dsm)
        self.square_dsm_bool = square_dsm_bool
        self.remove_within_day = remove_within_day
        self.pairwise_metric = pairwise_metric
        self.comparison_metric = comparison_metric
        
        if remove_within_day:
            #if remove within run correlations, remove these inds using matching day dsm
            day_array = np.zeros(num_trials)
            run_array = chunks
            day2_inds = np.intersect1d(np.where(run_array > 4)[0],np.where(run_array < 10)[0])
            day_array[day2_inds] = 1
            day3_inds = np.where(run_array >= 10)[0]
            day_array[day3_inds] = 2
            ds_day = dataset_wizard(day_array, targets=np.zeros(num_trials))
            dsm = PDist(pairwise_metric='matching', square=self.square_dsm_bool)
            res_day = dsm(ds_day)
            self.btwn_run_inds = np.where(res_day.samples == 1)[0]
        
        
    def _call(self,dataset):
        
        res_fmri = pdist(dataset,self.pairwise_metric)
        if self.square_dsm_bool:
            res_fmri = squareform(res_fmri)
        if self.comparison_metric == 'spearman':
            res_fmri = rankdata(res_fmri)
            
        if self.remove_within_day:            
            if self.comparison_metric == 'spearman':
                res = pearsonr(res_fmri[self.btwn_run_inds], rankdata(self.target_dsm)[self.btwn_run_inds])[0]
            elif self.comparison_metric == 'pearson':
                res = pearsonr(res_fmri[self.btwn_run_inds], self.target_dsm[self.btwn_run_inds])[0]
        else:
            if self.comparison_metric == 'spearman':
                res = pearsonr(res_fmri, rankdata(self.target_dsm))[0]
            elif self.comparison_metric == 'pearson':
                res = pearsonr(res_fmri, self.target_dsm)[0]
        
        return Dataset(np.array([res,]))
    
def roiClass_1Ss(ds, roi_mask_nii_path, clf = LinearCSVMC(), part = NFoldPartitioner(), sl=True):
    '''
    From pymvpaw https://github.com/rystoli/PyMVPAw/blob/master/pymvpaw/roi_wraps.py
    Executes classification on ROI with target_dm
    ---
    ds: pymvpa dataset
    roi_mask_nii_path: path to nifti of roi mask
    clf: specify classifier
    part: specify partitioner
    ---
    Return: Classification accuracy subracting chance level given number of targets
    '''
    if not sl:
        data_m = mask_dset(ds, roi_mask_nii_path)
        print('Dataset masked to shape: %s' % (str(data_m.shape)))
    else:
        data_m = ds
 
    #data prep
    remapper = data_m.copy()
    inv_mask = data_m.samples.std(axis=0)>0
    sfs = StaticFeatureSelection(slicearg=inv_mask)
    sfs.train(remapper)
    data_m = remove_invariant_features(data_m)

    print('Beginning roiClass analysis w/ targets %s...' % (data_m.UT))
    cv = CrossValidation(clf, part, enable_ca=['stats'], errorfx=lambda p, t: np.mean(p == t))
    res = cv(data_m)
    return np.mean(res.samples)

    
    