#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 13:00:41 2021

@author: logancross
"""

#prevent multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, '/state/partition1/home/lcross/Bundle_Value/mvpa')
os.chdir('/state/partition1/home/lcross/Bundle_Value/mvpa')

#from mvpa2.suite import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
import mvpa_utils_lab
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.base import mask_mapper
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import random
import time

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS
start_time = time.time()

bundle_path = '/state/partition1/home/lcross/Bundle_Value/'

subj = str(sys.argv[1])

analysis_name = 'xdecoding_abs_vs_rel_value'

relative_value = False

save = True

subsample = False

num_perms = 1000

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value, system='labrador')

trial_categ = fds.sa.trial_categ
sitem_inds = np.where(trial_categ < 3)[0]
bundle_inds = np.where(trial_categ > 2)[0]
#if subsample, take random bundle inds to make the same number of bundle and single item inds
if subsample:
    num_sitem_trials = len(sitem_inds)
    bundle_inds = np.array(random.sample(bundle_inds, num_sitem_trials))

#define model
alp=3
sk_ridge = linear_model.Ridge(alpha=10*alp)
r_scorer = make_scorer(get_correlation)
run_num = 15
gkf = GroupKFold(n_splits=run_num)

perm_score_dict = {}
mask_count = 0
for mask in mask_loop: 
    mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    masked = fmri_dataset(mask_name, mask=brain_mask)
    reshape_masked=masked.samples.reshape(fds.shape[1])
    reshape_masked=reshape_masked.astype(bool)
    mask_map = mask_mapper(mask=reshape_masked)
    
    mask_slice = mask_map[1].slicearg
    mask_inds = np.where(mask_slice == 1)[0]
    
    fds_mask = fds[:,mask_inds]
    
    X = fds_mask.samples
    
    abs_value = fds_mask.targets
    rel_value = np.zeros([len(abs_value)])
    zitem_values = scipy.stats.zscore(abs_value[sitem_inds])
    rel_value[sitem_inds] = zitem_values
    zbundle_values = scipy.stats.zscore(abs_value[bundle_inds])
    rel_value[bundle_inds] = zbundle_values
    cv_groups = fds_mask.chunks
    
    cv_score_s2abs = np.zeros([num_perms])
    cv_score_s2rel = np.zeros([num_perms])
    cv_score_b2abs = np.zeros([num_perms])
    cv_score_b2rel = np.zeros([num_perms])
    for p in range(num_perms):
        if p%100 == 0:
            current_time = time.time()
            time_dif_s = current_time - start_time
            print(mask,p,time_dif_s)
        cv_score_s2abs_temp = np.zeros([run_num])
        cv_score_s2rel_temp = np.zeros([run_num])
        cv_score_b2abs_temp = np.zeros([run_num])
        cv_score_b2rel_temp = np.zeros([run_num])
        cv_count = -1
        for train, test in gkf.split(X, abs_value, groups=cv_groups):
            #train within category test within and across category
            train_s = np.intersect1d(train, sitem_inds)
            test_s = np.intersect1d(test, sitem_inds)
            train_b = np.intersect1d(train, bundle_inds)
            test_b = np.intersect1d(test, bundle_inds)
            
            #train on single item
            sk_ridge.fit(X[train_s,:],abs_value[train_s])
            y_preds_test = sk_ridge.predict(X[test,:])
            #scramble y_preds_test
            inds2shuffle = np.arange(len(y_preds_test))
            sfl_inds = random.sample(inds2shuffle.tolist(), len(y_preds_test))
            y_preds_test = y_preds_test[sfl_inds]
            cv_score_s2abs_temp[cv_count] = get_correlation(y_preds_test,abs_value[test])
            cv_score_s2rel_temp[cv_count] = get_correlation(y_preds_test,rel_value[test])
            
            #train on bundle
            sk_ridge.fit(X[train_b,:],abs_value[train_b])
            y_preds_test = sk_ridge.predict(X[test,:])
            #scramble y_preds_test
            inds2shuffle = np.arange(len(y_preds_test))
            sfl_inds = random.sample(inds2shuffle.tolist(), len(y_preds_test))
            y_preds_test = y_preds_test[sfl_inds]
            cv_score_b2abs_temp[cv_count] = get_correlation(y_preds_test,abs_value[test])
            cv_score_b2rel_temp[cv_count] = get_correlation(y_preds_test,rel_value[test])
            
            cv_count+=1
        cv_score_s2abs[p] = np.mean(cv_score_s2abs_temp)
        cv_score_s2rel[p] = np.mean(cv_score_s2rel_temp)
        cv_score_b2abs[p] = np.mean(cv_score_b2abs_temp)
        cv_score_b2rel[p] = np.mean(cv_score_b2rel_temp)
        
    perm_score_dict[mask] = [cv_score_s2abs, cv_score_s2rel, cv_score_b2abs, cv_score_b2rel]
    
#save
if save:
    save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    np.save(save_path+'/perms_xdecode_abs_rel_scores.npy', perm_score_dict)