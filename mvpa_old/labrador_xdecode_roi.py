#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 25 10:44:50 2022

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
from mvpa2.base.hdf5 import h5load, h5save
from mvpa2.misc.neighborhood import Sphere
import mvpa_utils_lab
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.base import mask_mapper
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model
from sklearn.model_selection import GroupKFold
import random
import time
import numpy as np
import scipy
import pandas as pd

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS
start_time = time.time()

bundle_path = '/state/partition1/home/lcross/Bundle_Value/'

subj = str(sys.argv[1])

analysis_name = 'cross_decoding_rel_value'

relative_value = True

save = True

subsample = False

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

#which ds to use and which mask to use
#glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
#glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D_pfc_mask.nii.gz'
mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#mask_name = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

#make targets with mvpa utils
if relative_value:
    fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, mask_name, conditions, relative_value=True, system='labrador')
else:
    fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, mask_name, conditions, relative_value=False, system='labrador')

#zscore targets
if not relative_value:
    fds.targets = scipy.stats.zscore(fds.targets)
    
trial_categ = fds.sa.trial_categ
sitem_inds = np.where(trial_categ < 3)[0]
bundle_inds = np.where(trial_categ > 2)[0]
#if subsample, take random bundle inds to make the same number of bundle and single item inds
if subsample:
    num_sitem_trials = len(sitem_inds)
    bundle_inds = np.array(random.sample(list(bundle_inds), num_sitem_trials))
    
#define model
alp=3
sk_ridge = linear_model.Ridge(alpha=10**alp)
#sk_ridge = PLSRegression(n_components=50)
r_scorer = make_scorer(get_correlation)
run_num = 15
gkf = GroupKFold(n_splits=run_num)

mask_count = 0
subj_data = []
for mask in mask_loop: 
    mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
    #brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    masked = fmri_dataset(mask_name, mask=brain_mask)
    reshape_masked=masked.samples.reshape(fds.shape[1])
    reshape_masked=reshape_masked.astype(bool)
    mask_map = mask_mapper(mask=reshape_masked)
    
    mask_slice = mask_map[1].slicearg
    mask_inds = np.where(mask_slice == 1)[0]
    
    fds_mask = fds[:,mask_inds]
    
    X = fds_mask.samples
    y = fds_mask.targets
    cv_groups = fds_mask.chunks
    
    cv_score_s2s = np.zeros([run_num])
    cv_score_s2b = np.zeros([run_num])
    cv_score_b2b = np.zeros([run_num])
    cv_score_b2s = np.zeros([run_num])
    pred_bundle_vals = np.array([])
    real_bundle_vals = np.array([])
    cv_count = -1
    for train, test in gkf.split(X, y, groups=cv_groups):
        #train within category test within and across category
        train_s = np.intersect1d(train, sitem_inds)
        test_s = np.intersect1d(test, sitem_inds)
        train_b = np.intersect1d(train, bundle_inds)
        test_b = np.intersect1d(test, bundle_inds)
        
        #train on single item
        sk_ridge.fit(X[train_s,:],y[train_s])
        y_preds_in = sk_ridge.predict(X[test_s,:])
        cv_score_s2s[cv_count] = get_correlation(y_preds_in,y[test_s])
        y_preds_out = sk_ridge.predict(X[test_b,:])
        cv_score_s2b[cv_count] = get_correlation(y_preds_out,y[test_b])
        pred_bundle_vals = np.append(pred_bundle_vals, y_preds_out)
        real_bundle_vals = np.append(real_bundle_vals, y[test_b])
        
        #train on bundle
        sk_ridge.fit(X[train_b,:],y[train_b])
        y_preds_in = sk_ridge.predict(X[test_b,:])
        cv_score_b2b[cv_count] = get_correlation(y_preds_in,y[test_b])
        y_preds_out = sk_ridge.predict(X[test_s,:])
        cv_score_b2s[cv_count] = get_correlation(y_preds_out,y[test_s])
        
        cv_count+=1
        
    mask_label = mask_names[mask_count]
    
    mask_scores = {'Subj': [subj for i in range(run_num)], 'Mask': [mask_label for i in range(run_num)], 
                                'S2S': cv_score_s2s, 'S2B': cv_score_s2b, 'B2B': cv_score_b2b, 'B2S': cv_score_b2s}
    #mask_scores = [subj, mask_label, np.mean(cv_score_s2s), np.mean(cv_score_s2b), np.mean(cv_score_b2b), np.mean(cv_score_b2s)]
    subj_data.append(pd.DataFrame(mask_scores))
    
    mask_count += 1
    
#subj_df = pd.DataFrame(subj_data, columns = ['Subj', 'Mask','S2S','S2B','B2B','B2S']) 
subj_df = pd.concat(subj_data, ignore_index=True)
#save
if save:
    save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
    subj_df.to_csv(save_path+'xdecode_rois_2023.csv')