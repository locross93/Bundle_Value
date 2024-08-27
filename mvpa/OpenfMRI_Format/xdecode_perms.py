#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 14:32:04 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import random
from scipy.stats import percentileofscore
from statsmodels.stats.multitest import multipletests
from numpy.random import permutation
import pandas as pd

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

def get_voxel_sphere(center_coords, voxel_indices):
    radius = 4
    sphere = Sphere(radius)
    all_coords = sphere(center_coords)
    inds2use = []
    for coords in all_coords:
        coords = np.array(coords)
        temp_ind = np.where((voxel_indices == coords).all(axis=1))[0]
        if len(temp_ind) > 0:
            assert len(temp_ind) == 1
            inds2use.append(temp_ind[0])
            
    return inds2use

def get_pval(score_array, value):
    pval = (100-percentileofscore(score_array, value))/100
    
    return pval

###SCRIPT ARGUMENTS

start_time = time.time()

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj = '105'

analysis_name = 'cross_decoding_rel_value'

relative_value = True

save = True

subsample = True

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

#which ds to use and which mask to use
glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value)
    
#zscore targets
if not relative_value:
    fds.targets = scipy.stats.zscore(fds.targets)
    
trial_categ = fds.sa.trial_categ
sitem_inds = np.where(trial_categ < 3)[0]
bundle_inds = np.where(trial_categ > 2)[0]
#if subsample, take random bundle inds to make the same number of bundle and single item inds
if subsample:
    num_sitem_trials = len(sitem_inds)
    bundle_inds = np.array(random.sample(bundle_inds, num_sitem_trials))

#define model
alp=4
sk_ridge = linear_model.Ridge(alpha=10*alp)
r_scorer = make_scorer(get_correlation)
run_num = 15
gkf = GroupKFold(n_splits=run_num)
num_vox = fds.shape[1]

y = fds.targets
cv_groups = fds.chunks
voxel_indices = fds.fa.voxel_indices
num_voxs = fds.shape[1]

#load xdecode scores to get voxels with signal
s2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
s2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b'
b2b_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b'
b2s_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s'

scores_s2s = h5load(s2s_file)
scores_s2b = h5load(s2b_file)
scores_b2b = h5load(b2b_file)
scores_b2s = h5load(b2s_file)

#take 25 voxels in 90th+ percentile from each file
sig_thr_temp = np.percentile(scores_s2s, 90)
s2s_voxs = np.random.choice(np.where(scores_s2s > sig_thr_temp)[0], 25)
sig_thr_temp = np.percentile(scores_s2b, 90)
s2b_voxs = np.random.choice(np.where(scores_s2b > sig_thr_temp)[0], 25)
sig_thr_temp = np.percentile(scores_b2b, 90)
b2b_voxs = np.random.choice(np.where(scores_b2b > sig_thr_temp)[0], 25)
sig_thr_temp = np.percentile(scores_b2s, 90)
b2s_voxs = np.random.choice(np.where(scores_b2s > sig_thr_temp)[0], 25)
voxs2run = np.unique(np.array([s2s_voxs, s2b_voxs, b2b_voxs, b2s_voxs]))

while len(voxs2run) != 100:
    sig_thr_temp = np.percentile(scores_s2s, 90)
    s2s_voxs = np.random.choice(np.where(scores_s2s > sig_thr_temp)[0], 25)
    sig_thr_temp = np.percentile(scores_s2b, 90)
    s2b_voxs = np.random.choice(np.where(scores_s2b > sig_thr_temp)[0], 25)
    sig_thr_temp = np.percentile(scores_b2b, 90)
    b2b_voxs = np.random.choice(np.where(scores_b2b > sig_thr_temp)[0], 25)
    sig_thr_temp = np.percentile(scores_b2s, 90)
    b2s_voxs = np.random.choice(np.where(scores_b2s > sig_thr_temp)[0], 25)
    voxs2run = np.unique(np.array([s2s_voxs, s2b_voxs, b2b_voxs, b2s_voxs]))
    
perm_scores_s2s = []
perm_scores_s2b = []
perm_scores_b2b = []
perm_scores_b2s = []
num_perms=100000
for vox in voxs2run:
    print vox
    vox_coord = voxel_indices[vox,:]
    sphere_inds = get_voxel_sphere(vox_coord, voxel_indices)
    X = fds.samples[:,sphere_inds]
    
    cv_score_s2s = np.zeros([run_num])
    cv_score_s2b = np.zeros([run_num])
    cv_score_b2b = np.zeros([run_num])
    cv_score_b2s = np.zeros([run_num])
    cv_count = -1
    
    y_preds_s2s = np.zeros(len(y))
    y_preds_s2b = np.zeros(len(y))
    y_preds_b2b = np.zeros(len(y))
    y_preds_b2s = np.zeros(len(y))
    for train, test in gkf.split(X, y, groups=cv_groups):
        #train within category test within and across category
        train_s = np.intersect1d(train, sitem_inds)
        test_s = np.intersect1d(test, sitem_inds)
        train_b = np.intersect1d(train, bundle_inds)
        test_b = np.intersect1d(test, bundle_inds)
        
        #train on single item
        sk_ridge.fit(X[train_s,:],y[train_s])
        y_preds_s2s[test_s] = sk_ridge.predict(X[test_s,:])
        y_preds_s2b[test_b] = sk_ridge.predict(X[test_b,:])            
        
        #train on bundle
        sk_ridge.fit(X[train_b,:],y[train_b])
        y_preds_b2b[test_b] = sk_ridge.predict(X[test_b,:])
        y_preds_b2s[test_s] = sk_ridge.predict(X[test_s,:]) 
        
        cv_count+=1
    for perm in range(num_perms):        
        y_s_perm = np.random.permutation(y[sitem_inds])
        perm_scores_s2s.append(get_correlation(y_preds_s2s[sitem_inds], y_s_perm))
        perm_scores_b2s.append(get_correlation(y_preds_b2s[sitem_inds], y_s_perm))
        
        y_b_perm = np.random.permutation(y[bundle_inds])
        perm_scores_s2b.append(get_correlation(y_preds_s2b[bundle_inds], y_b_perm))
        perm_scores_b2b.append(get_correlation(y_preds_b2b[bundle_inds], y_b_perm))
        
if not os.path.isdir(bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/'):
    os.mkdir(bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/')
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2s'
np.save(vector_file,np.array(perm_scores_s2s))
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_s2b'
np.save(vector_file,np.array(perm_scores_s2b))
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2b'
np.save(vector_file,np.array(perm_scores_b2b))
vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/perm_tests/perms_b2s'
np.save(vector_file,np.array(perm_scores_b2s))
        
##get pvals
#pvals_s2s = np.zeros(num_voxs)
#lowest_pval = 1.0/len(perm_scores_s2s)
#max_perm_score = np.max(perm_scores_s2s)
##give any voxels that are higher than the max the lowest pval
#over_thr = np.where(scores_s2s > max_perm_score)[0]
#pvals_s2s[over_thr] = lowest_pval
#
##to speed up pval computation, give anything below the median 0.5, or write in a custom thr
#median_perm_score = np.median(perm_scores_s2s)
##median_perm_score = 0.07
#below_med = np.where(scores_s2s <= median_perm_score)[0]
#pvals_s2s[below_med] = 0.5
#und_thr = np.intersect1d(np.where(scores_s2s <= max_perm_score)[0], np.where(scores_s2s > median_perm_score)[0])
#start_time = time.time()
#for ind in und_thr:
#    pvals_s2s[ind] = get_pval(perm_scores_s2s, scores_s2s[ind])
#    #pvals_s2s[ind] = np.where(perm_scores_s2s > scores_s2s[ind])[0].shape[0]/float(len(perm_scores_s2s))
#print time.time() - start_time
#    
##make multiple comparisons correction
#for alpha in range(2,5):
#    alpha=10**-alpha
#    method='fdr_bh'
#    reject, pvalscorr = multipletests(pvals_s2s, alpha=alpha, method=method)[:2]
#    print 'Alpha ',alpha
#    print 'Number sig voxs ',np.where(reject == 1)[0].shape
#    print 'Threshold ',np.min(scores_s2s[reject])
#    fdr_thr = round(np.min(scores_s2s[reject]),2)