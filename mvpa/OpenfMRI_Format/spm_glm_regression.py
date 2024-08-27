#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 17:30:09 2019

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

###SCRIPT ARGUMENTS

subj = 108

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
#conditions = ['Food item', 'Trinket item']

###SCRIPT ARGUMENTS END

#which ds to use and which mask to use
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii.gz'
#mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/pfc_full_bin.nii.gz'

fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

##feature selection - remove voxels with no variance
#voxel_vars = np.var(fds.samples, axis=0)
#nz_inds = np.where(voxel_vars != 0)[0]
#fds_fs = fds[:,nz_inds]
#
##searchlight
## enable debug output for searchlight call
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
#res_sl = ridge_sl(fds_fs)
#print 'finished searchlight',time.time() - start_time

##save map
#scores_per_voxel = np.zeros(fds.shape[1])
#scores_per_voxel[nz_inds] = res_sl.samples
#
### reverse map scores back into nifti format
#vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/sl_ridge_'+analysis_name
#h5save(vector_file,scores_per_voxel)
#nimg = map2nifti(fds, scores_per_voxel)
#nii_file = vector_file+'.nii.gz'
#nimg.to_filename(nii_file)

#by roi
#zscore targets
fds.targets = scipy.stats.zscore(fds.targets)

#define model
alp=3
ridge = RidgeReg(lm=10**alp)
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)

sk_ridge = linear_model.Ridge(alpha=10*alp)

mask_loop = ['sup_frontal_gyr', 'acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC']
mask_loop = ['frontal_pole']

for mask in mask_loop:
    print mask
    
    mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/lowres/'+mask+'.nii.gz'
    brain_mask = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    masked = fmri_dataset(mask_name, mask=brain_mask)
    reshape_masked=masked.samples.reshape(fds.shape[1])
    reshape_masked=reshape_masked.astype(bool)
    mask_map = mask_mapper(mask=reshape_masked)
    
    mask_slice = mask_map[1].slicearg
    mask_inds = np.where(mask_slice == 1)[0]
    
    fds_mask = fds[:,mask_inds]
    
    fds_mask_small = fds_mask[:,:500]
    
    cv_score_pymvpa  = cv(fds_mask)
    print 'pymvpa score',np.mean(cv_score_pymvpa)
    
    X = fds_mask.samples
    y = fds_mask.targets
    cv_groups = fds_mask.chunks
    
    cv_score_sk  = cross_val_score(sk_ridge,X,y,groups=cv_groups,scoring=r_scorer,cv=15)
    print 'sk score',np.mean(cv_score_sk)
    print '\n'
 
grid_scores = []
for alp in range(-1,4):
    #ridge = RidgeReg(lm=10**alp)
    #cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks'), errorfx=correlation)
    #cv_score = cv(fds_mask)
    
    sk_ridge = linear_model.Ridge(alpha=10*alp)
    cv_score = cross_val_score(sk_ridge,X,y,groups=cv_groups,scoring=r_scorer,cv=15)
    
    grid_scores.append(np.mean(cv_score))
    
#pymvpa vs scikit learn
from sklearn.model_selection import cross_val_score
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer
from sklearn import linear_model

start_time = time.time()
cv_score_pymvpa = cv(fds_mask)
print 'pymvpa ',time.time() - start_time


def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

r_scorer = make_scorer(get_correlation)
X = fds_mask.samples
y = fds_mask.targets
cv_groups = fds_mask.chunks

sk_ridge = linear_model.Ridge(alpha=10*alp)

start_time = time.time()
cv_score_sk = cross_val_score(sk_ridge,X,y,groups=cv_groups,scoring=r_scorer,cv=15)
print 'sk ',time.time() - start_time
    
    