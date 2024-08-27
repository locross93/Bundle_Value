#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:16:37 2021

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

###SCRIPT ARGUMENTS

start_time = time.time()

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103']
subj_list = ['109']

analysis_name = 'cross_decoding_rel_value'

relative_value = True

save = True

subsample = True

#None or a number 
save_incomplete = 10000

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   

###SCRIPT ARGUMENTS END
for subj in subj_list:
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii.gz'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/pfc_full_bin.nii.gz'
    
    #fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
    
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
    
    scores_s2s = np.zeros([num_vox])
    scores_s2b = np.zeros([num_vox])
    scores_b2b = np.zeros([num_vox])
    scores_b2s = np.zeros([num_vox])
    
    voxs2run = np.arange(num_vox)
    prev_time = time.time()
    for vox in voxs2run:
        vox_coord = voxel_indices[vox,:]
        sphere_inds = get_voxel_sphere(vox_coord, voxel_indices)
        X = fds.samples[:,sphere_inds]
        
        cv_score_s2s = np.zeros([run_num])
        cv_score_s2b = np.zeros([run_num])
        cv_score_b2b = np.zeros([run_num])
        cv_score_b2s = np.zeros([run_num])
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
            
            #train on bundle
            sk_ridge.fit(X[train_b,:],y[train_b])
            y_preds_in = sk_ridge.predict(X[test_b,:])
            cv_score_b2b[cv_count] = get_correlation(y_preds_in,y[test_b])
            y_preds_out = sk_ridge.predict(X[test_s,:])
            cv_score_b2s[cv_count] = get_correlation(y_preds_out,y[test_s])
            
            cv_count+=1
        scores_s2s[vox] = np.mean(cv_score_s2s)
        scores_s2b[vox] = np.mean(cv_score_s2b)
        scores_b2b[vox] = np.mean(cv_score_b2b)
        scores_b2s[vox] = np.mean(cv_score_b2s)
        
        #how close are we to finishing
        vox_ind = np.where(vox == voxs2run)[0][0]
        print vox_ind,'/',voxs2run.shape[0]
        
        current_time = time.time()
        time_dif_s = current_time - start_time
        time_dif_hrs = round((current_time - start_time)/3600,2)
        
        #how long do we have left
        if vox_ind%50==0:
            log_flag=True
            print 'Time elapsed: ',time_dif_hrs,' hrs'
            
            #estimate analysis rate per voxel every 100 voxs
            time10vox = current_time - prev_time
            time_per_vox = time10vox/50
            remaining_vox = voxs2run.shape[0] - vox_ind
            time_left = round((remaining_vox*time_per_vox)/3600,2)
            print 'Estimated time left: ',time_left,' hrs'
            prev_time = current_time
            
        #save array every X voxels, delete temp folder contents when done
        if save_incomplete and vox_ind > 0 and vox_ind%save_incomplete==0:
            temp_folder = bundle_path+'mvpa/analyses/sub'+str(subj)+'/temp/'
            if not os.path.isdir(temp_folder):
                os.makedirs(temp_folder)
            vector_file = temp_folder+analysis_name+'_s2s_vox'+str(vox_ind)
            np.save(vector_file,scores_s2s)
            vector_file = temp_folder+analysis_name+'_s2b_vox'+str(vox_ind)
            np.save(vector_file,scores_s2b)
            vector_file = temp_folder+analysis_name+'_b2b_vox'+str(vox_ind)
            np.save(vector_file,scores_b2b)
            vector_file = temp_folder+analysis_name+'_b2s_vox'+str(vox_ind)
            np.save(vector_file,scores_b2s)
            
    #save
    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2s'
    h5save(vector_file,scores_s2s)
    nimg0 = map2nifti(fds, scores_s2s)
    nii_file0 = vector_file+'.nii.gz'
    nimg0.to_filename(nii_file0)
    
    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_s2b'
    h5save(vector_file,scores_s2b)
    nimg1 = map2nifti(fds, scores_s2b)
    nii_file1 = vector_file+'.nii.gz'
    nimg1.to_filename(nii_file1)
    
    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2b'
    h5save(vector_file,scores_b2b)
    nimg2 = map2nifti(fds, scores_b2b)
    nii_file2 = vector_file+'.nii.gz'
    nimg2.to_filename(nii_file2)
    
    vector_file = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'_b2s'
    h5save(vector_file,scores_b2s)
    nimg3 = map2nifti(fds, scores_b2s)
    nii_file3 = vector_file+'.nii.gz'
    nimg3.to_filename(nii_file3)
        