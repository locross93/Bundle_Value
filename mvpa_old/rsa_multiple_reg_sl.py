#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:55:15 2019

@author: logancross
"""

from mvpa2.suite import *
from pymvpaw import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.rsa import PDist
from mvpa2.measures.searchlight import sphere_searchlight
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from sklearn.preprocessing import MinMaxScaler
import mvpa_utils

###SCRIPT ARGUMENTS

subj = 101

analysis_name = 'abs_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
conditions = ['Food item', 'Trinket item']

###SCRIPT ARGUMENTS END

#which ds to use and which mask to use
#glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/tstat_all_trials_4D.nii'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'

fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

square_dsm_bool = True

#control dsm for day
num_trials = fds.shape[0]
trials_per_day = num_trials/3
day_array = np.array([c/trials_per_day for c in range(num_trials)])
ds_day = dataset_wizard(day_array, targets=np.zeros(num_trials))

dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_day = dsm(ds_day)

#control dsm for run
ds_run = dataset_wizard(fds.chunks, targets=np.zeros(num_trials))

dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_run = dsm(ds_run)

#plot_mtx(res_run, ds_value.sa.targets, 'ROI pattern correlation distances')

#stimulus identity
item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
inds_in_conds = np.where(item_list[:,1] == -1)[0]
ind_item_list = item_list[inds_in_conds, 0]

ds_item_identity = dataset_wizard(ind_item_list, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_stim_id = dsm(ds_item_identity)

#plot_mtx(res_stim_id, ds_value.sa.targets, 'ROI pattern correlation distances')

#value
value = fds.targets
#value_norm = zscore(value)
scaler = MinMaxScaler()
value_norm = scaler.fit_transform(value.reshape(-1,1)).reshape(-1)
ds_value = dataset_wizard(value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_value = dsm(ds_value)

#food value
trial_categ = fds.sa.trial_categ
food_inds = np.where(trial_categ == 1)[0]
food_value = value[food_inds]
#make the trinket value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
mean_value = np.mean(food_value)
food_value_norm = mean_value*np.ones(num_trials)
food_value_norm[food_inds] = food_value
food_value_norm = scaler.fit_transform(food_value_norm.reshape(-1,1))
ds_fvalue = dataset_wizard(food_value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_fvalue = dsm(ds_fvalue)

#plot_mtx(res_fvalue, ds_fvalue.sa.targets, 'ROI pattern correlation distances')

#trinket value
trinket_inds = np.where(trial_categ == 2)[0]
trinket_value = value[trinket_inds]
#make the food value trials have a value of the mean value so they are in the middle of a dissimilarity matrix with minmaxscaling
mean_value = np.mean(trinket_value)
trinket_value_norm = mean_value*np.ones(num_trials)
trinket_value_norm[trinket_inds] = trinket_value
trinket_value_norm = scaler.fit_transform(trinket_value_norm.reshape(-1,1))
ds_tvalue = dataset_wizard(trinket_value_norm, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_tvalue = dsm(ds_tvalue)

#plot_mtx(res_tvalue, ds_tvalue.sa.targets, 'ROI pattern correlation distances')

#food vs trinket category
ds_item_categ = dataset_wizard(trial_categ, targets=np.zeros(num_trials))
dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
res_item_categ = dsm(ds_item_categ)

sl_fmri_value = slRSA_m_1Ss(fds, res_value, partial_dsm = res_day, radius=3, cmetric='pearson')

# reverse map scores back into nifti format
scores_per_voxel = sl_fmri_value
vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_value_pcday'
h5save(vector_file,scores_per_voxel)
nimg = map2nifti(fds, scores_per_voxel)
nii_file = vector_file+'.nii.gz'
nimg.to_filename(nii_file)
