#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:00:43 2019

@author: logancross
"""

from mvpa2.suite import *
#from pymvpaw import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.rsa import PDist
from mvpa2.measures.searchlight import sphere_searchlight
from scipy.spatial.distance import squareform
from scipy.stats import zscore
from scipy.stats import rankdata, pearsonr
from sklearn.preprocessing import MinMaxScaler
import mvpa_utils

###SCRIPT ARGUMENTS

subj = 101

analysis_name = 'rel_value'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
conditions = ['Food item', 'Trinket item']

#which ds to use and which mask to use
glm_ds_file = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/tstat_all_trials_4D.nii'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

if analysis_name == 'rel_value':
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=True)
else:
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)

square_dsm_bool = False

remove_within_day = True

###SCRIPT ARGUMENTS END

target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool)

res_stim_id = target_dsms['stim_id']
res_fvalue = target_dsms['fvalue']
res_tvalue = target_dsms['tvalue']
res_bvalue = target_dsms['bvalue']
res_value = target_dsms['value']

if remove_within_day:
    res_day = target_dsms['day']
    btwn_run_inds = np.where(res_day.samples == 1)[0]
    
#model_dsms = np.column_stack((res_stim_id, res_fvalue.samples.reshape(-1), \
#                              res_tvalue.samples.reshape(-1), res_bvalue.samples.reshape(-1)))

model_dsms = np.column_stack((res_stim_id, res_value.samples.reshape(-1)))

rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation', keep_pairs=btwn_run_inds)
    
sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3, center_ids=vox2use)
sl_fmri_value = sl_rsa_reg(fds)

#if __debug__:
#    debug.active += ["SLC"]
#    
#sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3)
#sl_fmri_value = sl_rsa_reg(fds)

#TEMPORARY
mask = 'frontal_pole'
mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
fds_mask = mask_dset(fds, mask_name)

vox_inds = fds_mask.fa.voxel_indices
left_inds = np.where(vox_inds[:,0] < 23)[0]
voxel_vars = np.var(fds_mask.samples, axis=0)
nz_inds = np.where(voxel_vars != 0)[0]
selected = np.intersect1d(nz_inds, left_inds)
fds_mask_fs = fds_mask[:,selected]

if __debug__:
    debug.active += ["SLC"]
    
sl_rsa_reg = sphere_searchlight(rsa_reg, radius=3)
sl_fmri_value = sl_rsa_reg(fds_mask_fs)

scores_per_voxel = np.zeros([4,fds_mask.shape[1]])
scores_per_voxel[:,selected] = sl_fmri_value.samples[:-1,:]

#save
nimg0 = map2nifti(fds_mask, scores_per_voxel[1,:])
nii_file0 = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/test_fvalue_rel_val_novar2.nii.gz'
nimg0.to_filename(nii_file0)



    


 