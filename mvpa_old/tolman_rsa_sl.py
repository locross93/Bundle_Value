#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:46:08 2019

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
from mvpa2.measures import rsa
from mvpa2.measures.searchlight import sphere_searchlight
import time
import os
import sys

sys.path.insert(0, '/home/lcross/Bundle_Value/mvpa')
os.chdir('/home/lcross/Bundle_Value/mvpa')

from pymvpaw import *
import mvpa_utils

###SCRIPT ARGUMENTS

save_suffix = 'wbrain_rsa_ind_item_value_all_trials'

analysis_name = 'ind_item'

target_variable = 'ivalue'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle'] 
#conditions = ['Food item', 'Trinket item']

subj = int(sys.argv[1])

#which ds to use and which mask to use
#glm_ds_file = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/GLM_betas/all_trials_4D.nii.gz'
#glm_ds_file = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/tstat_all_trials_4D.nii'
#mask_name = '/home/lcross/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
#mask_name = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel_smooth/mask.nii'

glm_ds_file = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii'
mask_name = '/home/lcross/Bundle_Value/analysis/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'

square_dsm_bool = False

remove_within_day = True

###SCRIPT ARGUMENTS END

#make targets with mvpa utils
if analysis_name == 'rel_value':
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=True, tolman=True)
else:
    fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=False, tolman=True)

#ranking happens later in searchlight wrapper
target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked=False, tolman=True)

model_dsm = target_dsms[target_variable]
    
start_time = time.time()
print 'starting searchlight',time.time() - start_time

#sl_map = slRSA_m_1Ss(fds, res_value, partial_dsm = res_day)

num_trials = fds.shape[0]
chunks = fds.chunks
tdsm = mvpa_utils.rsa_custom(model_dsm, num_trials, chunks, square_dsm_bool, remove_within_day, pairwise_metric='correlation', comparison_metric='spearman')
sl_rsa = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), radius=3)
sl_fmri_res = sl_rsa(fds)

sl_time = time.time() - start_time
print 'finished searchlight',sl_time

comp_speed = sl_time/fds.shape[1]
print 'Analyzed at a speed of ',comp_speed,'  per voxel'

#save
scores_per_voxel = sl_fmri_res.samples.reshape(-1)
vector_file = '/home/lcross/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/'+save_suffix
h5save(vector_file,scores_per_voxel)

nimg0 = map2nifti(fds, scores_per_voxel)
nii_file0 = vector_file+'.nii.gz'
nimg0.to_filename(nii_file0)
