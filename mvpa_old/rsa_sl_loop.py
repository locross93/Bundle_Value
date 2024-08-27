#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 09:44:01 2021

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

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['105','107','108','109','110','111','113','114']

analysis_name = 'bundle_value_btrials'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
conditions = ['Food bundle','Trinket bundle','Mixed bundle']

for subj in subj_list:
    print subj
    #which ds to use and which mask to use
    #glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/tstat_all_trials_4D.nii'
    mask_name = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
    if analysis_name == 'rel_value':
        fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions, train_on='default', test_on='default', relative_value=True)
    else:
        fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)
    
    #h5save('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/all_trials_4D_pfc.hdf5',fds)
    
    #fds = h5load('/Users/logancross/Documents/Bundle_Value/mvpa/datasets/sub'+str(subj)+'/all_trials_4D_fullbrain')
    
    square_dsm_bool = False
    
    remove_within_day = True
    
    ranked = True
    
    target_variable = 'bvalue'
    
    ###SCRIPT ARGUMENTS END
    
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    res_value = target_dsms[target_variable]
    
    if __debug__:
        debug.active += ["SLC"]
    
    num_trials = fds.shape[0]
    chunks = fds.chunks
    tdsm = mvpa_utils.rsa_custom(res_value, num_trials, chunks, square_dsm_bool, remove_within_day, pairwise_metric='correlation', comparison_metric='spearman')
    sl_rsa = sphere_searchlight(ChainLearner([tdsm, TransposeMapper()]), radius=3)
    sl_fmri_res = sl_rsa(fds)
    
    # reverse map scores back into nifti format
    scores_per_voxel = sl_fmri_res.samples
    vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_'+analysis_name
    #vector_file = '/Users/logancross/Documents/Bundle_Value/mvpa/analyses/sub'+str(subj)+'/rsa_'+target_variable+'_tstat'
    h5save(vector_file,scores_per_voxel)
    nimg = map2nifti(fds, scores_per_voxel)
    nii_file = vector_file+'.nii.gz'
    nimg.to_filename(nii_file)