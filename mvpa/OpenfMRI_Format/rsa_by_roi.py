#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  7 16:41:04 2019

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

square_dsm_bool = True

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

#model_dsms = np.column_stack((res_stim_id, res_value.samples.reshape(-1)))

#rsa_reg = rsa.Regression(model_dsms, pairwise_metric='correlation', keep_pairs=btwn_run_inds)

mask_loop = ['sup_frontal_gyr', 'acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC']

mask_loop = ['m_OFC', 'l_OFC', 'posterior_OFC']


#temporarily add an SVR
alp=3
ridge = RidgeReg(lm=10**alp)
cv = CrossValidation(ridge, NFoldPartitioner(attr='chunks', count=11), errorfx=correlation)

for mask in mask_loop:
    
    mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
    fds_mask = mask_dset(fds, mask_name)
    
    print 'Mask',mask
    
    #temporary break ROI into parts 
    num_voxs = fds_mask.shape[1]
    num_splits = num_voxs / 1000
    all_inds = np.arange(num_voxs)
    split_inds = np.array_split(all_inds,num_splits)
    split_count = 0
    for split in split_inds[:5]:
        split_count+=1
        #ridge_res = cv(fds_mask[:,split])
        
        #print 'Split ',split_count
        #print 'Value class. acc ',np.mean(ridge_res)
        
        fmri_value = roiRSA_1Ss(fds_mask[:,split],mask_name,res_value,partial_dsm=res_day,cmetric='spearman')
        print 'Mask ',mask
        print 'Value Correl ',fmri_value[0]
        
    print '\n'
    

#    res_mask = rsa_reg(fds_mask)
#    
#    print 'Mask ',mask
#    print 'Stim ID Coef ',res_mask.samples[0]
#    print 'Value Coef ',res_mask.samples[1]
#    print '\n'
    
#    fmri_value = roiRSA_1Ss(fds_mask,mask_name,res_value,partial_dsm=res_day,cmetric='spearman')
#    print 'Mask ',mask
#    print 'Value Correl ',fmri_value[0]
#    print '\n'