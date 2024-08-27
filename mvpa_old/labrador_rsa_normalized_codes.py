#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:56:56 2022

@author: logancross
"""

#prevent multithreading
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
sys.path.insert(0, '/state/partition1/home/lcross/Bundle_Value/mvpa')
os.chdir('/state/partition1/home/lcross/Bundle_Value/mvpa')

from mvpa2.base.hdf5 import h5load, h5save
from mvpa2.datasets.mri import fmri_dataset
from mvpa2.datasets.base import mask_mapper
from mvpa2.measures.rsa import rankdata
from mvpa2.measures import rsa
from matplotlib import pyplot as plt
import mvpa_utils_lab
import time
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import seaborn as sns

def compute_divisive_norm(values, beta, sigma):
    avg_value = np.mean(values)
    div_normed_values = (beta + values.astype(float)) / (sigma + avg_value)
    
    return div_normed_values

def compute_monte_carlo_norm(values, beta, sigma):
    mc_normed_values = values
    for i,val in enumerate(values):
        if val == 0:
            mc_normed_values[i] = 0
        else:
            mc_normed_values[i] = (beta + values.astype(float)) / (sigma + np.mean(values[:i+1]))
        
    return mc_normed_values

def compute_rw_update(values, alpha, ev0, beta=0, sigma=1):
    ev = ev0
    rw_normed_values = values
    ev_t = np.zeros(len(values))
    for i,val in enumerate(values):
        pe = val - ev
        ev = ev + (alpha*pe)
        rw_normed_values[i] = (beta + val.astype(float)) / (sigma + ev)
        ev_t[i] = ev
        
    return rw_normed_values, ev_t


###SCRIPT ARGUMENTS
start_time = time.time()

bundle_path = '/state/partition1/home/lcross/Bundle_Value/'

subj_list = ['101','102','103']

save = True
save_prefix = 'rsa_normalized_codes'
save_dsms = True

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

square_dsm_bool = False
ranked = False
remove_within_day = True

value_type = ['Abs', 'Rel', 'Subtract Median','Subtract Mean','Divide Mean']
xaxis_labels = ['Absolute','Relative','WTP - Ref','Subtract Mean','Divide Mean']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

for subj in subj_list:
    print(subj)
    #which ds to use and which mask to use
    #glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D_pfc_mask.nii.gz'
    brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    
    fmri2model_matrix = np.zeros([len(mask_loop), len(value_type)])
    
    count = 0
    for val_type in value_type:
        print(val_type)
        if val_type == 'Abs':
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
        elif val_type == 'Rel':   
            relative_value = True
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            zscore_value = fds.targets
        elif val_type == 'Subtract Median':   
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            # wtp - ref - IMPROVE THIS WITH REAL REF
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) - np.median(abs_value[sitem_inds]) 
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) - np.median(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
        elif val_type == 'Subtract Mean':   
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) - np.mean(abs_value[sitem_inds])
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) - np.mean(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
        elif val_type == 'Divide Mean':   
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) / np.mean(abs_value[sitem_inds]) 
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) / np.mean(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
        elif val_type == 'Divisive':   
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            divnorm_value = np.zeros([len(abs_value)])
            beta = 0
            sigma = 0.4
            divnorm_value[sitem_inds] = compute_divisive_norm(abs_value[sitem_inds], beta, sigma) 
            divnorm_value[bundle_inds] = compute_divisive_norm(abs_value[bundle_inds], beta, sigma) 
            fds.targets = divnorm_value
        elif val_type == 'Running Average':
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            mcnorm_values = np.zeros([len(abs_value)])
            mcnorm_values[sitem_inds] = compute_monte_carlo_norm(abs_value[sitem_inds])
            mcnorm_values[bundle_inds] = compute_monte_carlo_norm(abs_value[bundle_inds])
            fds.targets = mcnorm_values
        elif val_type == 'RW Update':
            relative_value = False
            fds = mvpa_utils_lab.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            rwnorm_values = np.zeros([len(abs_value)])
            alpha = 0.05
            beta = 10
            sigma = 0
            rwnorm_values[sitem_inds], ev_t = compute_rw_update(abs_value[sitem_inds], alpha, ev0=np.mean(abs_value[sitem_inds]), beta=beta, sigma=sigma)
            rwnorm_values[bundle_inds], ev_t = compute_rw_update(abs_value[bundle_inds], alpha, ev0=np.mean(abs_value[bundle_inds]), beta=beta, sigma=sigma)
            fds.targets = rwnorm_values
            
        #load model dsms
        target_dsms = mvpa_utils_lab.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
        
        if remove_within_day:
            res_day = target_dsms['day']
            if ranked:
                day_values = np.unique(res_day)
                high_rank = np.max(day_values)
                btwn_day_inds = np.where(res_day == high_rank)[0]
            else:
                btwn_day_inds = np.where(res_day == 1)[0]
                
        num_trials = fds.shape[0]
        day_array = np.zeros(num_trials)
        run_array = fds.chunks
        day2_inds = np.intersect1d(np.where(run_array > 4)[0],np.where(run_array < 10)[0])
        day_array[day2_inds] = 1
        day3_inds = np.where(run_array >= 10)[0]
        day_array[day3_inds] = 2
                
        model_dsm_names = ['value']
        
        fmri_dsm_list = []
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
            
            dsm_func = rsa.PDist(pairwise_metric='Euclidean', square=square_dsm_bool)
            
            fmri_dsm = dsm_func(fds_mask)
            
            #just take samples to make lighter array and rank
            if ranked:
                fmri_dsm = rankdata(fmri_dsm.samples)
            else:
                fmri_dsm = fmri_dsm.samples.reshape(-1)
            
            fmri_dsm_list.append(fmri_dsm)
        
        for mask_num in range(len(mask_loop)):
            for model_num in range(len(model_dsm_names)):
                if remove_within_day:
                    temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
                    temp_model = target_dsms[model_dsm_names[model_num]][btwn_day_inds]
                else:
                    temp_fmri = fmri_dsm_list[mask_num]
                    temp_model = target_dsms[model_dsm_names[model_num]]
                temp_correl = pearsonr(temp_fmri, temp_model)[0]
                fmri2model_matrix[mask_num, model_num+(1*count)] = temp_correl
        count+=1
        
    if ranked:
        legend_label = 'Spearman Correlation ($\\rho$)'
    else:
        legend_label = 'Pearson Correlation (r)'
    vmax = 0.1
    mask_mat = np.zeros_like(fmri2model_matrix)
    mask_mat[np.triu_indices_from(mask_mat)] = True   
    f = sns.heatmap(fmri2model_matrix, annot=True, annot_kws={"size": 7},  
                    xticklabels=xaxis_labels, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
    
    subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=xaxis_labels)
    subj_df['ROI'] = subj_df.index
    subj_tidy_df = subj_df.melt(id_vars='ROI',value_vars=xaxis_labels)
    
    f = sns.barplot(x='ROI', y='value', hue='variable', data=subj_tidy_df)
    sns.despine()
    f.set_xticklabels(f.get_xticklabels(), rotation=90)
    plt.legend(bbox_to_anchor=(1.3, 1),borderaxespad=0)
    plt.title('RSA Absolute vs. Relative Value Sub'+subj)
    plt.ylabel('RSA Correlation')
    plt.show()
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        subj_df.to_csv(save_path+save_prefix+'.csv')
        
    # save fmri dsms and target dsms
    if save_dsms:
        save_path = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        save_file = save_path+'fmri_dsm_list'
        if ranked:
            save_file = save_file+'_ranked'
        h5save(save_file, fmri_dsm_list)
        
        save_file2 = save_path+'target_dsms'
        if ranked:
            save_file2 = save_file2+'_ranked'
        h5save(save_file2, target_dsms)
        temp_df = pd.DataFrame.from_dict(target_dsms)
        temp_df.to_csv(save_file2+'.csv')
        
        # other import variables
        subj_info_dict = {}
        subj_info_dict['abs_value'] = abs_value 
        subj_info_dict['trial_categ'] = trial_categ
        subj_info_dict['sitem_inds'] = sitem_inds
        subj_info_dict['bundle_inds'] = bundle_inds
        subj_info_dict['run_array'] = run_array
        subj_info_dict['day_array'] = day_array
        save_file3 = save_path+'info_dict'
        h5save(save_file3, subj_info_dict)
        subj_info_dict_list = [subj_info_dict[key] for key in subj_info_dict.keys()]
        h5save(save_file3+'_list',subj_info_dict_list)
        