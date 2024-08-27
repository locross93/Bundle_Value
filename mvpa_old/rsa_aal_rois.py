#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 12:46:57 2021

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
from mvpa2.measures import rsa
from mvpa2.measures.rsa import PDist
from scipy.spatial.distance import squareform
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import pandas as pd

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['106']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Med_Orb','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform','OFClat','OFCmed']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
             'Calcarine', 'Fusiform']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

#mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
#    'Insula','OFCant','OFClat','OFCmed','OFCpost',
#    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
#    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
#    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
#    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate
#             
#mask_names = mask_loop

#mask_loop = ['Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal']

#mask_names = mask_loop

relative_value = True
square_dsm_bool = False
ranked = True
remove_within_day = True
save = False

for subj in subj_list:
    #mvpa_prep_path = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/mvpa_prep2/'
    mvpa_prep_path = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/'
    #which ds to use and which mask to use
    glm_ds_file = mvpa_prep_path+'all_trials_4D.nii.gz'
    #glm_ds_file = mvpa_prep_path+'tstat_all_trials_4D.nii.gz'
    brain_mask = mvpa_prep_path+'mask.nii'
    
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
    
    #load model dsms
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    if remove_within_day:
        res_day = target_dsms['day']
        if ranked:
            day_values = np.unique(res_day)
            high_rank = np.max(day_values)
            btwn_day_inds = np.where(res_day == high_rank)[0]
        else:
            btwn_day_inds = np.where(res_day == 1)[0]
        
    model_dsm_names = ['stim_id', 'value', 'ivalue', 'bvalue', 'choice', 'lr_choice', 'choice_diff', 'item_or_bundle']
    model_dsm_names = ['value','choice','choice_diff','item_or_bundle','stim_id','lr_choice','rt']
        
    fmri_dsm_list = []
    for mask in mask_loop:
        mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        brain_mask = mvpa_prep_path+'mask.nii'
        #brain_mask = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
        masked = fmri_dataset(mask_name, mask=brain_mask)
        reshape_masked=masked.samples.reshape(fds.shape[1])
        reshape_masked=reshape_masked.astype(bool)
        mask_map = mask_mapper(mask=reshape_masked)
        
        mask_slice = mask_map[1].slicearg
        mask_inds = np.where(mask_slice == 1)[0]
        
        fds_mask = fds[:,mask_inds]
        
        #dsm_func = rsa.PDist(pairwise_metric='Correlation', square=square_dsm_bool)
        dsm_func = rsa.PDist(pairwise_metric='Euclidean', square=square_dsm_bool)
        
        #add PCA on the FMRI data
        #pca = PCA(n_components=5)
        #pca_fds_mask = pca.fit_transform(fds_mask.samples)
        #dataset_fmri = dataset_wizard(pca_fds_mask, targets=np.zeros(len(pca_fds_mask)))
        #fmri_dsm = dsm_func(dataset_fmri)
        
        #no PCA
        fmri_dsm = dsm_func(fds_mask)
        
        #just take samples to make lighter array and rank
        if ranked:
            fmri_dsm = rankdata(fmri_dsm.samples)
        else:
            fmri_dsm = fmri_dsm.samples.reshape(-1)
        
        fmri_dsm_list.append(fmri_dsm)
        
    fmri2model_matrix = np.zeros([len(mask_loop), len(model_dsm_names)])
    for mask_num in range(len(mask_loop)):
        for model_num in range(len(model_dsm_names)):
            if remove_within_day:
                temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
                temp_model = target_dsms[model_dsm_names[model_num]][btwn_day_inds]
            else:
                temp_fmri = fmri_dsm_list[mask_num]
                temp_model = target_dsms[model_dsm_names[model_num]]
            temp_correl = pearsonr(temp_fmri, temp_model)[0]
            fmri2model_matrix[mask_num, model_num] = temp_correl
    
    if ranked:
        legend_label = 'Spearman Correlation ($\\rho$)'
    else:
        legend_label = 'Pearson Correlation (r)'
    vmax = 0.1
    mask_mat = np.zeros_like(fmri2model_matrix)
    mask_mat[np.triu_indices_from(mask_mat)] = True   
    f = sns.heatmap(fmri2model_matrix, annot=True, annot_kws={"size": 7},  
                    xticklabels=model_dsm_names, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=model_dsm_names)
        subj_df.to_csv(save_path+'rsa_subcortical_scores.csv')
    
    #rsa regression
    #rsa_reg = Lasso(alpha=1.0, fit_intercept=True)
    #rsa_reg = LinearRegression(fit_intercept=True)
#    if remove_within_day:
#        model_dsm_names = ['stim_id', 'fvalue', 'tvalue', 'bvalue']
#        model_regressors = np.column_stack(([scale(target_dsms[model][btwn_day_inds]) for model in model_dsm_names]))
#    else:
#        model_dsm_names = ['run','day','stim_id', 'fvalue', 'tvalue', 'bvalue']
#        model_regressors = np.column_stack(([scale(target_dsms[model]) for model in model_dsm_names]))
#    
#    rsa_reg = Ridge(alpha=10**4, fit_intercept=True)
#    
#    fmri2model_matrix_reg = np.zeros([len(mask_loop), len(model_dsm_names)])
#    for mask_num in range(len(mask_loop)):
#        print mask_loop[mask_num]
#        if remove_within_day:
#            temp_fmri = scale(fmri_dsm_list[mask_num][btwn_day_inds])
#        else:
#            temp_fmri = scale(fmri_dsm_list[mask_num])
#        rsa_reg.fit(model_regressors, temp_fmri)
#        coefs = rsa_reg.coef_.reshape(-1)
#        fmri2model_matrix_reg[mask_num,:] = coefs
#        
#    mask_mat = np.zeros_like(fmri2model_matrix_reg)
#    mask_mat[np.triu_indices_from(mask_mat)] = True   
#    f = sns.heatmap(fmri2model_matrix_reg, annot=True, annot_kws={"size": 7},  
#                    xticklabels=model_dsm_names, yticklabels=mask_loop, vmin=0.0, vmax=0.1, cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}) 
#    f.set_title('Sub'+str(subj)+' RSA Regression', fontsize = 20)   
#    plt.show()

    #rsa by category
    #stim_id include individual item trials
#    def slice_dsm(inds, dsm):
#        #takes in 1D vector of trials to use, and outputs appropriate inds for dsm
#        square_dsm = squareform(dsm)
#        slice_square_dsm = square_dsm[inds, :]
#        slice_square_dsm = slice_square_dsm[:,inds]
#        new_dsm = squareform(slice_square_dsm)
#        
#        return new_dsm
#    
#    if remove_within_day:
#        res_day = target_dsms['day']
#        res_day_slice = 
#    
#    
#    stim_id_scores = []
#    ind_item_trials = np.where(fds.sa.trial_categ < 3)[0]
#    for mask_num in range(len(mask_loop)):
#        if remove_within_day:
#            inds2use = np.intersect1d(ind_item_trials, btwn_day_inds)
#            temp_fmri = fmri_dsm_list[mask_num][inds2use]
#            temp_model = target_dsms['stim_id'][inds2use]
#        else:
#            temp_fmri = fmri_dsm_list[mask_num][ind_item_trials]
#            temp_model = target_dsms['stim_id'][ind_item_trials]
            
    
        
    