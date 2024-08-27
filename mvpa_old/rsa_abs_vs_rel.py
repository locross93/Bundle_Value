#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 18:45:11 2021

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

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['105','106','107','108','109','110','111','112','113','114']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
             'Calcarine', 'Fusiform']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

square_dsm_bool = False
ranked = True
remove_within_day = True
save = True

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

for subj in subj_list:
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
    fmri2model_matrix = np.zeros([len(mask_loop), 2])
    
    count = 0
    bool_list = [True, False]
    for relative_value in bool_list:
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
                
        model_dsm_names = ['value']
        
        fmri_dsm_list = []
        for mask in mask_loop:
            mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
            brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
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
        
    xaxis_labels = ['Relative','Absolute']
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
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=xaxis_labels)
        subj_df.to_csv(save_path+'rsa_abs_vs_rel.csv')
        

