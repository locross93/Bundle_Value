#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 13:46:35 2021

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
from scipy.io import loadmat

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Med_Orb','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform','OFClat','OFCmed']

#mask_loop = ['Frontal_Inf_Oper']

square_dsm_bool = False
ranked = True
remove_within_day = True

for subj in subj_list:
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    #load model dsms
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    choice_item_values = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/choice_item_values.txt')
    bundle_inds = np.where(choice_item_values[:,1] > -1)[0]
    item_values_bundles = choice_item_values[bundle_inds,:]
    
    assert(fds.shape[0] == len(bundle_inds))
    num_trials = len(bundle_inds)
    
    ds_bitem_values = dataset_wizard(item_values_bundles, targets=np.zeros(num_trials))
    dsm = PDist(pairwise_metric='cityblock', square=square_dsm_bool)
    res_bun_vector = dsm(ds_bitem_values)
    if ranked:
        res_bun_vector = rankdata(res_bun_vector)
    else:
        res_bun_vector = res_bun_vector.samples.reshape(-1)
    target_dsms['bun_vector'] = res_bun_vector
    
    model_dsm_names = ['bvalue','bun_vector']
        
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
        
        #add PCA on the FMRI data
        #pca = PCA(n_components=5)
        #pca_fds_mask = pca.fit_transform(fds_mask.samples)
        #dataset_fmri = dataset_wizard(pca_fds_mask, targets=np.zeros(len(pca_fds_mask)))
        
        dsm_func = rsa.PDist(pairwise_metric='Correlation', square=square_dsm_bool)
        fmri_dsm = dsm_func(fds_mask)
        #fmri_dsm = dsm_func(dataset_fmri)
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
    
    mask_mat = np.zeros_like(fmri2model_matrix)
    mask_mat[np.triu_indices_from(mask_mat)] = True   
    f = sns.heatmap(fmri2model_matrix, annot=True, annot_kws={"size": 7},  
                    xticklabels=model_dsm_names, yticklabels=mask_loop, vmin=0.0, vmax=0.1, cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
        