#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 16:13:30 2021

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
import statsmodels.api as sm

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['111']

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

mask_loop = ['Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal']

mask_names = mask_loop

relative_value = True
square_dsm_bool = False
ranked = True
remove_within_day = True
save = True

all_sub_fmri_dsms_list = []
all_sub_model_dsms_list = []
for subj in subj_list:
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel2/all_trials_4D.nii.gz'
    #glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/tstat_all_trials_4D.nii'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
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
        
    #model_dsm_names = ['stim_id', 'value', 'ivalue', 'bvalue', 'choice', 'lr_choice', 'choice_diff', 'item_or_bundle']
    #model_dsm_names = ['value','stim_id','choice','lr_choice','choice_diff','item_or_bundle','rt']
    model_dsm_names = ['value','choice','choice_diff','item_or_bundle','stim_id','lr_choice']
        
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
            
        if remove_within_day:
            fmri_dsm = scipy.stats.zscore(fmri_dsm[btwn_day_inds])
        else:
            fmri_dsm = scipy.stats.zscore(fmri_dsm)
        
        fmri_dsm_list.append(fmri_dsm)
        
    #correlate mask dsms to each other
    mask_dsm_all = np.vstack(fmri_dsm_list)
    mask_correl = np.corrcoef(mask_dsm_all)
    #make heatmap
    mask_mat = np.zeros_like(mask_correl)
    mask_mat[np.triu_indices_from(mask_mat)] = True
    f = sns.heatmap(mask_correl, annot=True, annot_kws={"size": 7}, mask=mask_mat, 
                    xticklabels=mask_names, yticklabels=mask_names, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Correlation (r)'})
    plt.xticks(rotation=90)
    plt.title('Sub'+str(subj)+' RSA ROI Correlation')
    plt.show()
    
    #correlate model dsms to each other
#    model_dsms = []
#    for name in model_dsm_names:
#        model_dsms.append(target_dsms[name])
#    model_dsm_all = np.vstack(model_dsms)
#    model_correl = np.corrcoef(model_dsm_all)
#    #make heatmap
#    mask_mat = np.zeros_like(model_correl)
#    mask_mat[np.triu_indices_from(mask_mat)] = True
#    f = sns.heatmap(model_correl, annot=True, annot_kws={"size": 7}, mask=mask_mat, 
#                    xticklabels=model_dsm_names, yticklabels=model_dsm_names, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Correlation (r)'})
#    plt.xticks(rotation=90)
#    plt.show()
        
        
    fmri2model_matrix = np.zeros([len(mask_loop), len(model_dsm_names)])
    model_fits = np.zeros(len(mask_loop))
    if remove_within_day:
        model_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
    else:
        model_dsms = [target_dsms[model_dsm] for model_dsm in model_dsm_names]
    model_dsm_array = np.column_stack((model_dsms))
    model_dsm_array = scipy.stats.zscore(model_dsm_array, axis=0)
    for mask_num in range(len(mask_loop)):
        temp_fmri = fmri_dsm_list[mask_num]
        mod = sm.OLS(temp_fmri, model_dsm_array)
        res = mod.fit()
        fmri2model_matrix[mask_num, :] = res.params
        model_fits[mask_num] = res.rsquared
        
    #add to master matrix
    all_sub_fmri_dsms_list.append(fmri_dsm_list)
    all_sub_model_dsms_list.append(model_dsm_array)
        
    legend_label = 'Coefficient'
    vmax = 0.1
    mask_mat = np.zeros_like(fmri2model_matrix)
    mask_mat[np.triu_indices_from(mask_mat)] = True   
    f = sns.heatmap(fmri2model_matrix, annot=True, annot_kws={"size": 7},  
                    xticklabels=model_dsm_names, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
    
    plt.bar(np.arange(len(mask_names)), model_fits)
    plt.xticks(np.arange(len(mask_names)), mask_names, rotation=90)
    plt.ylabel('R-Squared')
    plt.title('Sub'+str(subj)+' RSA')
    plt.show()
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=model_dsm_names)
        subj_df.to_csv(save_path+'rsa_reg_subcortical_scores.csv')
        
##group regression
#fmri2model_matrix_group = np.zeros([len(mask_loop), len(model_dsm_names)])
#all_sub_model_dsms_array = np.vstack(all_sub_model_dsms_list)
#mask_group_scores = []
#for mask_num in range(len(mask_loop)):
#    print mask_names[mask_num]
#    mask_dsms = []
#    sub_num_list = []
#    for sub_num in range(len(subj_list)):
#        sublist = all_sub_fmri_dsms_list[sub_num]
#        mask_dsm_subj = sublist[mask_num]
#        mask_dsms.append(mask_dsm_subj)
#        sub_num_list.append([subj_list[sub_num] for i in range(len(mask_dsm_subj))])
#    mask_dsm_array = np.concatenate(mask_dsms)
#    sub_num_array = np.concatenate(sub_num_list)
#    assert len(mask_dsm_array) == len(sub_num_array)
#    assert len(mask_dsm_array) == len(all_sub_model_dsms_array)
#    all_data = np.column_stack([sub_num_array, mask_dsm_array, all_sub_model_dsms_array])
#    column_names = ['Subj','fMRI'] + model_dsm_names
#    df_mask = pd.DataFrame(data=all_data, columns=column_names)
#    #md = smf.mixedlm('fMRI ~ 1 + value + choice + choice_diff + item_or_bundle + stim_id + lr_choice', df_mask, groups=df_mask['Subj'], re_formula='~ 1 + value + choice + choice_diff + item_or_bundle + stim_id + lr_choice')
#    #md = smf.mixedlm('fMRI ~ 1 + value + choice + choice_diff + item_or_bundle + stim_id + lr_choice', df_mask, groups=df_mask['Subj'])
#    #mdf = md.fit()
#    mdf = smf.ols(formula = 'fMRI ~ 1 + value + choice + choice_diff + item_or_bundle + stim_id + lr_choice', data = df_mask).fit()
#    print(mdf.summary())
#    mask_group_scores.append(mdf)
#    fmri2model_matrix_group[mask_num, :] = mdf.params[1:]
#legend_label = 'Coefficient'
#vmax = 0.1
#mask_mat = np.zeros_like(fmri2model_matrix_group)
#mask_mat[np.triu_indices_from(mask_mat)] = True   
#f = sns.heatmap(fmri2model_matrix_group, annot=True, annot_kws={"size": 7},  
#                xticklabels=model_dsm_names, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
#f.set_title('RSA Regression - Group Level', fontsize = 20)   
#plt.show()
    
        
        
        
        