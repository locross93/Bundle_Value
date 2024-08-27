# -*- coding: utf-8 -*-
"""
Created on Mon May 17 14:37:54 2021

@author: locro
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
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
from sklearn.preprocessing import StandardScaler

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['107']

square_dsm_bool = False
ranked = False
remove_within_day = True
#put 1 to remove within day, 0 to remove between day
btwn_or_within = 1
save = False

for subj in subj_list:
    mvpa_prep_path = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/mvpa_value_bins/'
    #which ds to use and which mask to use
    glm_ds_file = mvpa_prep_path+'tstat_all_values_4D.nii'
    #glm_ds_file = mvpa_prep_path+'tstat_all_trials_4D.nii.gz'
    brain_mask = mvpa_prep_path+'mask.nii'
    
    tstat_info = pd.read_csv(mvpa_prep_path+'tstat_table.csv')
    num_tstats = len(tstat_info)
    avg_bin_value = np.mean(tstat_info[['Bin Min', 'Bin Max']].to_numpy(), axis=1)
    trial_cat = tstat_info[['Trial Type']].to_numpy()
    
    #get all values to zscore
    value_allruns, cond_by_trial = mvpa_utils.get_all_values(subj)
    #if relative value, z score across individual item trials and separately z score across bundle trials
    num_trials = len(value_allruns)
    item_inds = [c for c in range(num_trials) if cond_by_trial[c] in [1, 2]]
    bundle_inds = [c for c in range(num_trials) if cond_by_trial[c] in [3, 4, 5]]
    
    #zscore by category
    z_bin_value = np.mean(tstat_info[['Bin Min', 'Bin Max']].to_numpy(), axis=1)
    scaler = StandardScaler()
    scaler.fit(value_allruns[item_inds].reshape(-1, 1))
    sitem_inds = np.where(trial_cat == 0)[0]
    z_bin_value[sitem_inds] = scaler.transform(z_bin_value[sitem_inds].reshape(-1, 1)).reshape(-1)
    
    scaler.fit(value_allruns[bundle_inds].reshape(-1, 1))
    bun_inds = np.where(trial_cat == 1)[0]
    z_bin_value[bun_inds] = scaler.transform(z_bin_value[bun_inds].reshape(-1, 1)).reshape(-1)
    
    fds = fmri_dataset(samples=glm_ds_file, targets=avg_bin_value, mask=brain_mask)
    
    ds_abs_value = dataset_wizard(avg_bin_value, targets=np.zeros(num_tstats))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_abs_value = dsm(ds_abs_value)
    if ranked:
        res_abs_value = rankdata(res_abs_value)
    else:
        res_abs_value = res_abs_value.samples.reshape(-1)
        
    ds_rel_value = dataset_wizard(z_bin_value, targets=np.zeros(num_tstats))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_rel_value = dsm(ds_rel_value)
    if ranked:
        res_rel_value = rankdata(res_rel_value)
    else:
        res_rel_value = res_rel_value.samples.reshape(-1)
        
    #day
    day = tstat_info[['Day']].to_numpy()
    ds_day = dataset_wizard(day, targets=np.zeros(num_tstats))
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    res_day = dsm(ds_day)
    if ranked:
        res_day = rankdata(res_day)
    else:
        res_day = res_day.samples.reshape(-1)
        
    if remove_within_day:
        if ranked:
            day_values = np.unique(res_day)
            if btwn_or_within == 1:
                high_rank = np.max(day_values)
            elif btwn_or_within == 0:
                high_rank = np.min(day_values)
            btwn_day_inds = np.where(res_day == high_rank)[0]
        else:
            btwn_day_inds = np.where(res_day == btwn_or_within)[0]
        res_abs_value = res_abs_value[btwn_day_inds]
        res_rel_value = res_rel_value[btwn_day_inds]
    
    target_dsms = {}    
    target_dsms['abs_value'] = res_abs_value
    target_dsms['rel_value'] = res_rel_value
       
    mask_loop = ['ACC_pre','ACC_sup',
                 'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
                 'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
                 'Calcarine', 'Fusiform']
    
    mask_names = ['rACC','dACC',
                  'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
                  'dmPFC','dlPFC','MFG','IFG',
                  'V1','Fusiform']
    
    fmri_dsm_list = []
    for mask in mask_loop:
        mask_name = bundle_path+'/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
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
        
        if remove_within_day:
            fmri_dsm = fmri_dsm[btwn_day_inds]
        
        fmri_dsm_list.append(fmri_dsm)
        
    model_dsm_names = ['rel_value','abs_value']
    fmri2model_matrix = np.zeros([len(mask_loop),2])
    for mask_num in range(len(mask_loop)):
        for model_num in range(len(model_dsm_names)):
            temp_fmri = fmri_dsm_list[mask_num]
            temp_model = target_dsms[model_dsm_names[model_num]]
            temp_correl = pearsonr(temp_fmri, temp_model)[0]
            fmri2model_matrix[mask_num,model_num] = temp_correl
        
    subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=model_dsm_names)
    subj_df['ROI'] = subj_df.index
    subj_df = subj_df.rename(columns={"rel_value": "Relative Value", "abs_value": "Absolute Value"})
    
    tidy_df = subj_df.melt(id_vars='ROI',value_vars=['Relative Value','Absolute Value'])
    
    sns.barplot(x="ROI", y='value', hue='variable', data=tidy_df)
    plt.title('RSA Value Sub'+subj)
    plt.xticks(np.arange(len(mask_loop)),(mask_names),rotation=90)
    plt.ylabel('RSA Correlation')
    plt.legend(bbox_to_anchor=(1.3, 1),borderaxespad=0)
    plt.show()
        
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        tidy_df['Subj'] = [subj for i in range(len(tidy_df))]
        tidy_df.to_csv(save_path+'rsa_abs_rel_bin_btwnday.csv')