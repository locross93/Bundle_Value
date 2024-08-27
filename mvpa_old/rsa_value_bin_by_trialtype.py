# -*- coding: utf-8 -*-
"""
Created on Sun May 16 13:52:32 2021

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

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['105','106']

square_dsm_bool = False
ranked = True
remove_within_day = True
#put 1 for between day, 0 for within day
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
    fds = fmri_dataset(samples=glm_ds_file, targets=avg_bin_value, mask=brain_mask)
    
    #index by trial category
    trial_cat = tstat_info[['Trial Type']].to_numpy()
    sitem_inds = np.where(trial_cat == 0)[0]
    bundle_inds = np.where(trial_cat == 1)[0]
    fds_sitem = fds[sitem_inds,:]
    fds_bundle = fds[bundle_inds,:]
    sitem_value = avg_bin_value[sitem_inds]
    bundle_value = avg_bin_value[bundle_inds]
    
    ds_svalue = dataset_wizard(sitem_value, targets=np.zeros(len(sitem_value)))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_svalue = dsm(ds_svalue)
    if ranked:
        res_svalue = rankdata(res_svalue)
    else:
        res_svalue = res_svalue.samples.reshape(-1)
        
    ds_bvalue = dataset_wizard(bundle_value, targets=np.zeros(len(bundle_value)))
    dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
    res_bvalue = dsm(ds_bvalue)
    if ranked:
        res_bvalue = rankdata(res_bvalue)
    else:
        res_bvalue = res_bvalue.samples.reshape(-1)
        
    #create dsms for day
    dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
    day = tstat_info[['Day']].to_numpy()
    
    sday = day[sitem_inds]
    ds_sday = dataset_wizard(sday, targets=np.zeros(len(sday)))
    res_sday = dsm(ds_sday)
    if ranked:
        res_sday = rankdata(res_sday)
    else:
        res_sday = res_sday.samples.reshape(-1)
        
    bday = day[bundle_inds]
    ds_bday = dataset_wizard(bday, targets=np.zeros(len(bday)))
    res_bday = dsm(ds_bday)
    if ranked:
        res_bday = rankdata(res_bday)
    else:
        res_bday = res_bday.samples.reshape(-1)
        
    if remove_within_day:
        if ranked:
            sday_values = np.unique(res_sday)
            bday_values = np.unique(res_bday)
            if btwn_or_within == 1:
                high_rank1 = np.max(sday_values)
                high_rank2 = np.max(bday_values)
            elif btwn_or_within == 0:
                high_rank1 = np.min(sday_values)
                high_rank2 = np.min(bday_values)
            btwn_sday_inds = np.where(res_sday == high_rank1)[0]
            btwn_bday_inds = np.where(res_bday == high_rank2)[0]
        else:
            btwn_sday_inds = np.where(res_sday == btwn_or_within)[0]
            btwn_bday_inds = np.where(res_bday == btwn_or_within)[0]
        res_svalue = res_svalue[btwn_sday_inds]
        res_bvalue = res_bvalue[btwn_bday_inds]
        
    mask_loop = ['ACC_pre','ACC_sup',
                 'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
                 'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
                 'Calcarine', 'Fusiform']
    
    mask_names = ['rACC','dACC',
                  'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
                  'dmPFC','dlPFC','MFG','IFG',
                  'V1','Fusiform']
    
    fmri_dsm_list_sitem = []
    fmri_dsm_list_bundle = []
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
        
        fds_mask_s = fds_sitem[:,mask_inds]
        fds_mask_b = fds_bundle[:,mask_inds]
        
        #dsm_func = rsa.PDist(pairwise_metric='Correlation', square=square_dsm_bool)
        dsm_func = rsa.PDist(pairwise_metric='Euclidean', square=square_dsm_bool)
        
        #add PCA on the FMRI data
        #pca = PCA(n_components=5)
        #pca_fds_mask = pca.fit_transform(fds_mask.samples)
        #dataset_fmri = dataset_wizard(pca_fds_mask, targets=np.zeros(len(pca_fds_mask)))
        #fmri_dsm = dsm_func(dataset_fmri)
        
        #no PCA
        fmri_dsm_sitem = dsm_func(fds_mask_s)
        fmri_dsm_bundle = dsm_func(fds_mask_b)
        
        #just take samples to make lighter array and rank
        if ranked:
            fmri_dsm_sitem = rankdata(fmri_dsm_sitem.samples)
            fmri_dsm_bundle = rankdata(fmri_dsm_bundle.samples)
        else:
            fmri_dsm_sitem = fmri_dsm_sitem.samples.reshape(-1)
            fmri_dsm_bundle = fmri_dsm_bundle.samples.reshape(-1)
            
        if remove_within_day:
            fmri_dsm_sitem = fmri_dsm_sitem[btwn_sday_inds]
            fmri_dsm_bundle = fmri_dsm_bundle[btwn_bday_inds]
        
        fmri_dsm_list_sitem.append(fmri_dsm_sitem)
        fmri_dsm_list_bundle.append(fmri_dsm_bundle)
        
    fmri2model_matrix = np.zeros([len(mask_loop),2])
    #sitem
    for mask_num in range(len(mask_loop)):
        temp_fmri = fmri_dsm_list_sitem[mask_num]
        temp_correl = pearsonr(temp_fmri, res_svalue)[0]
        fmri2model_matrix[mask_num,0] = temp_correl
        
    #bundle
    for mask_num in range(len(mask_loop)):
        temp_fmri = fmri_dsm_list_bundle[mask_num]
        temp_correl = pearsonr(temp_fmri, res_bvalue)[0]
        fmri2model_matrix[mask_num,1] = temp_correl 
      
    xaxis_labels = ['Single Item','Bundle']
    subj_df = pd.DataFrame(fmri2model_matrix, index=mask_names, columns=xaxis_labels)
    subj_df['ROI'] = subj_df.index
    
    tidy_df = subj_df.melt(id_vars='ROI',value_vars=['Single Item','Bundle'])
    
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
        tidy_df.to_csv(save_path+'rsa_val_bin_trialtype_btwnday_unrank.csv')