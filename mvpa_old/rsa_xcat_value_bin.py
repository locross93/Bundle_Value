# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 15:35:27 2021

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
subj_list = ['114']

mask_loop = ['ACC_pre','ACC_sup',
                 'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
                 'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
                 'Calcarine', 'Fusiform']
    
mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

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
    
    sitem_inds = np.where(trial_cat == 0)[0]
    bun_inds = np.where(trial_cat == 1)[0]
    
    fds = fmri_dataset(samples=glm_ds_file, targets=avg_bin_value, mask=brain_mask)

    sitem_unique_vals = np.unique(avg_bin_value[sitem_inds])
    bun_unique_vals = np.unique(avg_bin_value[bun_inds])

    fmri_dsm_list = []
    mask_count = -1
    for mask in mask_loop:
        mask_count+=1
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
        
        dsm_func = rsa.PDist(pairwise_metric='Correlation', square=True)
        #dsm_func = rsa.PDist(pairwise_metric='Euclidean', square=True)
        
        #add PCA on the FMRI data
        #pca = PCA(n_components=5)
        #pca_fds_mask = pca.fit_transform(fds_mask.samples)
        #dataset_fmri = dataset_wizard(pca_fds_mask, targets=np.zeros(len(pca_fds_mask)))
        #fmri_dsm = dsm_func(dataset_fmri)
        
        #no PCA
        fmri_dsm = dsm_func(fds_mask)
        #fmri_dsm = fmri_dsm.samples
        
        #just take samples to make lighter array and rank
    
    #    if ranked:
    #        fmri_dsm = rankdata(fmri_dsm.samples)
    #    else:
    #        fmri_dsm = fmri_dsm.samples.reshape(-1)
        
        cross_cat_similarity = np.zeros([len(sitem_unique_vals), len(bun_unique_vals)])
        
        for i,sval in enumerate(sitem_unique_vals):
            for j,bval in enumerate(bun_unique_vals):
                s_inds = sitem_inds[np.where(avg_bin_value[sitem_inds] == sval)[0]]
                b_inds = bun_inds[np.where(avg_bin_value[bun_inds] == bval)[0]]
                cc_mat = fmri_dsm[s_inds, b_inds]
                cross_cat_similarity[i,j] = np.mean(cc_mat)
        
        f = sns.heatmap(cross_cat_similarity, annot=True, annot_kws={"size": 7},  
                        xticklabels=bun_unique_vals, yticklabels=sitem_unique_vals, cbar_kws={'label': 'Dissimilarity'}) 
        mask_name = mask_names[mask_count]
        f.set_title(mask_name+' RSA Single Item vs Bundle Values', fontsize = 20)   
        plt.ylabel('Single Item Value', fontsize = 16)
        plt.xlabel('Bundle Value', fontsize = 16)
        plt.show()