# -*- coding: utf-8 -*-
"""
Created on Fri May 14 18:52:58 2021

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

relative_value = True
square_dsm_bool = False
ranked = True
remove_within_day = True
save = False

subj = '104'

mvpa_prep_path = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/mvpa_value_bins/'
#which ds to use and which mask to use
glm_ds_file = mvpa_prep_path+'tstat_all_values_4D.nii'
#glm_ds_file = mvpa_prep_path+'tstat_all_trials_4D.nii.gz'
brain_mask = mvpa_prep_path+'mask.nii'

tstat_info = pd.read_csv(mvpa_prep_path+'tstat_table.csv')
num_tstats = len(tstat_info)
avg_bin_value = np.mean(tstat_info[['Bin Min', 'Bin Max']].to_numpy(), axis=1)
fds = fmri_dataset(samples=glm_ds_file, targets=avg_bin_value, mask=brain_mask)

ds_value = dataset_wizard(avg_bin_value, targets=np.zeros(num_tstats))
dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
res_value = dsm(ds_value)
if ranked:
    res_value = rankdata(res_value)
else:
    res_value = res_value.samples.reshape(-1)
    
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
        high_rank = np.max(day_values)
        btwn_day_inds = np.where(res_day == high_rank)[0]
    else:
        btwn_day_inds = np.where(res_day == 1)[0]
    res_value = res_value[btwn_day_inds]
    
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
    
fmri2model_matrix = np.zeros([len(mask_loop)])
for mask_num in range(len(mask_loop)):
    temp_fmri = fmri_dsm_list[mask_num]
    temp_correl = pearsonr(temp_fmri, res_value)[0]
    fmri2model_matrix[mask_num] = temp_correl
    
plt.bar(np.arange(len(mask_loop)), fmri2model_matrix)
plt.xticks(np.arange(len(mask_loop)),(mask_names),rotation=90)
plt.title('RSA Value Sub'+subj)
plt.ylabel('RSA Correlation')