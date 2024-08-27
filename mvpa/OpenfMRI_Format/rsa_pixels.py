#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 14:44:53 2021

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
from PIL import Image

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['114']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item','Trinket item']

mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    'Insula','OFCant','OFClat','OFCmed','OFCpost',
    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate

#mask_loop = ['ACC_pre','Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Med_Orb','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform','OFClat','OFCmed']

#mask_loop = ['Frontal_Inf_Oper']

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

fmri2model_matrix_allsubs = []
for subj in subj_list:
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    #load model dsms
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
    single_item_inds = np.where(item_list[:,1] == -1)[0]
    item_list_1item = item_list[single_item_inds,0].astype(int)
    
    assert(fds.shape[0] == len(item_list_1item))
    
    images_list = []
    for item in item_list_1item:
        if item < 100:
            item_str = str(item)
            image = Image.open('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/data/imgs_food/item_'+item_str+'.jpg')
        elif item >= 100:
            item_str = str(item - 100)
            image = Image.open('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/data/imgs_trinkets/item_'+item_str+'.jpg')
        #resize to 2400,1800,3 then flatten and add to list
        image_resize = image.resize((240,180))
        images_list.append(np.array(image_resize).reshape(-1))
    images_array = np.vstack(images_list)
    
    dataset_pix = dataset_wizard(images_array, targets=np.zeros(images_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='correlation', square=False)
    dsm_pix = dsm_func(dataset_pix)
    #just take samples to make lighter array
    if ranked:
        dsm_pix = rankdata(dsm_pix)
    else:
        dsm_pix = dsm_pix.samples.reshape(-1)
    target_dsms['pixels'] = dsm_pix
    
    if remove_within_day:
        res_day = target_dsms['day']
        if ranked:
            day_values = np.unique(res_day)
            high_rank = np.max(day_values)
            btwn_day_inds = np.where(res_day == high_rank)[0]
        else:
            btwn_day_inds = np.where(res_day == 1)[0]
        
    model_dsm_names = ['pixels','value','fvalue', 'tvalue']
        
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
    sns.set(rc={'figure.figsize':(10,10)})
    f = sns.heatmap(fmri2model_matrix, annot=True, annot_kws={"size": 7},  
                    xticklabels=model_dsm_names, yticklabels=mask_loop, vmin=0.0, vmax=0.05, cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
    
    fmri2model_matrix_allsubs.append(fmri2model_matrix)
    
fmri2model_avg = np.zeros([len(mask_loop), len(model_dsm_names)])
for mat in fmri2model_matrix_allsubs:
    fmri2model_avg+=mat
fmri2model_avg /= 9

mask_mat = np.zeros_like(fmri2model_avg)
mask_mat[np.triu_indices_from(mask_mat)] = True   
sns.set(rc={'figure.figsize':(10,10)})
f = sns.heatmap(fmri2model_avg, annot=True, annot_kws={"size": 7},  
                xticklabels=model_dsm_names[:5], yticklabels=mask_names, vmin=0.0, vmax=0.05, cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}) 
f.set_title('RSA By ROI', fontsize = 20)   
plt.show()