#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 13 20:16:03 2021

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
import statsmodels.api as sm
import pandas as pd

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']

#conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
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

relative_value = True
square_dsm_bool = False
ranked = False
remove_within_day = True
save = True

all_sub_fmri_dsms_list = []
all_sub_model_dsms_list = []
for subj in subj_list:
    print subj
    
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel2/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    #load model dsms
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
    single_item_inds = np.where(item_list[:,1] == -1)[0]
    item_list_1item = item_list[single_item_inds,0].astype(int)
    
    assert(fds.shape[0] == len(item_list_1item))
    
    vgg1_food = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block1_pool_feats_food.npy')
    vgg2_food = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block2_pool_feats_food.npy')
    vgg3_food = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block3_pool_feats_food.npy')
    vgg4_food = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block4_pool_feats_food.npy')
    vgg5_food = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block5_pool_feats_food.npy')
    vgg1_trinket = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block1_pool_feats_trinket.npy')
    vgg2_trinket = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block2_pool_feats_trinket.npy')
    vgg3_trinket = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block3_pool_feats_trinket.npy')
    vgg4_trinket = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block4_pool_feats_trinket.npy')
    vgg5_trinket = np.load(bundle_path+'mvpa/stimulus_features/vgg16/block5_pool_feats_trinket.npy')
    
    images_list = []
    vgg1_list = []
    vgg2_list = []
    vgg3_list = []
    vgg4_list = []
    vgg5_list = []
    for item in item_list_1item:
        if item < 100:
            item_str = str(item)
            image = Image.open('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/data/imgs_food/item_'+item_str+'.jpg')
            item_num = item - 1
            vgg1_temp = vgg1_food[item_num,:,:,:].reshape(-1)
            vgg2_temp = vgg2_food[item_num,:,:,:].reshape(-1)
            vgg3_temp = vgg3_food[item_num,:,:,:].reshape(-1)
            vgg4_temp = vgg4_food[item_num,:,:,:].reshape(-1)
            vgg5_temp = vgg5_food[item_num,:,:,:].reshape(-1)
        elif item >= 100:
            item_str = str(item - 100)
            image = Image.open('/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/data/imgs_trinkets/item_'+item_str+'.jpg')
            item_num = item - 101
            vgg1_temp = vgg1_trinket[item_num,:,:,:].reshape(-1)
            vgg2_temp = vgg2_trinket[item_num,:,:,:].reshape(-1)
            vgg3_temp = vgg3_trinket[item_num,:,:,:].reshape(-1)
            vgg4_temp = vgg4_trinket[item_num,:,:,:].reshape(-1)
            vgg5_temp = vgg5_trinket[item_num,:,:,:].reshape(-1)
        #resize to 2400,1800,3 then flatten and add to list
        image_resize = image.resize((240,180))
        images_list.append(np.array(image_resize).reshape(-1))
        vgg1_list.append(vgg1_temp)
        vgg2_list.append(vgg2_temp)
        vgg3_list.append(vgg3_temp)
        vgg4_list.append(vgg4_temp)
        vgg5_list.append(vgg5_temp)
    images_array = np.vstack(images_list)
    vgg1_array = np.vstack(vgg1_list)
    vgg2_array = np.vstack(vgg2_list)
    vgg3_array = np.vstack(vgg3_list)
    vgg4_array = np.vstack(vgg4_list)
    vgg5_array = np.vstack(vgg5_list)
    
    dataset_pix = dataset_wizard(images_array, targets=np.zeros(images_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_pix = dsm_func(dataset_pix)
    #just take samples to make lighter array
    if ranked:
        dsm_pix = rankdata(dsm_pix)
    else:
        dsm_pix = dsm_pix.samples.reshape(-1)
    target_dsms['pixels'] = dsm_pix
    
    ds_vgg1 = dataset_wizard(vgg1_array, targets=np.zeros(vgg1_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_vgg1 = dsm_func(ds_vgg1)
    #just take samples to make lighter array
    if ranked:
        dsm_vgg1  = rankdata(dsm_vgg1)
    else:
        dsm_vgg1 = dsm_vgg1.samples.reshape(-1)
    target_dsms['vgg1'] = dsm_vgg1
    
    ds_vgg2 = dataset_wizard(vgg2_array, targets=np.zeros(vgg2_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_vgg2 = dsm_func(ds_vgg2)
    #just take samples to make lighter array
    if ranked:
        dsm_vgg2  = rankdata(dsm_vgg2)
    else:
        dsm_vgg2 = dsm_vgg2.samples.reshape(-1)
    target_dsms['vgg2'] = dsm_vgg2
    
    ds_vgg3 = dataset_wizard(vgg3_array, targets=np.zeros(vgg3_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_vgg3 = dsm_func(ds_vgg3)
    #just take samples to make lighter array
    if ranked:
        dsm_vgg3  = rankdata(dsm_vgg3)
    else:
        dsm_vgg3 = dsm_vgg3.samples.reshape(-1)
    target_dsms['vgg3'] = dsm_vgg3
    
    ds_vgg4 = dataset_wizard(vgg4_array, targets=np.zeros(vgg4_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_vgg4 = dsm_func(ds_vgg4)
    #just take samples to make lighter array
    if ranked:
        dsm_vgg4  = rankdata(dsm_vgg4)
    else:
        dsm_vgg4 = dsm_vgg4.samples.reshape(-1)
    target_dsms['vgg4'] = dsm_vgg4
    
    ds_vgg5 = dataset_wizard(vgg5_array, targets=np.zeros(vgg5_array.shape[0]))   
    dsm_func = rsa.PDist(pairwise_metric='euclidean', square=False)
    dsm_vgg5 = dsm_func(ds_vgg5)
    #just take samples to make lighter array
    if ranked:
        dsm_vgg5  = rankdata(dsm_vgg5)
    else:
        dsm_vgg5 = dsm_vgg5.samples.reshape(-1)
    target_dsms['vgg5'] = dsm_vgg5
    
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
    model_dsm_names = ['value','stim_id','pixels','vgg1','vgg2','vgg3','vgg4','vgg5']
        
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
        subj_df.to_csv(save_path+'rsa_reg_vgg_scores2.csv')