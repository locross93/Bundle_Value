#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 15:51:24 2021

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
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['110']

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

#mask_loop = ['ACC_pre','ACC_sup',
#             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
#             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri',
#             'Calcarine', 'Fusiform']
#
#mask_names = ['rACC','dACC',
#              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
#              'dmPFC','dlPFC','MFG','IFG',
#              'V1','Fusiform']

square_dsm_bool = False
ranked = False
remove_within_day = True

fmri2model_matrix_allsubs = []
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
    
    #compare model dsms in heatmap
#    model_dsm_all_list = [dsm_pix, dsm_vgg1, dsm_vgg2, dsm_vgg3, dsm_vgg4, dsm_vgg5]
#    model_dsm_all = np.vstack(model_dsm_all_list)
#    model_correl = np.corrcoef(model_dsm_all)
#    #make heatmap
#    mask_mat = np.zeros_like(model_correl)
#    mask_mat[np.triu_indices_from(mask_mat)] = True
#    model_names = ['Pixels', 'VGG Pool 1','VGG Pool 2','VGG Pool 3','VGG Pool 4','VGG Pool 5']
#    f = sns.heatmap(model_correl, annot=True, annot_kws={"size": 7}, mask=mask_mat, 
#                    xticklabels=model_names, yticklabels=model_names, vmin=0.0, vmax=1.0, cbar_kws={'label': 'Correlation (r)'})
#    plt.xticks(rotation=90)
#    plt.show()
    
    #save dsms and load in another file to compare to fmri data to reduce memory load
    #h5save(target_dsms, bundle_path+'mvpa/presaved_data/sub'+subj+'/target_dsms_vgg16')
    
    if remove_within_day:
        res_day = target_dsms['day']
        if ranked:
            day_values = np.unique(res_day)
            high_rank = np.max(day_values)
            btwn_day_inds = np.where(res_day == high_rank)[0]
        else:
            btwn_day_inds = np.where(res_day == 1)[0]
    
    model_dsm_names = ['pixels','vgg1','vgg2','vgg3','vgg4','vgg5']
        
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
        #pca = PCA(n_components=100)
        #pca_fds_mask = pca.fit_transform(fds_mask.samples)
        #dataset_fmri = dataset_wizard(pca_fds_mask, targets=np.zeros(len(pca_fds_mask)))
        
        #dsm_func = rsa.PDist(pairwise_metric='Correlation', square=square_dsm_bool)
        dsm_func = rsa.PDist(pairwise_metric='Euclidean', square=square_dsm_bool)
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
fmri2model_avg /= len(subj_list)

mask_mat = np.zeros_like(fmri2model_avg)
mask_mat[np.triu_indices_from(mask_mat)] = True   
sns.set(rc={'figure.figsize':(10,10)})
f = sns.heatmap(fmri2model_avg, annot=True, annot_kws={"size": 7},  
                xticklabels=model_dsm_names, yticklabels=mask_loop, vmin=0.0, vmax=0.5, cbar_kws={'label': 'Spearman Correlation ($\\rho$)'}) 
f.set_title('RSA By ROI', fontsize = 20)   
plt.show()
    