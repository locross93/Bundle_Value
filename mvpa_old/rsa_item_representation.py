#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  9 12:52:57 2021

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
from scipy.spatial.distance import cdist

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

#subj_list = ['104','105','107','108','109','110','111','113','114']
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['104']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item']

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

mask_loop = ['Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial',
    'Insula','OFCant','OFClat','OFCmed','OFCpost',
    'Paracentral_Lobule','Precentral','Supp_Motor_Area','ACC_pre','ACC_sub','ACC_sup', #motor, frontal
    'Calcarine', 'Lingual','Occipital_Inf','Occipital_Mid','Occipital_Sup','Fusiform','Temporal_Inf','Temporal_Mid','Temporal_Pole_Mid','Temporal_Pole_Sup','Temporal_Sup', #visual areas
    'Amygdala','Caudate','Putamen','N_Acc','Hippocampus','ParaHippocampal', #subcortical areas
    'Cingulate_Mid','Cingulate_Post','Cuneus','Precuneus','Parietal_Inf','Parietal_Sup','Postcentral','SupraMarginal','Angular'] #parietal, cingulate
             
mask_names = mask_loop

relative_value = True
square_dsm_bool = False
ranked = False
remove_within_day = True
save = False
rank_later = True

for subj in subj_list:
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
    
    chunks = fds.chunks
    
    item_list = np.genfromtxt(bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
    
    ind_item_trials = np.where(item_list[:,1] == -1)[0]
    
    items_ii = item_list[ind_item_trials,0]
    unique_items, item_counts = np.unique(items_ii, return_counts=True)
    all_day_items = unique_items[np.where(item_counts == 15)[0]]
    
    df_scores = pd.DataFrame(columns = ['Mask','Same Item Diss.','Diff Item Diss.'])
    for mask in mask_loop:
        print mask
        mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
        masked = fmri_dataset(mask_name, mask=brain_mask)
        reshape_masked=masked.samples.reshape(fds.shape[1])
        reshape_masked=reshape_masked.astype(bool)
        mask_map = mask_mapper(mask=reshape_masked)
        mask_slice = mask_map[1].slicearg
        mask_inds = np.where(mask_slice == 1)[0]
        fds_mask = fds[:,mask_inds]
        
        diag_diss_cv = []
        nondiag_diss_cv = []
        for cv in range(15):
            mask_item_reps_in = []
            mask_item_reps_out = []
            for item in all_day_items:
                item_inds = np.where(items_ii == item)[0]
                #delete left out index in left out run
                item_inds_in = np.delete(item_inds, cv)
                item_pattern_in = np.mean(fds_mask.samples[item_inds_in,:], axis=0)
                mask_item_reps_in.append(item_pattern_in)
                #add left out run pattern to a separate list
                item_pattern_out = fds_mask.samples[item_inds[cv],:]
                mask_item_reps_out.append(item_pattern_out)
            mask_item_reps_in = np.vstack(mask_item_reps_in)
            mask_item_reps_out = np.vstack(mask_item_reps_out)
            item_pattern_dsm = cdist(mask_item_reps_in, mask_item_reps_out, 'euclidean')
            
            diag_avg = np.mean(item_pattern_dsm.diagonal())
            nondiag_avg = np.mean(item_pattern_dsm[~np.eye(item_pattern_dsm.shape[0],dtype=bool)])
            diag_diss_cv.append(diag_avg)
            nondiag_diss_cv.append(nondiag_avg)
        diag_diss = np.mean(diag_diss_cv)
        nondiag_diss = np.mean(nondiag_diss_cv)
        d = {'Mask': mask,'Same Item Diss.':diag_diss, 'Diff Item Diss.': nondiag_diss}
        df_scores = df_scores.append(d, ignore_index=True)
            
    
    