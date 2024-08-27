#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 17:01:33 2021

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
from sklearn import neighbors
from sklearn.model_selection import cross_val_score

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
ranked = True
remove_within_day = True

fmri2model_matrix_allsubs = []
for subj in subj_list:
    print subj
    
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=False)
    
    #load model dsms
    target_dsms = mvpa_utils.get_target_dsm(subj, fds, conditions, square_dsm_bool, ranked)
    
    item_list = np.genfromtxt('/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/item_list.txt')
    single_item_inds = np.where(item_list[:,1] == -1)[0]
    item_list_1item = item_list[single_item_inds,0].astype(int)
    
    assert(fds.shape[0] == len(item_list_1item))
    
    fds.sa.item = item_list_1item
    
    item_unique, item_counts = np.unique(item_list_1item, return_counts=True)
    #take classes that repeat more than 5 times
    all_day_items = item_unique[np.where(item_counts > 5)[0]]
    all_day_item_inds = np.where([item in all_day_items for item in item_list_1item])[0]
    
    fds_adi = fds[all_day_item_inds,:]
    fds_adi.sa.item = fds.sa.item[all_day_item_inds]
    
    class_scores = {}
    class_scores_perm = {}
    for mask in mask_loop:
        mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
        #brain_mask = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
        masked = fmri_dataset(mask_name, mask=brain_mask)
        reshape_masked=masked.samples.reshape(fds_adi.shape[1])
        reshape_masked=reshape_masked.astype(bool)
        mask_map = mask_mapper(mask=reshape_masked)
        
        mask_slice = mask_map[1].slicearg
        mask_inds = np.where(mask_slice == 1)[0]
        
        fds_mask = fds_adi[:,mask_inds]
        
        X = fds_mask.samples
        y = fds_adi.sa.item
        rand_inds = np.random.choice(y.shape[0], y.shape[0], replace=False)
        y_shuffle = y[rand_inds]
        
        from sklearn.metrics import log_loss
        from sklearn.model_selection import GroupKFold
        
        #logit_scorer = make_scorer(log_loss)
    
#        n_neighbors = 10
#        clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
#        clf = LogisticRegression(C = 1, solver='lbfgs', max_iter=1000, multi_class='multinomial')
#        #clf.fit(X, y)
#        class_scores[mask] = cross_val_score(clf,X,y,groups=fds_adi.chunks,scoring=logit_scorer, cv=15)
#        #class_scores_perm[mask] = cross_val_score(clf,X,y_shuffle,groups=fds_adi.chunks,cv=15)
        
        gkf = GroupKFold(n_splits=15)
        clf = LogisticRegression(C = 1, solver='lbfgs', max_iter=1000, multi_class='multinomial')
        temp_loss = []
        temp_acc = []
        cv_count = -1
        for train, test in gkf.split(X, y, groups=fds_adi.chunks):
            if np.array_equal(np.unique(y[train]),np.unique(y[test])):
                cv_count+=1
                clf.fit(X[train,:],y[train])
                #temp_scores[cv_count] = logit_scorer(y[test],clf.predict_proba(X[test,:]))
                loss = log_loss(y[test],clf.predict_proba(X[test,:]))
                acc = clf.score(X[test,:],y[test])
                temp_loss.append(loss)
                temp_acc.append(acc)
        class_scores[mask] = [np.mean(temp_loss), np.mean(temp_acc)]
            
    for mask in mask_loop:
        print mask,class_scores[mask]
    print '\n'
    