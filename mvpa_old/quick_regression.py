#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 13:06:06 2020

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
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103']
subj_list = ['110']

analysis_name = 'rel_value'

save = False

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
#conditions = ['Food bundle','Trinket bundle','Mixed bundle']
#conditions = ['Food item', 'Trinket item']
#conditions = ['Food item']

#mask_loop = ['sup_frontal_gyr', 'acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

###SCRIPT ARGUMENTS END

for subj in subj_list:
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii.gz'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/pfc_full_bin.nii.gz'
    
    #fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=True)
    
    #zscore targets
    #fds.targets = scipy.stats.zscore(fds.targets)
    
    #define model
    alp=2
    sk_ridge = linear_model.Ridge(alpha=10*alp)
    #sk_ridge = PLSRegression(n_components=50)
    r_scorer = make_scorer(get_correlation)
    run_num = 15
    gkf = GroupKFold(n_splits=run_num)
    
    score_dict = {}
    
    for mask in mask_loop: 
#        if int(subj) > 103:
#            mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/lowres/'+mask+'.nii.gz'
#        else:
#            mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
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
        #pca = PCA(n_components=50)
        #X = pca.fit_transform(fds_mask.samples)
        
        X = fds_mask.samples
        y = fds_mask.targets
        cv_groups = fds_mask.chunks
        
        #cv_score_sk = cross_val_score(sk_ridge,X,y,groups=cv_groups,scoring=r_scorer,cv=run_num)
        #cv_score_sk = cross_val_score(sk_ridge,X,y,groups=cv_groups,cv=run_num)
        
        cv_score_sk = np.zeros([run_num])
        cv_count = -1
        y_preds_list = []
        y_test_list = []
        for train, test in gkf.split(X, y, groups=cv_groups):
            #print 'Train mean:',np.mean(y[train])
            #print 'Test mean:',np.mean(y[test])
            
            sk_ridge.fit(X[train,:],y[train])
            y_preds = sk_ridge.predict(X[test,:])
            cv_score_sk[cv_count] = get_correlation(y_preds,y[test])
            #cv_score_sk[cv_count] = r2_score(y[test],y_preds)
            #y_preds = sk_ridge.predict(X[train,:])
            #cv_score_sk[cv_count] = r2_score(y[train],y_preds)
            cv_count+=1
            y_preds_list.append(y_preds)
            y_test_list.append(y[test])
            
        print mask,np.mean(cv_score_sk)
        print '\n'
        
        score_dict[mask] = np.mean(cv_score_sk)
        
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        np.save(save_path+'/aal_reg_scores_'+analysis_name+'.npy', score_dict)

y_preds_all = np.concatenate(y_preds_list)
y_test_all = np.concatenate(y_test_list)      
plt.scatter(y_test_all, y_preds_all)