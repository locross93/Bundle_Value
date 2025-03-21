#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 11:44:30 2021

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
import random
import pandas as pd
import seaborn as sns

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
#subj_list = ['101','102','103']
subj_list = ['104']

analysis_name = 'cross_decoding_abs_value'

relative_value = False

save = False

subsample = False

#'sitem2bundle' or 'bundle2sitem'
#train_test_split = 'sitem2bundle'

#which conditions to include in the analysis - ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']
conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']   
#conditions = ['Food bundle','Trinket bundle','Mixed bundle']

mask_loop = ['ACC_pre','ACC_sub','ACC_sup','Amygdala','Caudate','Cingulate_Mid','Cingulate_Post','Cuneus',
	'Frontal_Inf_Oper','Frontal_Inf_Orb_2','Frontal_Inf_Tri','Frontal_Med_Orb','Frontal_Mid_2','Frontal_Sup_2','Frontal_Sup_Medial','Fusiform',
	'Hippocampus','Insula','N_Acc','OFCant','OFClat','OFCmed','OFCpost','Paracentral_Lobule','Precentral','Precuneus','Putamen','Supp_Motor_Area']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

mask_loop = ['Frontal_Mid_2']

mask_names = ['MFG']

###SCRIPT ARGUMENTS END

data_list = []
for subj in subj_list:
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    #brain_mask = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii.gz'
    #mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/pfc_full_bin.nii.gz'
    
    #fds = mvpa_utils.make_targets(subj, glm_ds_file, mask_name, conditions)
    fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value)
    
    #zscore targets
#    if not relative_value:
#        fds.targets = scipy.stats.zscore(fds.targets)
        
    trial_categ = fds.sa.trial_categ
    sitem_inds = np.where(trial_categ < 3)[0]
    bundle_inds = np.where(trial_categ > 2)[0]
    #if subsample, take random bundle inds to make the same number of bundle and single item inds
    if subsample:
        num_sitem_trials = len(sitem_inds)
        bundle_inds = np.array(random.sample(bundle_inds, num_sitem_trials))
    
    #define model
    alp=1e4
    #alp=30
    sk_ridge = linear_model.Ridge(alpha=alp)
    #sk_ridge = PLSRegression(n_components=50)
    r_scorer = make_scorer(get_correlation)
    run_num = 15
    gkf = GroupKFold(n_splits=run_num)
    
    mask_count = 0
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
        
        cv_score_s2s = np.zeros([run_num])
        cv_score_s2b = np.zeros([run_num])
        cv_score_b2b = np.zeros([run_num])
        cv_score_b2s = np.zeros([run_num])
        
        s2s_pred_item_vals = np.array([])
        s2s_real_item_vals = np.array([])
        s2b_pred_bundle_vals = np.array([])
        s2b_real_bundle_vals = np.array([])
        cv_count = 0
        for train, test in gkf.split(X, y, groups=cv_groups):
            #train within category test within and across category
            train_s = np.intersect1d(train, sitem_inds)
            test_s = np.intersect1d(test, sitem_inds)
            train_b = np.intersect1d(train, bundle_inds)
            test_b = np.intersect1d(test, bundle_inds)
            
            #train on single item
            sk_ridge.fit(X[train_s,:],y[train_s])
            y_preds_in = sk_ridge.predict(X[test_s,:])
            cv_score_s2s[cv_count] = get_correlation(y_preds_in,y[test_s])
            #cv_score_s2s[cv_count] = r2_score(y_preds_in,y[test_s])
            s2s_pred_item_vals = np.append(s2s_pred_item_vals, y_preds_in)
            s2s_real_item_vals = np.append(s2s_real_item_vals, y[test_s])
            
            # test on bundles
            y_preds_out = sk_ridge.predict(X[test_b,:])
            cv_score_s2b[cv_count] = get_correlation(y_preds_out,y[test_b])
            #cv_score_s2b[cv_count] = r2_score(y_preds_out,y[test_b])
            s2b_pred_bundle_vals = np.append(s2b_pred_bundle_vals, y_preds_out)
            s2b_real_bundle_vals = np.append(s2b_real_bundle_vals, y[test_b])
            
            #train on bundle
            sk_ridge.fit(X[train_b,:],y[train_b])
            y_preds_in = sk_ridge.predict(X[test_b,:])
            cv_score_b2b[cv_count] = get_correlation(y_preds_in,y[test_b])
            #cv_score_b2b[cv_count] = r2_score(y_preds_in,y[test_b])
            y_preds_out = sk_ridge.predict(X[test_s,:])
            cv_score_b2s[cv_count] = get_correlation(y_preds_out,y[test_s])
            #cv_score_b2s[cv_count] = r2_score(y_preds_out,y[test_s])
            
            cv_count+=1
          
        mask_label = mask_names[mask_count]
        print mask_label
        print 'In Category s2s:',np.mean(cv_score_s2s)
        print 'Cross Category s2b:',np.mean(cv_score_s2b)
        print 'In Category b2b:',np.mean(cv_score_b2b)
        print 'Cross Category b2s:',np.mean(cv_score_b2s)
        print '\n'
        
        mask_scores = [subj, mask_label, np.mean(cv_score_s2s), np.mean(cv_score_s2b), np.mean(cv_score_b2b), np.mean(cv_score_b2s)]
        data_list.append(mask_scores)
        
        mask_count += 1
        
#df_data = pd.DataFrame(data_list, columns = ['Subj', 'Mask','S2S','S2B','B2B','B2S']) 
#cols = [2,3,4,5]
#df_plot = pd.melt(df_data, id_vars=["Subj","Mask"], var_name="Decoding Type", value_name="Accuracy")
#
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=68)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding All Subjects')

#plt.scatter(s2s_real_item_vals, s2s_pred_item_vals)
#plt.plot(np.arange(10), np.arange(10))
#
#unique_item_vals = np.unique(s2s_real_item_vals)
#avg_pred_svals = np.zeros(len(unique_item_vals))
#for i,sval in enumerate(unique_item_vals):
#    match_inds = np.where(s2s_real_item_vals == sval)[0]
#    avg_pred_svals[i] = np.mean(s2s_pred_item_vals[match_inds])
#    
#plt.scatter(unique_item_vals, avg_pred_svals)
#plt.plot(np.arange(10), np.arange(10))
#    
##plt.scatter(real_bundle_vals, pred_bundle_vals)
##plt.plot(np.arange(20), np.arange(20))
##
##unique_bundle_vals = np.unique(real_bundle_vals)
##avg_pred_vals = np.zeros(len(unique_bundle_vals))
##for i,bval in enumerate(unique_bundle_vals):
##    match_inds = np.where(real_bundle_vals == bval)[0]
##    avg_pred_vals[i] = np.mean(pred_bundle_vals[match_inds])
##    
##plt.scatter(unique_bundle_vals, avg_pred_vals)
##plt.plot(np.arange(20), np.arange(20))
#
##alp=3
#alphas = [1e3, 2.5e3, 5e3, 1e4, 2.5e4, 5e4]
#cv_scores = np.zeros(len(alphas))
#for i,alp in enumerate(alphas):
#    sk_ridge = linear_model.Ridge(alpha=10*alp, fit_intercept=True)
#    cv_scores[i] = np.mean(cross_val_score(sk_ridge, X, y, groups=cv_groups, cv=15))
#
##for sweeping
#plt.plot(alphas,cv_scores,'ro-', label="Crossval Fit")
#plt.xlabel('Regularization Alpha')
#plt.ylabel('R Squared')
#plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)