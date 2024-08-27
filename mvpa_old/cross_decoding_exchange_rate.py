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
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
import random
import pandas as pd
import seaborn as sns

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

def test_diff_in_zeros(subj, mask_label, s2s_real_item_vals, s2s_pred_item_vals, s2b_real_bundle_vals, s2b_pred_bundle_vals, 
                          b2s_real_item_vals, b2s_pred_item_vals, b2b_real_bundle_vals, b2b_pred_bundle_vals):

    inds0_item = np.where(s2s_real_item_vals == 0)[0]
    inds0_bundle = np.where(s2b_real_bundle_vals == 0)[0]
    
    pred0s_item = s2s_pred_item_vals[inds0_item]
    pred0s_bundle = s2b_pred_bundle_vals[inds0_bundle]
    
    item0_mean = np.mean(pred0s_item)
    bundle0_mean = np.mean(pred0s_bundle)
    
    t_diff = stats.ttest_ind(pred0s_item, pred0s_bundle)
    
    data = np.concatenate([pred0s_item, pred0s_bundle])
    
    cat = ['Item' for i in range(len(pred0s_item))] + ['Bundle' for i in range(len(pred0s_bundle))]
    
    data_dict = {'Predicted WTP': data, 'Category': cat}
    
    df_data_strain = pd.DataFrame(data_dict)
    
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))
    fig.suptitle('CV Predictions for 0s '+mask_label+' Sub'+subj)
    
    ax = sns.boxplot(x="Category", y="Predicted WTP", data=df_data_strain, ax=axes[0])
    xlims = axes[0].get_xlim()
    ylims = axes[0].get_ylim()
    axes[0].text(xlims[1]*0.6, ylims[1]*0.95, 'Item Mean: %2.3f'%item0_mean, ha='center', va='center')
    axes[0].text(xlims[1]*0.6, ylims[1]*0.89, 'Bundle Mean: %2.3f'%bundle0_mean, ha='center', va='center')
    axes[0].text(xlims[1]*0.6, ylims[1]*0.84, 'T-test: %2.3f'%t_diff[0], ha='center', va='center')
    axes[0].set_title('Train on Single Item')
    
    inds0_item = np.where(b2s_real_item_vals == 0)[0]
    inds0_bundle = np.where(b2b_real_bundle_vals == 0)[0]
    
    pred0s_item = b2s_pred_item_vals[inds0_item]
    pred0s_bundle = b2b_pred_bundle_vals[inds0_bundle]
    
    item0_mean = np.mean(pred0s_item)
    bundle0_mean = np.mean(pred0s_bundle)
    
    t_diff = stats.ttest_ind(pred0s_item, pred0s_bundle)
    
    data = np.concatenate([pred0s_item, pred0s_bundle])
    
    cat = ['Item' for i in range(len(pred0s_item))] + ['Bundle' for i in range(len(pred0s_bundle))]
    
    data_dict = {'Predicted WTP': data, 'Category': cat}
    
    df_data_btrain = pd.DataFrame(data_dict)
    
    ax = sns.boxplot(x="Category", y="Predicted WTP", data=df_data_btrain, ax=axes[1])
    xlims = axes[1].get_xlim()
    ylims = axes[1].get_ylim()
    axes[1].text(xlims[1]*0.6, ylims[1]*0.95, 'Item Mean: %2.3f'%item0_mean, ha='center', va='center')
    axes[1].text(xlims[1]*0.6, ylims[1]*0.89, 'Bundle Mean: %2.3f'%bundle0_mean, ha='center', va='center')
    axes[1].text(xlims[1]*0.6, ylims[1]*0.84, 'T-test: %2.3f'%t_diff[0], ha='center', va='center')
    axes[1].set_title('Train on Bundle')
    plt.show()

###SCRIPT ARGUMENTS

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['104','105','106','108','109','110','111','112','113','114']
#subj_list = ['114']

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

mask_loop = ['Frontal_Med_Orb','Frontal_Sup_Medial']

mask_names = ['vmPFC','dmPFC']

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
    
    #temp median split control
#    trial_values = fds.targets
#    temp_sitem_inds = np.where(trial_categ < 3)[0]
#    median_sitem_val = np.median(trial_values[temp_sitem_inds])
#    median_sitem_val = 4
#    below_median_inds = np.where(trial_values <= median_sitem_val)[0]
#    above_median_inds = np.where(trial_values > median_sitem_val)[0]
#    sitem_inds = np.intersect1d(temp_sitem_inds, below_median_inds)
#    bundle_inds = np.intersect1d(temp_sitem_inds, above_median_inds)
    
    #if subsample, take random bundle inds to make the same number of bundle and single item inds
    if subsample:
        num_sitem_trials = len(sitem_inds)
        bundle_inds = np.array(random.sample(bundle_inds, num_sitem_trials))
    
    #define model
    alp=1e4
    #alp=30
    sk_ridge = linear_model.Ridge(alpha=alp)
    #sk_ridge = SVR(C=1.0, epsilon=0.2)
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
        b2s_pred_item_vals = np.array([])
        b2s_real_item_vals = np.array([])
        b2b_pred_bundle_vals = np.array([])
        b2b_real_bundle_vals = np.array([])
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
            
            # train on bundle
            sk_ridge.fit(X[train_b,:],y[train_b])
            y_preds_in = sk_ridge.predict(X[test_b,:])
            cv_score_b2b[cv_count] = get_correlation(y_preds_in,y[test_b])
            #cv_score_b2b[cv_count] = r2_score(y_preds_in,y[test_b])
            b2b_pred_bundle_vals = np.append(b2b_pred_bundle_vals, y_preds_in)
            b2b_real_bundle_vals = np.append(b2b_real_bundle_vals, y[test_b])
            
            # test on single item
            y_preds_out = sk_ridge.predict(X[test_s,:])
            cv_score_b2s[cv_count] = get_correlation(y_preds_out,y[test_s])
            #cv_score_b2s[cv_count] = r2_score(y_preds_out,y[test_s])
            b2s_pred_item_vals = np.append(b2s_pred_item_vals, y_preds_out)
            b2s_real_item_vals = np.append(b2s_real_item_vals, y[test_s])
            
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
        
#        if mask_count == 0:
#            # plot all values against themselves - absolute code
#            plt.scatter(s2b_real_bundle_vals, s2b_real_bundle_vals, c='b', label="Bundle")
#            plt.scatter(s2s_real_item_vals, s2s_real_item_vals-0.1, c='r', label="Single Item")
#            plt.xlabel('WTP Bid')
#            plt.ylabel('Predicted WTP Bid')
#            plt.title('Predictions of Absolute Code Sub'+subj)
#            plt.legend(bbox_to_anchor=(1.3, 1.03))
#            plt.show()
#            
#            # plot values vs their zscore within category
#            z_item_vals = scipy.stats.zscore(s2s_real_item_vals)
#            z_bundle_vals = scipy.stats.zscore(s2b_real_bundle_vals)
#            plt.scatter(s2b_real_bundle_vals, z_bundle_vals, c='b', label="Bundle")
#            plt.scatter(s2s_real_item_vals, z_item_vals, c='r', label="Single Item")
#            plt.xlabel('WTP Bid')
#            plt.ylabel('Predicted WTP Bid')
#            plt.title('Predictions of Relative Code Sub'+subj)
#            plt.legend(bbox_to_anchor=(1.3, 1.03))
#            plt.show()
            
        test_diff_in_zeros(subj, mask_label, s2s_real_item_vals, s2s_pred_item_vals, s2b_real_bundle_vals, s2b_pred_bundle_vals, 
                          b2s_real_item_vals, b2s_pred_item_vals, b2b_real_bundle_vals, b2b_pred_bundle_vals)
        
        mask_count += 1