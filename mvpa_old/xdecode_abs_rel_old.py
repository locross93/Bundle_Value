#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 13:30:25 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
import os
#os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
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
import seaborn as sns
import pandas as pd

def get_correlation(y, y_pred):
    correlation = pearsonr(y, y_pred)[0]
    
    return correlation

###SCRIPT ARGUMENTS

#bundle_path = '/Users/logancross/Documents/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['101','102','103','104','105','106','107','108','109','110','111','113','114']
#subj_list = ['101','102','103']
subj_list = ['110']

analysis_name = 'xdecoding_abs_vs_rel_value'

relative_value = False

save = False
save_group = False

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
    alp=3
    sk_ridge = linear_model.Ridge(alpha=10*alp)
    #sk_ridge = PLSRegression(n_components=50)
    r_scorer = make_scorer(get_correlation)
    run_num = 15
    gkf = GroupKFold(n_splits=run_num)
    
    mask_count = 0
    subj_scores = {}
    for mask in mask_loop: 
        print mask
#        if int(subj) > 103:
#            mask_name = '/Users/logancross/Documents/Bundle_Value/fmri/CIT_masks/lowres/'+mask+'.nii.gz'
#        else:
#            mask_name = '/Users/logancross/Documents/Bundle_Value/mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/'+mask+'.nii.gz'
        #mask_name = bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/masks/pfc_mask.nii.gz'
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
        
        abs_value = fds_mask.targets
        rel_value = np.zeros([len(abs_value)])
        zitem_values = scipy.stats.zscore(abs_value[sitem_inds])
        rel_value[sitem_inds] = zitem_values
        zbundle_values = scipy.stats.zscore(abs_value[bundle_inds])
        rel_value[bundle_inds] = zbundle_values
        cv_groups = fds_mask.chunks
        
        cv_score_s2abs = np.zeros([run_num])
        cv_score_s2rel = np.zeros([run_num])
        cv_score_b2abs = np.zeros([run_num])
        cv_score_b2rel = np.zeros([run_num])
        cv_count = -1
        for train, test in gkf.split(X, abs_value, groups=cv_groups):
            #train within category test within and across category
            train_s = np.intersect1d(train, sitem_inds)
            test_s = np.intersect1d(test, sitem_inds)
            train_b = np.intersect1d(train, bundle_inds)
            test_b = np.intersect1d(test, bundle_inds)
            
            #train on single item
            sk_ridge.fit(X[train_s,:],abs_value[train_s])
            y_preds_test = sk_ridge.predict(X[test,:])
            cv_score_s2abs[cv_count] = get_correlation(y_preds_test,abs_value[test])
            cv_score_s2rel[cv_count] = get_correlation(y_preds_test,rel_value[test])
            
            #train on bundle
            sk_ridge.fit(X[train_b,:],abs_value[train_b])
            y_preds_test = sk_ridge.predict(X[test,:])
            cv_score_b2abs[cv_count] = get_correlation(y_preds_test,abs_value[test])
            cv_score_b2rel[cv_count] = get_correlation(y_preds_test,rel_value[test])
            
            cv_count+=1
          
        mask_label = mask_names[mask_count]
#        print mask_label
#        print 'Train single, absolute value:',np.mean(cv_score_s2abs)
#        print 'Train single, relative value:',np.mean(cv_score_s2rel)
#        print 'Train bundle, absolute value:',np.mean(cv_score_b2abs)
#        print 'Train bundle, relative value:',np.mean(cv_score_b2rel)
#        print '\n'
        
        mask_scores = [subj, mask_label, np.mean(cv_score_s2abs), np.mean(cv_score_s2rel), np.mean(cv_score_b2abs), np.mean(cv_score_b2rel)]
        data_list.append(mask_scores)
        subj_scores[mask] = mask_scores
        if save:
            save_path = bundle_path+'mvpa/analyses/sub'+str(subj)
            if not os.path.isdir(save_path):
                os.makedirs(save_path)
            np.save(save_path+'/xdecode_abs_rel_scores.npy', subj_scores)
        
        mask_count += 1
        
df_data = pd.DataFrame(data_list, columns = ['Subj', 'Mask','S2Abs','S2Rel','B2Abs','B2Rel']) 
cols = [2,3,4,5]
df_plot = pd.melt(df_data, id_vars=["Subj","Mask"], var_name="Decoding Type", value_name="Accuracy")

#save
if save_group:
    save_path = bundle_path+'mvpa/analyses/Group/Cross_decoding'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    np.save(save_path+'/xdecode_abs_rel_scores_lowres_subs.npy', df_plot)

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Abs vs Rel All Subjects')
plt.show()


df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2Abs') + (df_plot['Decoding Type'] == 'S2Rel')]

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Train on Single Item All Subjects')
plt.show()

df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2Abs') + (df_plot['Decoding Type'] == 'B2Rel')]

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_btrain, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Train on Bundles All Subjects')
plt.show()

# stats
#s2abs_mask = df_plot[(df_plot['Decoding Type'] == 'S2Abs') & (df_plot['Mask'] == 'rACC')]['Accuracy'].values
#s2rel_mask = df_plot[(df_plot['Decoding Type'] == 'S2Rel') & (df_plot['Mask'] == 'rACC')]['Accuracy'].values
#
#stats.ttest_rel(s2abs_mask, s2rel_mask)
#stats.wilcoxon(s2abs_mask, s2rel_mask)