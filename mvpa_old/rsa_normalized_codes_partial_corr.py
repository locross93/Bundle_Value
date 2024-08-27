# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 11:56:15 2023

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
from sklearn.model_selection import ParameterGrid
import pandas as pd
import statsmodels
import statsmodels.api as sm

def compute_divisive_norm(values, beta, sigma):
    avg_value = np.mean(values)
    div_normed_values = (beta + values.astype(float)) / (sigma + avg_value)
    
    return div_normed_values

def compute_monte_carlo_norm(values, beta, sigma):
    mc_normed_values = values
    for i,val in enumerate(values):
        if val == 0:
            mc_normed_values[i] = 0
        else:
            mc_normed_values[i] = (beta + values.astype(float)) / (sigma + np.mean(values[:i+1]))
        
    return mc_normed_values

def compute_rw_update(values, alpha, ev0, beta=0, sigma=1):
    ev = ev0
    rw_normed_values = values
    ev_t = np.zeros(len(values))
    for i,val in enumerate(values):
        pe = val - ev
        ev = ev + (alpha*pe)
        rw_normed_values[i] = (beta + val.astype(float)) / (sigma + ev)
        ev_t[i] = ev
        
    return rw_normed_values, ev_t

def rsa_regression(temp_fmri, partial_dsms, target_dsm, num_params):
    model_dsms = partial_dsms
    model_dsms.append(target_dsm)
    model_dsm_array = np.column_stack((model_dsms))
    X = sm.add_constant(model_dsm_array)
    mod = sm.OLS(temp_fmri, X)
    res = mod.fit()
    target_dsm_tval = res.tvalues[-1]
    df_modelwc = res.df_model + 1 + num_params
    nobs = nobs = X.shape[0]
    bic = statsmodels.tools.eval_measures.bic(res.llf, nobs, df_modelwc)
    adj_r2 = 1 - (nobs-1)/(res.df_resid-num_params) * (1-res.rsquared)
    
    return target_dsm_tval, bic, adj_r2
        

#bundle_path = '/Users/logancross/Documents/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']

save = True
save_prefix = 'rsa_abs_value_adjr2'
save_dsms = False

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

square_dsm_bool = False
ranked = False
remove_within_day = True

value_type = ['Abs', 'Rel', 'Subtract Median','Subtract Mean','Divide Mean']
xaxis_labels = ['Absolute','Relative','WTP - Ref','Subtract Mean','Divide Mean']

value_type = ['Abs']
xaxis_labels = ['Absolute']

#value_type = ['Divide Mean','Running Average', 'RW Update', 'Divisive']
#
#xaxis_labels = ['Divide Mean', 'Running Average', 'RW Update', 'Divisive']
#
#value_type = ['Divisive']
#
#xaxis_labels = ['Divisive']

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

for subj in subj_list:
    start_time = time.time()
    print subj
    #which ds to use and which mask to use
    glm_ds_file = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/all_trials_4D.nii.gz'
    brain_mask = bundle_path+'fmri/GLM/SPM/sub'+str(subj)+'/beta_everytrial_fullmodel/mask.nii'
    
    fmri_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/fmri_dsm_list'
    fmri_dsm_list = h5load(fmri_dsms_file)
    
    if int(subj) < 104:
        target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms.csv'
        target_dsms_df = pd.read_csv(target_dsms_file)
        target_dsms = target_dsms_df.to_dict()
        
        subj_info_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/info_dict'
        subj_info_dict = h5load(subj_info_file+'_list')
        abs_value = subj_info_dict[0]
        trial_categ = subj_info_dict[1]
        sitem_inds = subj_info_dict[2]
        bundle_inds = subj_info_dict[3]
        run_array = subj_info_dict[4]
        day_array = subj_info_dict[5]
        
        #item or bundle?
        item_or_bundle = trial_categ
        item_or_bundle[sitem_inds] = 0
        item_or_bundle[bundle_inds] = 1
        assert np.max(item_or_bundle) == 1
        
        num_trials = len(item_or_bundle)
        ds_trial_cat = dataset_wizard(item_or_bundle, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_trial_cat = dsm(ds_trial_cat)
        if ranked:
            res_trial_cat = rankdata(res_trial_cat)
        else:
            res_trial_cat = res_trial_cat.samples.reshape(-1)
        target_dsms['item_or_bundle'] = res_trial_cat
        
        #choice 
        choice = mvpa_utils.get_fmri_choices(subj, conditions)
        ds_choice = dataset_wizard(choice, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_choice = dsm(ds_choice)
        if ranked:
            res_choice = rankdata(res_choice)
        else:
            res_choice = res_choice.samples.reshape(-1)
        target_dsms['choice'] = res_choice
        
        #left vs right choice
        lr_choice_list = np.genfromtxt(bundle_path+'mvpa/OpenfMRI_Format/sub'+str(subj)+'/model/task_info/lr_choice.txt')
        lr_choice = lr_choice_list[:,1]
        #if only individual item trials, only include these trials
        if len(conditions) < 5:
            lr_choice = lr_choice[inds_in_conds]
        ds_lr_choice = dataset_wizard(lr_choice, targets=np.zeros(num_trials))
        dsm = PDist(pairwise_metric='matching', square=square_dsm_bool)
        res_lr_choice = dsm(ds_lr_choice)
        if ranked:
            res_lr_choice = rankdata(res_lr_choice)
        else:
            res_lr_choice = res_lr_choice.samples.reshape(-1)
        target_dsms['lr_choice'] = res_lr_choice
        
        if remove_within_day:
            res_day = np.array(target_dsms['day'].values())
            if ranked:
                day_values = np.unique(res_day)
                high_rank = np.max(day_values)
                btwn_day_inds = np.where(res_day == high_rank)[0]
            else:
                btwn_day_inds = np.where(res_day == 1)[0]
    else:
        target_dsms_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/target_dsms'
        target_dsms = h5load(target_dsms_file)

        subj_info_file = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/info_dict'
        subj_info_dict = h5load(subj_info_file)
        abs_value = subj_info_dict['abs_value']
        trial_categ = subj_info_dict['trial_categ']
        sitem_inds = subj_info_dict['sitem_inds']
        bundle_inds = subj_info_dict['bundle_inds']
        run_array = subj_info_dict['run_array']
        day_array = subj_info_dict['day_array']
    
        if remove_within_day:
            res_day = target_dsms['day']
            if ranked:
                day_values = np.unique(res_day)
                high_rank = np.max(day_values)
                btwn_day_inds = np.where(res_day == high_rank)[0]
            else:
                btwn_day_inds = np.where(res_day == 1)[0]
    
    fmri2model_matrix_save = np.zeros([len(mask_loop), len(xaxis_labels)*4])
    for col,val_type in enumerate(value_type):
        print val_type
        if val_type == 'Abs':
            norm_values = abs_value
            num_params = 0
        elif val_type == 'Rel':   
            relative_value = True
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            zscore_value = fds.targets
            num_params = 0
        elif val_type == 'Subtract Median':   
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            # wtp - ref - IMPROVE THIS WITH REAL REF
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) - np.median(abs_value[sitem_inds]) 
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) - np.median(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
            num_params = 0
        elif val_type == 'Subtract Mean':   
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) - np.mean(abs_value[sitem_inds])
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) - np.mean(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
            num_params = 0
        elif val_type == 'Divide Mean':   
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            subnorm_value = np.zeros([len(abs_value)])
            subnorm_value[sitem_inds] = abs_value[sitem_inds].astype(float) / np.mean(abs_value[sitem_inds]) 
            subnorm_value[bundle_inds] = abs_value[bundle_inds].astype(float) / np.mean(abs_value[bundle_inds]) 
            fds.targets = subnorm_value
            num_params = 0
        elif val_type == 'Divisive':   
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            divnorm_value = np.zeros([len(abs_value)])
            beta = 0
            sigma = 0.4
            divnorm_value[sitem_inds] = compute_divisive_norm(abs_value[sitem_inds], beta, sigma) 
            divnorm_value[bundle_inds] = compute_divisive_norm(abs_value[bundle_inds], beta, sigma) 
            fds.targets = divnorm_value
            num_params = 2
        elif val_type == 'Running Average':
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            mcnorm_values = np.zeros([len(abs_value)])
            mcnorm_values[sitem_inds] = compute_monte_carlo_norm(abs_value[sitem_inds])
            mcnorm_values[bundle_inds] = compute_monte_carlo_norm(abs_value[bundle_inds])
            fds.targets = mcnorm_values
            num_params = 0
        elif val_type == 'RW Update':
            relative_value = False
            fds = mvpa_utils.make_targets(subj, glm_ds_file, brain_mask, conditions, relative_value=relative_value)
            abs_value = fds.targets
            trial_categ = fds.sa.trial_categ
            sitem_inds = np.where(trial_categ < 3)[0]
            bundle_inds = np.where(trial_categ > 2)[0]
            rwnorm_values = np.zeros([len(abs_value)])
            alpha = 0.05
            beta = 10
            sigma = 0
            rwnorm_values[sitem_inds], ev_t = compute_rw_update(abs_value[sitem_inds], alpha, ev0=np.mean(abs_value[sitem_inds]), beta=beta, sigma=sigma)
            rwnorm_values[bundle_inds], ev_t = compute_rw_update(abs_value[bundle_inds], alpha, ev0=np.mean(abs_value[bundle_inds]), beta=beta, sigma=sigma)
            fds.targets = rwnorm_values
            num_params = 3
            
        ds_value = dataset_wizard(norm_values, targets=np.zeros(len(norm_values)))
        dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
        res_value = dsm(ds_value)
        if ranked:
            res_value = rankdata(res_value)
        else:
            res_value = res_value.samples.reshape(-1)
        
        model_dsm_names = ['choice','item_or_bundle','lr_choice']
        fmri2model_tval = np.zeros([len(mask_loop)])
        fmri2model_bic = np.zeros([len(mask_loop)])
        fmri2model_adjr2 = np.zeros([len(mask_loop)])
        fmri2model_corr = np.zeros([len(mask_loop)])
        for mask_num in range(len(mask_loop)):
            partial_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
            if remove_within_day:
                temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
                temp_model = res_value[btwn_day_inds]
            else:
                temp_fmri = fmri_dsm_list[mask_num]
                temp_model = res_value
            temp_tval, bic, adj_r2 = rsa_regression(temp_fmri, partial_dsms, temp_model, num_params)
            fmri2model_tval[mask_num] = temp_tval
            fmri2model_bic[mask_num] = bic
            fmri2model_adjr2[mask_num] = adj_r2
            temp_correl = pearsonr(temp_fmri, temp_model)[0]
            fmri2model_corr[mask_num] = temp_correl
                
        fmri2model_matrix_save[:,col*4] = fmri2model_adjr2
        fmri2model_matrix_save[:,(col*4)+1] = fmri2model_tval
        fmri2model_matrix_save[:,(col*4)+2] = fmri2model_bic
        fmri2model_matrix_save[:,(col*4)+3] = fmri2model_corr   
        
    column_names = [[mdl_label+'_adjr2',mdl_label+'_tval',mdl_label+'_bic',mdl_label+'_corr'] for mdl_label in xaxis_labels]
    column_names = [element for sublist in column_names for element in sublist]
    subj_df = pd.DataFrame(fmri2model_matrix_save, index=mask_names, columns=column_names)
    subj_df['ROI'] = subj_df.index
    
    print 'Finished ',time.time() - start_time
    
    #save
    if save:
        save_path = bundle_path+'mvpa/analyses/sub'+str(subj)+'/'
        subj_df.to_csv(save_path+save_prefix+'.csv')
        
    # save fmri dsms and target dsms
    if save_dsms:
        save_path = bundle_path+'mvpa/presaved_data/sub'+str(subj)+'/'
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        # fmri dsms
        save_file = save_path+'fmri_dsm_list'
        if ranked:
            save_file = save_file+'_ranked'
        h5save(save_file, fmri_dsm_list)
        
        # target dsms
        save_file2 = save_path+'target_dsms'
        if ranked:
            save_file2 = save_file2+'_ranked'
        h5save(save_file2, target_dsms)
        
        # other import variables
        subj_info_dict = {}
        subj_info_dict['abs_value'] = abs_value 
        subj_info_dict['trial_categ'] = trial_categ
        subj_info_dict['sitem_inds'] = sitem_inds
        subj_info_dict['bundle_inds'] = bundle_inds
        save_file3 = save_path+'info_dict'
        h5save(save_file3, subj_info_dict)