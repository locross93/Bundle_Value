#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 13:01:42 2022

@author: logancross
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
    mc_normed_values = np.zeros([len(values)])
    for i,val in enumerate(values):
        if val == 0:
            mc_normed_values[i] = 0
        else:
            mc_normed_values[i] = (beta + val.astype(float)) / (sigma + np.mean(values[:i+1]))
        
    return mc_normed_values

def compute_monte_carlo_norm_by_day(values, beta, sigma, day_array):
    mc_normed_values = np.zeros([len(values)])
    day_start_ind = 0
    for i,val in enumerate(values):
        if day_array[i] != day_array[day_start_ind]:
            day_start_ind = i
            
        if val == 0:
            mc_normed_values[i] = 0
        else:
            mc_normed_values[i] = (beta + val.astype(float)) / (sigma + np.mean(values[day_start_ind:i+1]))
        
    return mc_normed_values

def compute_rw_update(values, alpha, ev0, beta, sigma):
    ev = ev0
    rw_normed_values = np.zeros([len(values)])
    for i,val in enumerate(values):
        pe = val - ev
        ev = ev + (alpha*pe)
        rw_normed_values[i] = (beta + val.astype(float)) / (sigma + ev)
        
    return rw_normed_values

def compute_rw_update_by_day(values, alpha, ev0_type, beta, sigma, day_array):
    rw_normed_values = np.zeros([len(values)])
    for day in range(3):
        day_inds = np.where(day_array == day)[0]
        if ev0_type == '0':
            ev = 0
        elif ev0_type == 'Mean':
            ev = np.mean(values[day_inds])
        for ind in day_inds:
            val = values[ind]
            pe = val - ev
            ev = ev + (alpha*pe)
            rw_normed_values[ind] = (beta + val.astype(float)) / (sigma + ev)
        
    return rw_normed_values

def compute_weighted_rw_update_by_day(values, alpha, ev0_type, beta, sigma, lmbda, day_array, sitem_inds, bundle_inds):
    rw_normed_values = np.zeros([len(values)])
    for day in range(3):
        day_inds = np.where(day_array == day)[0]
        sitem_inds_day = np.intersect1d(sitem_inds, day_inds)
        bundle_inds_day = np.intersect1d(bundle_inds, day_inds)
        if ev0_type == '0':
            ev_s = 0
            ev_b = 0
        elif ev0_type == 'Mean':
            ev_s = (lmbda*np.mean(values[sitem_inds_day])) + ((1-lmbda)*np.mean(values[bundle_inds_day]))
            ev_b = (lmbda*np.mean(values[bundle_inds_day])) + ((1-lmbda)*np.mean(values[sitem_inds_day]))
        for ind in day_inds:
            val = values[ind]
            pe_s = val - ev_s
            pe_b = val - ev_b
            if ind in sitem_inds_day:
                ev_s = ev_s + (alpha*pe_s*lmbda)
                ev_b = ev_b + (alpha*pe_b*(1-lmbda))
                ev_t = ev_s
            elif ind in bundle_inds_day:
                ev_s = ev_s + (alpha*pe_s*(1-lmbda))
                ev_b = ev_b + (alpha*pe_b*lmbda)
                ev_t = ev_b
            rw_normed_values[ind] = (beta + val.astype(float)) / (sigma + ev_t)
        
    return rw_normed_values

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

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

save = True
save_prefix = 'rsa_advanced_frac_model_adjr2'
#save_prefix = 'rsa_div_rw_mc_norm_sweeps_adjr2'
#save_prefix = 'rsa_mc_rw_by_day'
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

param_grid1 = {'model': ['MC'], 'beta': [0, 0.1, 0.3, 1, 10, 100, 1000], 'sigma': [0, 0.1, 0.3, 0.4, 1, 10]}
params_sweep1 = list(ParameterGrid(param_grid1))
param_grid2 = {'model': ['MC by day'], 'beta': [0, 0.1, 0.3, 1, 10, 100, 1000], 'sigma': [0, 0.1, 0.3, 0.4, 1, 10]}
params_sweep2 = list(ParameterGrid(param_grid2))
param_grid3 = {'model': ['RW Update'], 'alpha': [0.01, 0.02, 0.05, 0.1, 0.3, 0.5], 'beta': [0, 0.1, 0.3, 1, 10, 100], 'sigma': [0, 0.1, 0.4, 1, 3, 10]}
params_sweep3 = list(ParameterGrid(param_grid3))
param_grid4 = {'model': ['RW Update by day'], 'alpha': [0.01, 0.02, 0.05, 0.1, 0.3, 0.5], 'beta': [0, 0.1, 0.3, 1, 10, 100], 'sigma': [0, 0.1, 0.4, 1, 3, 10]}
params_sweep4 = list(ParameterGrid(param_grid4))
params_sweep = params_sweep1 + params_sweep2 + params_sweep3 + params_sweep4

model_labels = ['MC', 'MC by day', 'RW Update', 'RW Update by day']
xaxis_labels = model_labels

param_grid = {'model': ['Divisive'], 'beta': [0, 0.1, 0.5, 1, 2, 3, 5, 10, 100, 1000], 'sigma': [0, 0.1, 0.3, 0.4, 1, 2, 3, 5, 10, 100]}
params_sweep = list(ParameterGrid(param_grid))

model_labels = ['Divisive']
xaxis_labels = ['Advanced Fractional Model']


##############
# param_grid1 = {'model': ['MC by day'], 'beta': [0], 'sigma': [0]}
# params_sweep1 = list(ParameterGrid(param_grid1))
# param_grid2 = {'model': ['RW Update by day'], 'alpha': [0.01, 0.02, 0.05], 'beta': [0, 0.1, 0.3, 1, 10,], 'sigma': [0, 0.1, 0.4, 1, 3]}
# params_sweep2 = list(ParameterGrid(param_grid2))
# params_sweep = params_sweep1 + params_sweep2


# model_labels = ['MC by day', 'RW Update by day']
# xaxis_labels = model_labels

conditions = ['Food item', 'Trinket item', 'Food bundle','Trinket bundle','Mixed bundle']

for subj in subj_list:
    start_time = time.time()
    print subj
    
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
      
    # hyperparameter sweep
    fmri2model_tval = np.zeros([len(mask_loop), len(params_sweep)])
    fmri2model_bic = np.zeros([len(mask_loop), len(params_sweep)])
    fmri2model_adjr2 = np.zeros([len(mask_loop), len(params_sweep)])
    fmri2model_corr = np.zeros([len(mask_loop), len(params_sweep)])
    model_inds = []
    print 'Sweeping over ',len(params_sweep),' params'
    for model_num,params in enumerate(params_sweep):
        if model_num%50 == 0:
            print model_num
        norm_values = np.zeros([len(abs_value)])
        model = params['model']
        model_inds.append(model)
        beta = params['beta']
        sigma = params['sigma']
        
        if model == 'Divisive':
            norm_values[sitem_inds] = compute_divisive_norm(abs_value[sitem_inds], beta, sigma) 
            norm_values[bundle_inds] = compute_divisive_norm(abs_value[bundle_inds], beta, sigma) 
            num_params = 2
        elif model == 'RW Update':
            alp = params['alpha']
            ev0 = np.mean(abs_value[sitem_inds])
            norm_values[sitem_inds] = compute_rw_update(abs_value[sitem_inds], alp, ev0 , beta, sigma)
            ev0 = np.mean(abs_value[bundle_inds])
            norm_values[bundle_inds] = compute_rw_update(abs_value[bundle_inds], alp, ev0, beta, sigma)
            num_params = 3
        elif model == 'RW Update by day':
            alp = params['alpha']
            ev0_type = 'Mean'
            norm_values[sitem_inds] = compute_rw_update_by_day(abs_value[sitem_inds], alp, ev0_type , beta, sigma, day_array[sitem_inds])
            norm_values[bundle_inds] = compute_rw_update_by_day(abs_value[bundle_inds], alp, ev0_type , beta, sigma, day_array[bundle_inds])
            num_params = 3
        elif model == 'MC':
            norm_values[sitem_inds] = compute_monte_carlo_norm(abs_value[sitem_inds], beta, sigma)
            norm_values[bundle_inds] = compute_monte_carlo_norm(abs_value[bundle_inds], beta, sigma)
            num_params = 2
        elif model == 'MC by day':
            norm_values[sitem_inds] = compute_monte_carlo_norm_by_day(abs_value[sitem_inds], beta, sigma, day_array[sitem_inds])
            norm_values[bundle_inds] = compute_monte_carlo_norm_by_day(abs_value[bundle_inds], beta, sigma, day_array[bundle_inds])
            num_params = 2
        elif model == 'Weighted RW Update by day':
            alp = params['alpha']
            lmbda = params['lambda']
            ev0_type = 'Mean'
            norm_values = compute_weighted_rw_update_by_day(abs_value, alp, ev0_type, beta, sigma, lmbda, day_array, sitem_inds, bundle_inds)
            num_params = 4
            
        ds_value = dataset_wizard(norm_values, targets=np.zeros(len(norm_values)))
        dsm = PDist(pairwise_metric='euclidean', square=square_dsm_bool)
        res_value = dsm(ds_value)
        if ranked:
            res_value = rankdata(res_value)
        else:
            res_value = res_value.samples.reshape(-1)
        
        model_dsm_names = ['choice','item_or_bundle','lr_choice']
        for mask_num in range(len(mask_loop)):
            partial_dsms = [target_dsms[model_dsm][btwn_day_inds] for model_dsm in model_dsm_names]
            if remove_within_day:
                temp_fmri = fmri_dsm_list[mask_num][btwn_day_inds]
                temp_model = res_value[btwn_day_inds]
            else:
                temp_fmri = fmri_dsm_list[mask_num]
                temp_model = res_value
            temp_tval, bic, adj_r2 = rsa_regression(temp_fmri, partial_dsms, temp_model, num_params)
            fmri2model_tval[mask_num, model_num] = temp_tval
            fmri2model_bic[mask_num, model_num] = bic
            fmri2model_adjr2[mask_num, model_num] = adj_r2
            temp_correl = pearsonr(temp_fmri, temp_model)[0]
            fmri2model_corr[mask_num, model_num] = temp_correl
            
    fmri2model_matrix_plot = np.zeros([len(mask_loop), len(xaxis_labels)])
    fmri2model_matrix_save = np.zeros([len(mask_loop), len(xaxis_labels)*4])
    best_params = {}
    for col,model_label in enumerate(model_labels):
        match_inds = [x for x in range(len(model_inds)) if model_inds[x] == model_label]
        scores_across_rois = np.mean(fmri2model_adjr2[:,match_inds], axis=0)
        best_ind = np.argmax(scores_across_rois)
        max_orig_ind = match_inds[best_ind]
        fmri2model_matrix_plot[:,col] = fmri2model_adjr2[:,max_orig_ind]
        fmri2model_matrix_save[:,col*4] = fmri2model_adjr2[:,max_orig_ind]
        fmri2model_matrix_save[:,(col*4)+1] = fmri2model_tval[:,max_orig_ind]
        fmri2model_matrix_save[:,(col*4)+2] = fmri2model_bic[:,max_orig_ind]
        fmri2model_matrix_save[:,(col*4)+3] = fmri2model_corr[:,max_orig_ind]
        best_params[model_label] = params_sweep[max_orig_ind]
        
    if ranked:
        legend_label = 'Spearman Correlation ($\\rho$)'
    else:
        legend_label = 'Pearson Correlation (r)'
    vmax = 0.1
    mask_mat = np.zeros_like(fmri2model_matrix_plot)
    mask_mat[np.triu_indices_from(mask_mat)] = True   
    f = sns.heatmap(fmri2model_matrix_plot, annot=True, annot_kws={"size": 7},  
                    xticklabels=xaxis_labels, yticklabels=mask_names, vmin=0.0, vmax=vmax, cbar_kws={'label':legend_label}) 
    f.set_title('Sub'+str(subj)+' RSA', fontsize = 20)   
    plt.show()
    
    subj_df_plot = pd.DataFrame(fmri2model_matrix_plot, index=mask_names, columns=xaxis_labels)
    subj_df_plot['ROI'] = subj_df_plot.index
    subj_tidy_df = subj_df_plot.melt(id_vars='ROI',value_vars=xaxis_labels)
    f = sns.barplot(x='ROI', y='value', hue='variable', data=subj_tidy_df)
    sns.despine()
    f.set_xticklabels(f.get_xticklabels(), rotation=90)
    plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
    plt.title('RSA Absolute vs. Relative Value Sub'+subj)
    plt.ylabel('RSA Adj R2')
    plt.show()
    
    column_names = [[mdl_label+'_adjr2',mdl_label+'_tval',mdl_label+'_bic',mdl_label+'_corr'] for mdl_label in xaxis_labels]
    column_names = [element for sublist in column_names for element in sublist]
    subj_df = pd.DataFrame(fmri2model_matrix_save, index=mask_names, columns=column_names)
    subj_df['ROI'] = subj_df.index

    for model_label in model_labels:
        print 'Best params ',model_label,best_params[model_label]
        
    print '\n'
    
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
        
        
        
        

