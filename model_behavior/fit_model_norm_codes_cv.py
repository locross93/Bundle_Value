#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:02:03 2022

@author: logancross
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import ParameterGrid

def get_trial_category(item_list):
    trial_cat = []
    for row in item_list_temp:
        if row[1] == -1:
            if row[0] < 100:
                trial_cat.append('Food item')
            elif row[0] >=100:
                trial_cat.append('Trinket item')
        else:
            if row[0] < 100 and row[1] < 100:
                trial_cat.append('Food bundle')
            elif row[0] >= 100 and row[1] >= 100:
                trial_cat.append('Trinket bundle')
            else:
                trial_cat.append('Mixed bundle')
                
    return trial_cat

def get_last_trial_choice(all_day_choices, all_day_runs):
    assert all_day_runs.shape[0] == all_day_choices.shape[0]
    last_trial_choice = np.zeros(len(all_day_choices)).astype(int)
    prec_ind = -1
    for i in range(len(all_day_choices)):
        if i == 0 or all_day_runs[prec_ind] != all_day_runs[i]:
            last_trial_choice[i] = 100
            prec_ind+=1
        else:
            prev_choice = all_day_choices[prec_ind]
#            if prev_choice == 0:
#                prev_choice = -1
            last_trial_choice[i] = prev_choice
            prec_ind+=1
            
    return last_trial_choice

def softmax_actions(values, ref_amount, beta):
    action_vals = np.column_stack([values, ref_amount])
    choice_probs = np.zeros(action_vals.shape)
    numer = np.exp(np.multiply(action_vals, beta))
    denom = np.sum(np.exp(np.multiply(action_vals, beta)), axis=1)
    choice_probs[:,0] = numer[:,0] / denom
    choice_probs[:,1] = numer[:,1] / denom
    
    # edit infs from overflow
    nan_inds = np.where(np.isnan(choice_probs))
    if len(nan_inds[0]) > 0:
        for i in range(len(nan_inds[0])):
            row = nan_inds[0][i]
            col = nan_inds[1][i]
            choice_probs[row,col] = 1
        
    #assert np.sum(choice_probs)
    
    return choice_probs

def compute_divisive_norm_behavior(values, ref_amount):
    avg_value = np.mean(values)
    div_normed_values = values.astype(float) / avg_value
    div_normed_ref = ref_amount.astype(float) / avg_value
    
    return div_normed_values, div_normed_ref

def compute_monte_carlo_norm_behavior(values, ref_amount):
    mc_normed_values = np.zeros([len(values)])
    mc_normed_ref = np.zeros([len(ref_amount)])
    
    for i,val in enumerate(values):
        ref = ref_amount[i]
        mc_ev = np.mean(values[:i+1])
        if mc_ev == 0:
            mc_normed_values[i] = 0
            mc_normed_ref[i] = ref
        else:
            mc_ev = np.mean(values[:i+1])
            mc_normed_values[i] = val.astype(float) / mc_ev
            mc_normed_ref[i] = ref.astype(float) / mc_ev
            
    return mc_normed_values, mc_normed_ref

def compute_rw_update_behavior(values, ref_amount, alpha, ev0):
    ev = ev0
    rw_normed_values = np.zeros([len(values)])
    rw_normed_ref = np.zeros([len(ref_amount)])
    for i,val in enumerate(values):
        pe = val - ev
        ev = ev + (alpha*pe)
        rw_normed_values[i] = val.astype(float) / ev
        ref = ref_amount[i]
        rw_normed_ref[i] = ref.astype(float) / ev
        
    return rw_normed_values, rw_normed_ref

def compute_rw_update_by_day_behavior(values, ref_amount, alpha, ev0_type, day_array):
    rw_normed_values = np.zeros([len(values)])
    rw_normed_ref = np.zeros([len(ref_amount)])
    for day in range(1,4):
        day_inds = np.where(day_array == day)[0]
        if ev0_type == '0':
            ev = 0
        elif ev0_type == 'Mean':
            ev = np.mean(values[day_inds])
        for ind in day_inds:
            val = values[ind]
            pe = val - ev
            ev = ev + (alpha*pe)
            rw_normed_values[ind] = val.astype(float) / ev
            ref = ref_amount[ind]
            rw_normed_ref[ind] = ref.astype(float) / ev
        
    return rw_normed_values, rw_normed_ref

def fit_softmax(betas, values, ref_amount, choice, num_params):
    nll_betas = np.zeros(len(betas))
    bic_betas = np.zeros(len(betas))
    for i,params in enumerate(betas):
        choice_probs = softmax_actions(values, ref_amount, params)
        nLL = 0
        for trial_num,c in enumerate(choice):
            if c==1: 
                prob_choice = choice_probs[trial_num,0]
                nLL = np.subtract(nLL, np.log(prob_choice))
            elif c==0:
                prob_choice = choice_probs[trial_num,1]
                nLL = np.subtract(nLL, np.log(prob_choice))
        nll_betas[i] = nLL
        bic_betas[i] = (2*nLL) + (np.log(len(choice))*num_params)
        
    nll_mle = np.min(nll_betas)
    beta_mle = betas[np.argmin(nll_betas)]
    bic_mle = bic_betas[np.argmin(nll_betas)]
    
    return nll_mle, beta_mle, bic_mle

def get_softmax_nll(beta, values, ref_amount, choice, num_params):
    choice_probs = softmax_actions(values, ref_amount, beta)
    nLL = 0
    for trial_num,c in enumerate(choice):
        if c==1: 
            prob_choice = choice_probs[trial_num,0]
            nLL = np.subtract(nLL, np.log(prob_choice))
        elif c==0:
            prob_choice = choice_probs[trial_num,1]
            nLL = np.subtract(nLL, np.log(prob_choice))
    bic = (2*nLL) + (np.log(len(choice))*num_params)
    
    return nLL, bic

def cross_val_score(betas, values, ref_amount, choice, day_array):
    gkf = GroupKFold(n_splits=3)
    cv_scores = []
    for train, test in gkf.split(choice, choice, groups=day_array):
        # fit model and find optimal params using training days
        nll_mle, beta_mle, bic_mle = fit_softmax(betas, values[train], ref_amount[train], choice[train], num_params=1)
        # compute cv accuracy on held out day
        choice_probs = softmax_actions(values[test], ref_amount[test], beta_mle)
        model_preds = np.argmax(np.flip(choice_probs, axis=1), axis=1)
        # find where model is 50/50 and randomly sample
        ties = choice_probs[:,0] == choice_probs[:,1]
        for ind in np.where(ties)[0]:
            model_preds[ind] = random.randint(0, 1)
        cv_accuracy = np.mean(model_preds == choice[test])
        cv_scores.append(cv_accuracy)
        
    return np.mean(cv_scores)

def cross_val_score_rw(alphas, betas, values, ref_amount, choice, day_array):
    trial_type = df_all_choice['Trial_type'].values
    ev0_type = 'Mean'
    
    param_grid = {'alpha': alphas, 'beta': betas}
    params_sweep = list(ParameterGrid(param_grid))
    
    gkf = GroupKFold(n_splits=3)
    cv_scores = []
    for train, test in gkf.split(choice, choice, groups=day_array):
        print day_array[train][0]
        print 'Sweeping over ',len(params_sweep),' params'
        # fit model and find optimal beta using training days
        train_values = values[train]
        train_ref_amount = ref_amount[train]
        train_choice = choice[train]
        train_day = day_array[train]
        train_trial_type = trial_type[train]
        s_inds_in = np.where(train_trial_type == 0)[0]
        b_inds_in = np.where(train_trial_type == 1)[0]
        test_values = values[test]
        test_ref_amount = ref_amount[test]
        test_choice = choice[test]
        test_day = day_array[test]
        test_trial_type = trial_type[test]
        s_inds_out = np.where(test_trial_type == 0)[0]
        b_inds_out = np.where(test_trial_type == 1)[0]
        nll_params = np.zeros(len(params_sweep))
        for model_num,params in enumerate(params_sweep):
            if model_num%50 == 0:
                print model_num
            train_alp = params['alpha']
            train_beta = params['beta']
            rwnorm_values = np.zeros([len(train_values)])
            rwnorm_ref_amount = np.zeros([len(train_ref_amount)])
            rwnorm_values[s_inds_in], rwnorm_ref_amount[s_inds_in] = compute_rw_update_by_day_behavior(train_values[s_inds_in], train_ref_amount[s_inds_in], train_alp, ev0_type, train_day[s_inds_in])
            rwnorm_values[b_inds_in], rwnorm_ref_amount[b_inds_in] = compute_rw_update_by_day_behavior(train_values[b_inds_in], train_ref_amount[b_inds_in], train_alp, ev0_type, train_day[b_inds_in])
            rw_nll_mle_temp, rw_bic_mle_temp = get_softmax_nll(train_beta, rwnorm_values, rwnorm_ref_amount, train_choice, num_params=2)
            nll_params[model_num] = rw_nll_mle_temp
        best_params_ind = np.argmin(nll_params)
        # compute cv accuracy on held out day
        fit_params = params_sweep[best_params_ind]
        fit_alp = fit_params['alpha']
        fit_beta = fit_params['beta']
        rwnorm_values = np.zeros([len(test_values)])
        rwnorm_ref_amount = np.zeros([len(test_ref_amount)])
        rwnorm_values[s_inds_out], rwnorm_ref_amount[s_inds_out] = compute_rw_update_by_day_behavior(test_values[s_inds_out], test_ref_amount[s_inds_out], fit_alp, ev0_type, test_day[s_inds_out])
        rwnorm_values[b_inds_out], rwnorm_ref_amount[b_inds_out] = compute_rw_update_by_day_behavior(test_values[b_inds_out], test_ref_amount[b_inds_out], fit_alp, ev0_type, test_day[b_inds_out])
        choice_probs = softmax_actions(rwnorm_values, rwnorm_ref_amount, fit_beta)
        model_preds = np.argmax(np.flip(choice_probs, axis=1), axis=1)
        # find where model is 50/50 and randomly sample
        ties = choice_probs[:,0] == choice_probs[:,1]
        for ind in np.where(ties)[0]:
            model_preds[ind] = random.randint(0, 1)
        cv_accuracy = np.mean(model_preds == test_choice)
        cv_scores.append(cv_accuracy)
        
    return np.mean(cv_scores)

cond_dict = {
	'Food item' : 1,
	'Trinket item' : 2,
	'Food bundle' : 3,
	'Trinket bundle' : 4,
	'Mixed bundle' : 5
	}

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

subs2test = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subs2test = ['114']

subj_scores_df = pd.DataFrame.from_dict({'Softmax': [], 'Norm Softmax': [], 'MC Softmax': [], 'Subj': []})

for subID in subs2test:
    print '\n'
    print subID
    
    df_all_choice = pd.DataFrame({'Choice':[0], 'Value':[0],
                              'Ref_Pos':[0], 'Ref_Amount':[0], 'Trial_type':[0], 'Prev_choice':[0], 'Subj': [0]})
    for day in range(1,4):
        subID_temp = subID+'-'+str(day)
        sub_logs_temp = log_folder+'bdm_items_sub_'+subID_temp+'.mat'
        sub_data_temp = loadmat(sub_logs_temp)
        bdm_item_value = sub_data_temp['value'].reshape(-1)
        bdm_item = sub_data_temp['item'].reshape(-1)
        sub_logs_temp2 = log_folder+'bdm_bundle_sub_'+subID_temp+'.mat'
        sub_data_temp2 = loadmat(sub_logs_temp2)
        bdm_bundle_value = sub_data_temp2['value'].reshape(-1)
        bdm_bundle_items = sub_data_temp2['item']
    
        choice_list = []
        item_list = []
        ref_pos_list = []
        trial_cat_list = []
        ref_amount_list = []
        run_list = []
        for run in range(1,6):
            run_log_file = log_folder+'choice_run'+str(run)+'_sub_'+subID+'-'+str(day)+'.mat'
            sub_choice_temp = loadmat(run_log_file)
            choice_list.append(sub_choice_temp['choice'].reshape(-1))
            ref_pos_list.append(sub_choice_temp['ref_pos'].reshape(-1))
            item_list_temp = sub_choice_temp['item']
            item_list.append(item_list_temp)
            # item category
            trial_cat_temp = get_trial_category(item_list_temp)
            trial_cat_list.append(trial_cat_temp)
            # reference
            median_bid_item = sub_choice_temp['median_bid_item'][0][0]
            median_bid_bundle = sub_choice_temp['median_bid_bundle'][0][0]
            temp_ref = []
            for t in trial_cat_temp:
                if cond_dict[t] < 3:
                    temp_ref.append(median_bid_item)
                elif cond_dict[t] >= 3:
                    temp_ref.append(median_bid_bundle)
            ref_amount_list.append(temp_ref)
            # run
            run_list.append([run for i in range(len(item_list_temp))])
        all_day_choices = np.vstack(choice_list).reshape(-1)
        all_day_items = np.vstack(item_list)
        trial_cat_allruns = [item for sublist in trial_cat_list for item in sublist]
        cond_nums_allruns  = [cond_dict[condition] for condition in trial_cat_allruns]
        cond_nums_allruns  = np.array([cond_dict[condition] for condition in trial_cat_allruns])
        all_day_ref_pos = np.vstack(ref_pos_list).reshape(-1)
        all_day_ref_amount = np.vstack(ref_amount_list).reshape(-1)
        all_day_runs = np.vstack(run_list).reshape(-1)
        
        item_or_bundle = (cond_nums_allruns > 2).astype(int)
        
        last_trial_choice = get_last_trial_choice(all_day_choices, all_day_runs)
        
        sitem_inds = np.where(cond_nums_allruns < 3)[0]
        bundle_inds = np.where(cond_nums_allruns > 2)[0]
        # get stimulus values
        stim_values = np.zeros(all_day_choices.shape[0])
        for i in range(len(all_day_choices)):
            if item_or_bundle[i] == 0:
                # sitem
                temp_item = all_day_items[i,0]
                match_ind = np.where(bdm_item == temp_item)[0]
                stim_values[i] = bdm_item_value[match_ind]
            elif item_or_bundle[i] == 1:
                # bundle
                temp_bundle = all_day_items[i,:]
                match_ind = np.where((bdm_bundle_items == temp_bundle).all(axis=1))[0]
                stim_values[i] = bdm_bundle_value[match_ind]
                
        # divisive norm
        dvnorm_values = np.zeros([len(stim_values)])
        dvnorm_ref = np.zeros([len(all_day_ref_amount)])
        dvnorm_values[sitem_inds] , dvnorm_ref[sitem_inds] = compute_divisive_norm_behavior(stim_values[sitem_inds], all_day_ref_amount[sitem_inds]) 
        dvnorm_values[bundle_inds] , dvnorm_ref[bundle_inds]= compute_divisive_norm_behavior(stim_values[bundle_inds], all_day_ref_amount[bundle_inds])
        
        # monte carlo
        mcnorm_values = np.zeros([len(stim_values)])
        mcnorm_ref = np.zeros([len(all_day_ref_amount)])
        mcnorm_values[sitem_inds] , mcnorm_ref[sitem_inds] = compute_monte_carlo_norm_behavior(stim_values[sitem_inds], all_day_ref_amount[sitem_inds])
        mcnorm_values[bundle_inds] , mcnorm_ref[bundle_inds] = compute_monte_carlo_norm_behavior(stim_values[bundle_inds], all_day_ref_amount[bundle_inds])
        
        # wtp - ref
        wtp_ref_values = stim_values - all_day_ref_amount
        
        # wtp - ref / mean
        dvwr_norm_values = np.zeros([len(stim_values)])
        avg_svalue = np.mean(stim_values[sitem_inds])
        dvwr_norm_values[sitem_inds] = wtp_ref_values[sitem_inds].astype(float) / avg_svalue
        avg_bvalue = np.mean(stim_values[bundle_inds])
        dvwr_norm_values[bundle_inds] = wtp_ref_values[bundle_inds].astype(float) / avg_bvalue
        
        df_all_choice_day = pd.DataFrame({'Choice': all_day_choices, 'Value': stim_values, 'Ref_Amount': all_day_ref_amount,
                                        'DV_Norm_Value': dvnorm_values, 'DV_Norm_Ref_Amount': dvnorm_ref,
                                        'MC_Norm_Value': mcnorm_values, 'MC_Norm_Ref_Amount': mcnorm_ref,
                                        'WTP_Ref': wtp_ref_values, 'DV_WR_Norm_Value': dvwr_norm_values, 'Ref_Pos': all_day_ref_pos,
                                        'Trial_type': item_or_bundle,'Prev_choice': last_trial_choice,
                                        'Day': [day for i in range(len(all_day_choices))],
                                        'Subj': [subID for i in range(len(all_day_choices))]})
    
        # drop error trials
        temp_choices = df_all_choice_day['Choice'].values
        err_inds = np.where(temp_choices == 100)[0]
        if len(err_inds) > 0:
            df_all_choice_day = df_all_choice_day[~df_all_choice_day.index.isin(err_inds)]
        assert len(np.unique(df_all_choice_day['Choice'])) == 2
        
        # make choices boolean
        df_all_choice_day.loc[:,'Choice'] = df_all_choice_day['Choice'].values.astype(bool)
        
        df_all_choice = pd.concat([df_all_choice, df_all_choice_day], ignore_index=True, sort=True)

    # drop first element
    df_all_choice = df_all_choice.drop([0])     
    df_all_choice = df_all_choice.reset_index(drop=True)  

    choice = df_all_choice['Choice'].values
    values = df_all_choice['Value'].values
    ref_amount = df_all_choice['Ref_Amount'].values
    day_array = df_all_choice['Day'].values
    norm_values = df_all_choice['DV_Norm_Value'].values
    norm_ref_amount = df_all_choice['DV_Norm_Ref_Amount'].values
    mc_values = df_all_choice['MC_Norm_Value'].values
    mc_ref_amount = df_all_choice['MC_Norm_Ref_Amount'].values

    betas = np.linspace(0.01, 10, num=100)
    
    cv_score = cross_val_score(betas, values, ref_amount, choice, day_array)
    print 'Softmax'
    print cv_score
    
    norm_cv_score = cross_val_score(betas, norm_values, norm_ref_amount, choice, day_array)
    print 'Norm Softmax'
    print norm_cv_score
    
    mc_cv_score = cross_val_score(betas, mc_values, mc_ref_amount, choice, day_array)
    print 'MC Softmax'
    print mc_cv_score
    
    alphas = np.linspace(0.005, 0.9, num=100)
    rw_cv_score = cross_val_score_rw(alphas, betas, values, ref_amount, choice, day_array)
    print 'RW Softmax'
    print rw_cv_score    
    
    scores_dict = {'Softmax': [cv_score], 'Norm Softmax': [norm_cv_score], 
                   'MC Softmax': [mc_cv_score], 'RW Softmax': [rw_cv_score], 'Subj': [subID]}
    temp_df = pd.DataFrame.from_dict(scores_dict)
    subj_scores_df = pd.concat([subj_scores_df, temp_df], sort=True, ignore_index=True)
  
models = ['Softmax','Norm Softmax','MC Softmax', 'RW Softmax']

for model in models:
    temp_vals = subj_scores_df[model].values
    print model
    print np.mean(temp_vals)
    print np.std(temp_vals)
    print '\n'
