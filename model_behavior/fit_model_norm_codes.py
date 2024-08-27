#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 20 15:46:54 2022

@author: logancross
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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

# Compute the correlation between value representations
def compute_value_correlations(values, norm_values, mc_values, rw_values, rw_day_values, wtp_ref_values):
    value_df = pd.DataFrame({'Absolute Value': values,
                             'Divisive Norm': norm_values,
                             'Monte Carlo': mc_values,
                             'Rescorla Wagner': rw_values,
                             'Rescorla Wagner By Day': rw_day_values,
                             'WTP - Ref': wtp_ref_values})
    
    corr_matrix = value_df.corr()
    return corr_matrix

cond_dict = {
	'Food item' : 1,
	'Trinket item' : 2,
	'Food bundle' : 3,
	'Trinket bundle' : 4,
	'Mixed bundle' : 5
	}

#log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'
log_folder = '/Users/locro/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

#subs2test = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
subs2test = ['114']

choice_bias_nlls = []
softmax_nlls = []
norm_softmax_nlls = []
mc_softmax_nlls = []
rw_softmax_nlls = []
rw_day_softmax_nlls = []
sbtrt_softmax_nlls = []

choice_bias_bic = []
softmax_bic = []
norm_softmax_bic = []
mc_softmax_bic = []
rw_softmax_bic = []
rw_day_softmax_bic = []
sbtrt_softmax_bic = []

# Initialize lists to store correlation matrices for each subject
corr_matrices = []

for subID in subs2test:
    print('\n')
    print(subID)
    
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
    norm_values = df_all_choice['DV_Norm_Value'].values
    norm_ref_amount = df_all_choice['DV_Norm_Ref_Amount'].values
    mc_values = df_all_choice['MC_Norm_Value'].values
    mc_ref_amount = df_all_choice['MC_Norm_Ref_Amount'].values
    
    bias_params = np.linspace(0.01, 1, num=100)
    bias_params = bias_params[:-1]
    
    nll_bias = np.zeros(len(bias_params))
    
    for i,bias in enumerate(bias_params):
        nLL = 0
        for c in choice:
            if c==1:
                prob_choice = bias
            elif c==0:
                prob_choice = 1 - bias
            nLL = np.subtract(nLL, np.log(prob_choice))
        nll_bias[i] = nLL
            
    print('Choice bias')
    print(np.min(nll_bias))
    print(bias_params[np.argmin(nll_bias)])
    
    choice_bias_nlls.append(np.min(nll_bias))

    betas = np.linspace(0.01, 10, num=100)
    
    nll_mle, beta_mle, bic_mle = fit_softmax(betas, values, ref_amount, choice, num_params=1)
    print('Softmax')
    print(nll_mle)
    print(beta_mle)
    softmax_nlls.append(nll_mle)
    softmax_bic.append(bic_mle)
    
    norm_nll_mle, norm_beta_mle, norm_bic_mle = fit_softmax(betas, norm_values, norm_ref_amount, choice, num_params=1)
    print('Norm Softmax')
    print(norm_nll_mle)
    print(norm_beta_mle)
    norm_softmax_nlls.append(norm_nll_mle)
    norm_softmax_bic.append(norm_bic_mle)
    
    mc_nll_mle, mc_beta_mle, mc_bic_mle = fit_softmax(betas, mc_values, mc_ref_amount, choice, num_params=1)
    print('MC')
    print(mc_nll_mle)
    print(mc_beta_mle)
    mc_softmax_nlls.append(mc_nll_mle)
    mc_softmax_bic.append(mc_bic_mle)
    
    # rescorla wagner normalization
    trial_type = df_all_choice['Trial_type'].values
    s_inds = np.where(trial_type == 0)[0]
    b_inds = np.where(trial_type == 1)[0]
    sitem_ev0 = np.mean(values[s_inds])
    bundle_ev0 = np.mean(values[b_inds])
    alphas = np.linspace(0.005, 0.9, num=100)
    nll_alphas = np.zeros(len(alphas))
    bic_alphas = np.zeros(len(alphas))
    for i,alp in enumerate(alphas):
        rwnorm_values = np.zeros([len(values)])
        rwnorm_ref_amount = np.zeros([len(ref_amount)])
        rwnorm_values[s_inds], rwnorm_ref_amount[s_inds] = compute_rw_update_behavior(values[s_inds], ref_amount[s_inds], alp, sitem_ev0)
        rwnorm_values[b_inds], rwnorm_ref_amount[b_inds] = compute_rw_update_behavior(values[b_inds], ref_amount[b_inds], alp, bundle_ev0)
        rw_nll_mle_temp, rw_beta_mle_temp, rw_bic_mle_temp = fit_softmax(betas, rwnorm_values, rwnorm_ref_amount, choice, num_params=2)
        nll_alphas[i] = rw_nll_mle_temp
        bic_alphas[i] = rw_bic_mle_temp
    rw_nll_mle = np.min(nll_alphas)
    rw_alpha_mle = alphas[np.argmin(nll_alphas)]
    rw_bic_mle = bic_alphas[np.argmin(nll_alphas)]
    print('RW')
    print(rw_nll_mle)
    print(rw_alpha_mle)
    rw_softmax_nlls.append(rw_nll_mle)
    rw_softmax_bic.append(rw_bic_mle)
    # get value representation for best alpha
    rwnorm_values = np.zeros([len(values)])
    rwnorm_ref_amount = np.zeros([len(ref_amount)])
    rwnorm_values[s_inds], rwnorm_ref_amount[s_inds] = compute_rw_update_behavior(values[s_inds], ref_amount[s_inds], rw_alpha_mle, sitem_ev0)
    rwnorm_values[b_inds], rwnorm_ref_amount[b_inds] = compute_rw_update_behavior(values[b_inds], ref_amount[b_inds], rw_alpha_mle, bundle_ev0)
    
    
    # rescorla wagner normalization by day
    day_array = df_all_choice['Day'].values
    nll_alphas = np.zeros(len(alphas))
    bic_alphas = np.zeros(len(alphas))
    ev0_type = 'Mean'
    for i,alp in enumerate(alphas):
        rwnorm_values_day = np.zeros([len(values)])
        rwnorm_ref_amount_day = np.zeros([len(ref_amount)])
        rwnorm_values_day[s_inds], rwnorm_ref_amount_day[s_inds] = compute_rw_update_by_day_behavior(values[s_inds], ref_amount[s_inds], alp, ev0_type, day_array[s_inds])
        rwnorm_values_day[b_inds], rwnorm_ref_amount_day[b_inds] = compute_rw_update_by_day_behavior(values[b_inds], ref_amount[b_inds], alp, ev0_type, day_array[b_inds])
        rw_nll_mle_temp, rw_beta_mle_temp, rw_bic_mle_temp = fit_softmax(betas, rwnorm_values_day, rwnorm_ref_amount_day, choice, num_params=2)
        nll_alphas[i] = rw_nll_mle_temp
        bic_alphas[i] = rw_bic_mle_temp
    rw_nll_mle = np.min(nll_alphas)
    rw_alpha_mle_day = alphas[np.argmin(nll_alphas)]
    rw_bic_mle = bic_alphas[np.argmin(nll_alphas)]
    print('RW By Day')
    print(rw_nll_mle)
    print(rw_alpha_mle_day)
    rw_day_softmax_nlls.append(rw_nll_mle)
    rw_day_softmax_bic.append(rw_bic_mle)
    # get value representation for best alpha
    rwnorm_values_day = np.zeros([len(values)])
    rwnorm_ref_amount_day = np.zeros([len(ref_amount)])
    rwnorm_values_day[s_inds], rwnorm_ref_amount_day[s_inds] = compute_rw_update_by_day_behavior(values[s_inds], ref_amount[s_inds], rw_alpha_mle_day, ev0_type, day_array[s_inds])
    rwnorm_values_day[b_inds], rwnorm_ref_amount_day[b_inds] = compute_rw_update_by_day_behavior(values[b_inds], ref_amount[b_inds], rw_alpha_mle_day, ev0_type, day_array[b_inds])
    
    # wtp - ref normalization
    wtp_minus_ref = values - ref_amount
    ref_zeros = ref_amount - ref_amount
    sbtrt_nll_mle, sbtrt_beta_mle, sbtrt_bic_mle = fit_softmax(betas, wtp_minus_ref, ref_zeros, choice, num_params=1)
    print('WTP - Ref')
    print(sbtrt_nll_mle)
    print(sbtrt_beta_mle)
    sbtrt_softmax_nlls.append(sbtrt_nll_mle)
    sbtrt_softmax_bic.append(sbtrt_bic_mle)
    
    # Compute the correlation between value representations
    corr_matrix = compute_value_correlations(values, norm_values, mc_values, rwnorm_values, rwnorm_values_day, wtp_minus_ref)
    corr_matrices.append(corr_matrix)
        

#df_nlls = pd.DataFrame(list(zip(choice_bias_nlls, softmax_nlls, norm_softmax_nlls, subs2test)),
#               columns =['Choice Bias', 'Softmax','Norm Softmax','Subj'])
#tidy_df = df_nlls.melt(id_vars='Subj',value_vars=['Choice Bias','Softmax','Norm Softmax'])

models = ['Softmax','Norm Softmax','MC Softmax', 'RW Softmax', 'RW By Day Softmax', 'WTP-Ref']
 
df_nlls = pd.DataFrame(list(zip(softmax_nlls, norm_softmax_nlls, mc_softmax_nlls, rw_softmax_nlls, rw_day_softmax_nlls, sbtrt_softmax_nlls, subs2test)),
               columns =['Softmax','Norm Softmax','MC Softmax', 'RW Softmax', 'RW By Day Softmax', 'WTP-Ref', 'Subj'])
tidy_df = df_nlls.melt(id_vars='Subj',value_vars=['Softmax','Norm Softmax','MC Softmax', 'RW Softmax', 'RW By Day Softmax', 'WTP-Ref'])

sns.barplot(x='Subj', y='value', hue='variable', data=tidy_df)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.show()

sns.barplot(x='variable', y='value', data=tidy_df, ci=None)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.show()

for model in models:
    temp_vals = df_nlls[model].values
    print(model)
    print(np.mean(temp_vals))
    print(np.std(temp_vals))
    print('\n')
    
# find how many subjects are best fit by each model
nll_array = np.array(df_nlls.values)[:,:-1]
nll_array = np.delete(nll_array, 3, axis=1)
best_model_inds = np.argmin(nll_array , axis=1)

plt.hist(best_model_inds)
plt.xticks(np.arange(4), ['Absolute Value', 'Divisive Norm. (DN)', 'DN Monte Carlo', 'DN Rescorla Wagner']) 
plt.title('Number of Subjects Best Fit by Model', fontsize=16)
plt.ylabel('Number of Subjects', fontsize=14)
plt.xlabel('Model', fontsize=16)

# Compute the average correlation matrix across subjects
avg_corr_matrix = sum(corr_matrices) / len(corr_matrices)

# Plot the average correlation matrix
plt.figure(figsize=(8, 6))
sns.heatmap(avg_corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Average Correlation Matrix of Value Representations')
plt.show()