#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 14:53:08 2022

@author: logancross
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
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

def confusion2accuracy(table):
    a = table[0,0]
    b = table[0,1]
    c = table[1,0]
    d = table[1,1]
    acc = float(a+d) / (a+b+c+d)
    
    return acc

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

def compute_divisive_norm(values, beta, sigma):
    avg_value = np.mean(values)
    div_normed_values = (beta + values.astype(float)) / (sigma + avg_value)
    
    return div_normed_values

cond_dict = {
	'Food item' : 1,
	'Trinket item' : 2,
	'Food bundle' : 3,
	'Trinket bundle' : 4,
	'Mixed bundle' : 5
	}

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

subs2test = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
subs2test = ['114']

df_all_choice = pd.DataFrame({'Choice':[0], 'Value':[0], 'Norm_Value':[0],
                              'Ref_Pos':[0], 'Ref_Amount':[0], 'Trial_type':[0], 'Prev_choice':[0], 'Subj': [0]})

for subID in subs2test:
    print '\n'
    print subID
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
        beta = 0
        sigma = 0
        dvnorm_values[sitem_inds] = compute_divisive_norm(stim_values[sitem_inds], beta, sigma) 
        dvnorm_values[bundle_inds] = compute_divisive_norm(stim_values[bundle_inds], beta, sigma)
        
        # wtp - ref
        wtp_ref_values = stim_values - all_day_ref_amount
        
        # wtp - ref / mean
        dvwr_norm_values = np.zeros([len(stim_values)])
        avg_svalue = np.mean(stim_values[sitem_inds])
        dvwr_norm_values[sitem_inds] = wtp_ref_values[sitem_inds].astype(float) / avg_svalue
        avg_bvalue = np.mean(stim_values[bundle_inds])
        dvwr_norm_values[bundle_inds] = wtp_ref_values[bundle_inds].astype(float) / avg_bvalue
        
        df_all_choice_day = pd.DataFrame({'Choice': all_day_choices, 'Value': stim_values, 'DV_Norm_Value': dvnorm_values,
                                        'WTP_Ref': wtp_ref_values, 'DV_WR_Norm_Value': dvwr_norm_values,
                                        'Ref_Pos': all_day_ref_pos, 'Ref_Amount': all_day_ref_amount, 
                                        'Trial_type': item_or_bundle,'Prev_choice': last_trial_choice,
                                        'Subj': [subID for i in range(len(all_day_choices))]})
    
        # drop error trials
        temp_choices = df_all_choice_day['Choice'].values
        err_inds = np.where(temp_choices == 100)[0]
        if len(err_inds) > 0:
            df_all_choice_day = df_all_choice_day[~df_all_choice_day.index.isin(err_inds)]
        assert len(np.unique(df_all_choice_day['Choice'])) == 2
        
        # make choices boolean
        df_all_choice_day.loc[:,'Choice'] = df_all_choice_day['Choice'].values.astype(bool)
        
        df_all_choice = pd.concat([df_all_choice, df_all_choice_day], ignore_index=True)
        
# drop first element
df_all_choice = df_all_choice.drop([0])     
df_all_choice = df_all_choice.reset_index(drop=True)  

# delete all elements with 100s
df_all_choice = df_all_choice[df_all_choice.Prev_choice != 100].reset_index(drop=True)

#formula_string = 'Choice ~ Value + Ref_Amount + Trial_type + Prev_choice + Value:Trial_type'
formula_string = 'Choice ~ Value + Trial_type + Prev_choice + Value:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

#formula_string = 'Choice ~ Norm_Value + Ref_Amount + Trial_type + Prev_choice + Value:Trial_type'
formula_string = 'Choice ~ DV_Norm_Value + Trial_type + Prev_choice + DV_Norm_Value:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

#formula_string = 'Choice ~ WTP_Ref + Trial_type + Prev_choice + WTP_Ref:Trial_type'
formula_string = 'Choice ~ DV_WR_Norm_Value + Trial_type + Prev_choice + DV_WR_Norm_Value:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice