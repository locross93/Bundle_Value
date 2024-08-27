#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:58:36 2022

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

def get_preceding_trial_streak(item_or_bundle, all_day_runs):
    assert all_day_runs.shape[0] == item_or_bundle.shape[0]
    # how many preceding trials were the same category (item or bundle)
    prec_trial_same_cat = np.zeros(len(item_or_bundle)).astype(int)
    for i,trial in enumerate(item_or_bundle):
        if i == 0:
            prec_trial_same_cat[i] = 0
        else:
            count = 0
            prec_ind = i - count
            while item_or_bundle[prec_ind] == trial and all_day_runs[prec_ind] == all_day_runs[i]:
                prec_trial_same_cat[i] = count
                count+=1
                prec_ind = i - count
                
    return prec_trial_same_cat

def get_trial_type_lag(item_or_bundle, all_day_runs, lag):
    runs = np.unique(all_day_runs)
    lag_item_or_bundle = 100*np.ones(len(item_or_bundle))
    for run in runs:
        run_inds = np.where(all_day_runs == run)[0]
        item_or_bundle_run = item_or_bundle[run_inds]
        lag_item_or_bundle[run_inds[lag:]] = item_or_bundle_run[:-lag]
        
    return lag_item_or_bundle
        
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

def get_choice_kernel(all_day_choices, all_day_runs, alpha):
    assert all_day_runs.shape[0] == all_day_choices.shape[0]
    choice_kernel = np.zeros(len(all_day_choices))
    prec_ind = -1
    for i,c in enumerate(all_day_choices):
        if i == 0 or all_day_runs[prec_ind] != all_day_runs[i]:
            choice_kernel[i] = 100
            ck = 0.5
        else:
            choice_kernel[i] = ck
        ck = ck + alpha*(c - ck)
        prec_ind+=1
        
    return choice_kernel    
            
cond_dict = {
	'Food item' : 1,
	'Trinket item' : 2,
	'Food bundle' : 3,
	'Trinket bundle' : 4,
	'Mixed bundle' : 5
	}
    

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

subs2test = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']

#subs2test = ['109']

#linear model intercept
intercept = True

df_item_choice = pd.DataFrame({'Choice':[0], 'Item_Value':[0], 'Ref_Pos':[0], 'Ref_Amount':[0], 'Cat_streak':[0], 'Subj': [0]})
df_bundle_choice = pd.DataFrame({'Choice':[0], 'Bundle_Value':[0], 'Item_Val1':[0], 'Item_Val2':[0], 
                                 'Ref_Pos':[0], 'Ref_Amount':[0], 'Cat_streak':[0], 'Subj': [0]})
df_all_choice = pd.DataFrame({'Choice':[0], 'Value':[0], 'Ref_Pos':[0], 'Ref_Amount':[0], 'Trial_type':[0], 'Subj': [0]})

subj_scores = {}
subj_scores['Item'] = []
subj_scores['Bundle'] = []
subj_scores['Bundle2'] = []
subj_scores['subID'] = []

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
        
        #item_or_bundle = (cond_nums_allruns > 2).astype(int)
        item_or_bundle = (cond_nums_allruns <= 2).astype(int)
        #prec_trial_same_cat = get_preceding_trial_streak(item_or_bundle, all_day_runs)
        lag_trial_type_list = []
        for lag in range(1,6):
            lag_trial_type_list.append(get_trial_type_lag(item_or_bundle, all_day_runs, lag))
        
        last_trial_choice = get_last_trial_choice(all_day_choices, all_day_runs)
        #choice_kernel = get_choice_kernel(all_day_choices, all_day_runs, alpha=0.5)
        #last_trial_choice = choice_kernel
        
        sitem_inds = np.where(cond_nums_allruns < 3)[0]
        bundle_inds = np.where(cond_nums_allruns > 2)[0]
        # get stimulus values
        stim_values = np.zeros(all_day_choices.shape[0])
        for i in range(len(all_day_choices)):
            #if item_or_bundle[i] == 0:
            if item_or_bundle[i] == 1:
                # sitem
                temp_item = all_day_items[i,0]
                match_ind = np.where(bdm_item == temp_item)[0]
                stim_values[i] = bdm_item_value[match_ind]
            #elif item_or_bundle[i] == 1:
            elif item_or_bundle[i] == 0:    
                # bundle
                temp_bundle = all_day_items[i,:]
                match_ind = np.where((bdm_bundle_items == temp_bundle).all(axis=1))[0]
                stim_values[i] = bdm_bundle_value[match_ind]
            
#        df_all_choice_day = pd.DataFrame({'Choice': all_day_choices, 'Value': stim_values, 
#                                        'Ref_Pos': all_day_ref_pos, 'Ref_Amount': all_day_ref_amount, 
#                                        'Cat_streak': prec_trial_same_cat, 'Prev_choice': last_trial_choice,
#                                        'Trial_type': item_or_bundle,
#                                        'Subj': [subID for i in range(len(all_day_choices))]})
    
        df_all_choice_day = pd.DataFrame({'Choice': all_day_choices, 'Value': stim_values, 
                                        'Ref_Pos': all_day_ref_pos, 'Ref_Amount': all_day_ref_amount, 
                                        'Lag1': lag_trial_type_list[0], 'Lag2': lag_trial_type_list[1],
                                        'Lag3': lag_trial_type_list[2], 'Lag4': lag_trial_type_list[3],
                                        'Lag5': lag_trial_type_list[4],
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
df_all_choice = df_all_choice[df_all_choice.Lag5 != 100].reset_index(drop=True)

#add column WTP - ref
wtp_minus_ref = df_all_choice['Value'].values - df_all_choice['Ref_Amount'].values
df_all_choice['WTP_v_Ref'] = wtp_minus_ref

formula_string = 'Choice ~ Value + Ref_Amount + Trial_type + Prev_choice + Value:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

formula_string = 'Choice ~ Value + Ref_Amount + Trial_type + Prev_choice + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Value:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

formula_string = 'Choice ~ WTP_v_Ref + Trial_type + Prev_choice + WTP_v_Ref:Trial_type'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

formula_string = 'Choice ~ WTP_v_Ref + Trial_type + Prev_choice + WTP_v_Ref:Trial_type'
#formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5'
#formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Lag1:Lag2 + Lag1:Lag2:Lag3 + Lag1:Lag2:Lag3:Lag4 + Lag1:Lag2:Lag3:Lag4:Lag5'
formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 '
#formula_string += ' + Trial_type:Lag1 + Trial_type:Lag2 + Trial_type:Lag3 + Trial_type:Lag4 + Trial_type:Lag5 '
formula_string += '+ Trial_type:Lag1 + Trial_type:Lag1:Lag2 + Trial_type:Lag1:Lag2:Lag3 +Trial_type: Lag1:Lag2:Lag3:Lag4 + Trial_type:Lag1:Lag2:Lag3:Lag4:Lag5'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

# f-test
A = np.identity(len(res_choice.params))
res_choice.f_test(A[5:,:])

# single items
df_item_choice = df_all_choice[df_all_choice.Trial_type == 0].reset_index(drop=True)

formula_string = 'Choice ~ WTP_v_Ref + Prev_choice'
#formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5'
formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Lag1:Lag2 + Lag1:Lag2:Lag3 + Lag1:Lag2:Lag3:Lag4 + Lag1:Lag2:Lag3:Lag4:Lag5'
res_choice = smf.logit(formula=formula_string, data=df_item_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

# f-test
A = np.identity(len(res_choice.params))
res_choice.f_test(A[3:,:])

# bundles
df_bundle_choice = df_all_choice[df_all_choice.Trial_type == 1].reset_index(drop=True)

formula_string = 'Choice ~ WTP_v_Ref + Prev_choice'
#formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5'
formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + Lag1:Lag2 + Lag1:Lag2:Lag3 + Lag1:Lag2:Lag3:Lag4 + Lag1:Lag2:Lag3:Lag4:Lag5'
#formula_string += ' + Lag1 + Lag2 + Lag3 + Lag4 + Lag5 + WTP_v_Ref:Lag1 + WTP_v_Ref:Lag1:Lag2 + WTP_v_Ref:Lag1:Lag2:Lag3 + WTP_v_Ref:Lag1:Lag2:Lag3:Lag4 + WTP_v_Ref:Lag1:Lag2:Lag3:Lag4:Lag5'
res_choice = smf.logit(formula=formula_string, data=df_bundle_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

# f-test
A = np.identity(len(res_choice.params))
res_choice.f_test(A[3:,:])
