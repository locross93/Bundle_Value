#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 12:34:44 2022

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
df_all_choice = pd.DataFrame({'Choice':[0], 'Value':[0], 'Ref_Pos':[0], 'Ref_Amount':[0], 'Cat_streak':[0], 'Trial_type':[0], 'Subj': [0]})

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
        
        item_or_bundle = (cond_nums_allruns > 2).astype(int)
        prec_trial_same_cat = get_preceding_trial_streak(item_or_bundle, all_day_runs)
        
        last_trial_choice = get_last_trial_choice(all_day_choices, all_day_runs)
        #choice_kernel = get_choice_kernel(all_day_choices, all_day_runs, alpha=0.5)
        #last_trial_choice = choice_kernel
        
        sitem_inds = np.where(cond_nums_allruns < 3)[0]
        bundle_inds = np.where(cond_nums_allruns > 2)[0]
        # get item values
        item_values = np.zeros(sitem_inds.shape[0])
        sitem_ids = all_day_items[sitem_inds,0]
        for i,sid in enumerate(sitem_ids):
            match_ind = np.where(bdm_item == sid)[0]
            item_values[i] = bdm_item_value[match_ind]
            
        df_item_choice_day = pd.DataFrame({'Choice': all_day_choices[sitem_inds], 'Item_Value': item_values, 
                                        'Ref_Pos': all_day_ref_pos[sitem_inds], 'Ref_Amount': all_day_ref_amount[sitem_inds], 
                                        'Cat_streak': prec_trial_same_cat[sitem_inds], 'Prev_choice': last_trial_choice[sitem_inds],
                                        'Subj': [subID for i in range(len(sitem_inds))]})
    
        # drop error trials
        temp_choices = df_item_choice_day['Choice'].values
        err_inds = np.where(temp_choices == 100)[0]
        if len(err_inds) > 0:
            df_item_choice_day = df_item_choice_day[~df_item_choice_day.index.isin(err_inds)]
        assert len(np.unique(df_item_choice_day['Choice'])) == 2
    
        df_item_choice_day.loc[:,'Choice'] = df_item_choice_day['Choice'].values.astype(bool)
        df_item_choice = pd.concat([df_item_choice, df_item_choice_day], ignore_index=True)
        
        df_all_choice_temp = df_item_choice_day.rename(columns={'Item_Value': 'Value'})
        #cat_column = [int(-1) for i in range(len(df_all_choice_temp))]
        cat_column = [int(0) for i in range(len(df_all_choice_temp))]
        df_all_choice_temp['Trial_type'] = cat_column 
        df_all_choice = pd.concat([df_all_choice, df_all_choice_temp], ignore_index=True)
            
        # get bundle values
        bundle_values = np.zeros(bundle_inds.shape[0])
        bundle_item_values = np.zeros([bundle_inds.shape[0],2])
        bundle_ids = all_day_items[bundle_inds,:]
        for i,bids in enumerate(bundle_ids):
            match_ind = np.where((bdm_bundle_items == bids).all(axis=1))[0]
            assert bdm_bundle_items[match_ind,0] == bids[0] and bdm_bundle_items[match_ind,1] == bids[1]
            bundle_values[i] = bdm_bundle_value[match_ind]
            for j,bid in enumerate(bids):
                match_ind = np.where(bdm_item == bid)[0]
                bundle_item_values[i,j] = bdm_item_value[match_ind]
        # sort bundle_item_values so highest value item is in first column
        sort_bundle_item_vals = -np.sort(-bundle_item_values, axis=1)
        
        df_bundle_choice_day = pd.DataFrame({'Choice': all_day_choices[bundle_inds], 'Bundle_Value': bundle_values, 
                                        'Item_Val1': sort_bundle_item_vals[:,0], 'Item_Val2': sort_bundle_item_vals[:,1],
                                        'Ref_Pos': all_day_ref_pos[bundle_inds], 'Ref_Amount': all_day_ref_amount[bundle_inds], 
                                        'Cat_streak': prec_trial_same_cat[bundle_inds], 'Prev_choice': last_trial_choice[bundle_inds],
                                        'Subj': [subID for i in range(len(bundle_inds))]})
    
        # drop error trials
        temp_choices = df_bundle_choice_day['Choice'].values
        err_inds = np.where(temp_choices == 100)[0]
        if len(err_inds) > 0:
            df_bundle_choice_day = df_bundle_choice_day[~df_bundle_choice_day.index.isin(err_inds)]
        assert len(np.unique(df_bundle_choice_day['Choice'])) == 2
    
        df_bundle_choice_day.loc[:,'Choice'] = df_bundle_choice_day['Choice'].values.astype(bool)
        
        df_bundle_choice = pd.concat([df_bundle_choice, df_bundle_choice_day], ignore_index=True)
        
        df_all_choice_temp = df_bundle_choice_day.rename(columns={'Bundle_Value': 'Value'})
        df_all_choice_temp = df_all_choice_temp.drop(['Item_Val1', 'Item_Val2'], axis=1)
        cat_column = [int(1) for i in range(len(df_all_choice_temp))]
        df_all_choice_temp['Trial_type'] = cat_column 
        df_all_choice = pd.concat([df_all_choice, df_all_choice_temp], ignore_index=True)
        
# drop first element
df_all_choice = df_all_choice.drop([0])     
df_all_choice = df_all_choice.reset_index(drop=True)  

# one hot encode subject and category streak
y = pd.get_dummies(df_all_choice.Cat_streak, prefix='Streak')
df_all_choice = pd.concat([df_all_choice, y], axis=1)

y = pd.get_dummies(df_all_choice.Subj, prefix='Subj')
df_all_choice = pd.concat([df_all_choice, y], axis=1)

# get Xs and ys
df_columns = df_all_choice.columns.values.tolist()
df_columns.remove('Cat_streak')
df_columns.remove('Subj')
df_columns.remove('Choice')
xcols = df_columns
ycol = ['Choice']

xcols = ['Value','Ref_Amount','Trial_type','Cat_streak']
Xtrain = df_all_choice[xcols]
ytrain = df_all_choice[ycol].values

#add column WTP - ref
wtp_minus_ref = df_all_choice['Value'].values - df_all_choice['Ref_Amount'].values
df_all_choice['WTP_v_Ref'] = wtp_minus_ref
  
# building the model and fitting the data
#res_choice = sm.Logit(ytrain, Xtrain).fit()      
#res_choice = smf.logit(formula='Choice ~ Value + Ref_Amount + Trial_type + Cat_streak + Ref_Amount:Trial_type', data=df_all_choice).fit() 
formula_string = 'Choice ~ WTP_v_Ref + Trial_type + Cat_streak + WTP_v_Ref:Trial_type + WTP_v_Ref:Cat_streak + Trial_type:Cat_streak'
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

formula_string = 'Choice ~ WTP_v_Ref + Trial_type + WTP_v_Ref:Trial_type'
#for subj in subs2test:
#    formula_string = formula_string + ' + Subj_'+subj
for strk in range(4):
    formula_string = formula_string + ' + Streak_'+str(strk)
    formula_string = formula_string + ' + WTP_v_Ref:Streak_'+str(strk)
    formula_string = formula_string + ' + Trial_type:Streak_'+str(strk)
res_choice = smf.logit(formula=formula_string, data=df_all_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

# previous choice
temp_choices = df_all_choice['Prev_choice'].values
err_inds = np.where(temp_choices == 100)[0]
if len(err_inds) > 0:
    df_prev_choice = df_all_choice[~df_all_choice.index.isin(err_inds)]
#assert len(np.unique(df_prev_choice['Prev_choice'])) == 2

formula_string = 'Choice ~ WTP_v_Ref + Trial_type + WTP_v_Ref:Trial_type + Prev_choice + WTP_v_Ref:Prev_choice + Trial_type:Prev_choice'
#for subj in subs2test:
#    formula_string = formula_string + ' + Subj_'+subj
#for strk in range(4):
#    formula_string = formula_string + ' + Streak_'+str(strk)
#    formula_string = formula_string + ' + Prev_choice:Streak_'+str(strk)
res_choice = smf.logit(formula=formula_string, data=df_prev_choice).fit()
print res_choice.summary()
pred_table = res_choice.pred_table()
acc_choice = confusion2accuracy(pred_table)        
print acc_choice

    
########Separate models for single item trials and bundle trials
##drop error trials and first element
#df_item_choice = df_item_choice.drop([0])
#sitem_choices = df_item_choice['Choice'].values
#err_inds = np.where(sitem_choices == 100)[0]
#if len(err_inds) > 0:
#    df_item_choice = df_item_choice[~df_item_choice.index.isin(err_inds)]
#assert len(np.unique(df_item_choice['Choice'])) == 2
#
#y = pd.get_dummies(df_item_choice.Cat_streak, prefix='Streak')
#
#df_item_choice = pd.concat([df_item_choice, y], axis=1)
#
#df_bundle_choice = df_bundle_choice.drop([0])
#bundle_choices = df_bundle_choice['Choice'].values
#err_inds = np.where(bundle_choices == 100)[0]
#if len(err_inds) > 0:
#    df_bundle_choice = df_bundle_choice[~df_bundle_choice.index.isin(err_inds)]
#assert len(np.unique(df_bundle_choice['Choice'])) == 2
#
#y = pd.get_dummies(df_bundle_choice.Cat_streak, prefix='Streak')
#
#df_bundle_choice = pd.concat([df_bundle_choice, y], axis=1)
#
## defining the dependent and independent variables
#xcols = ['Item_Value','Ref_Amount','Streak_0.0','Streak_1.0','Streak_2.0','Streak_3.0','Streak_4.0']
#ycol = ['Choice']
#
#Xtrain = df_item_choice[xcols]
#ytrain = df_item_choice[ycol]
#  
## building the model and fitting the data
#res_item = sm.Logit(ytrain, Xtrain).fit()       
##res_item = smf.logit(formula='Choice ~ Item_Value + Ref_Amount + Streak_0.0 + Streak_1.0 + Streak_2.0 + Streak_3.0 + Streak_4.0', data=df_item_choice).fit()        
#print res_item.summary()
#pred_table = res_item.pred_table()
#acc_item = confusion2accuracy(pred_table)        
#print acc_item    
#subj_scores['Item'].append(acc_item)
#
#xcols = ['Bundle_Value','Ref_Amount','Streak_0.0','Streak_1.0','Streak_2.0','Streak_3.0','Streak_4.0']
#ycol = ['Choice']
#
#Xtrain = df_bundle_choice[xcols]
#ytrain = df_bundle_choice[ycol]
#
#res_bun = sm.Logit(ytrain, Xtrain).fit()
##res_bun = smf.logit(formula='Choice ~ Bundle_Value + Ref_Amount + Streak_0.0 + Streak_1.0 + Streak_2.0 + Streak_3.0 + Streak_4.0', data=df_bundle_choice).fit()        
#print res_bun.summary()
#pred_table = res_bun.pred_table()
#acc_bun = confusion2accuracy(pred_table)        
#print acc_bun
#subj_scores['Bundle'].append(acc_bun)
#
#res_bun = smf.logit(formula='Choice ~ Bundle_Value + Ref_Amount + Cat_streak', data=df_bundle_choice).fit()        
#print res_bun.summary()
#pred_table = res_bun.pred_table()
#acc_bun2 = confusion2accuracy(pred_table)        
#print acc_bun2
#subj_scores['Bundle2'].append(acc_bun2)
#
#subj_scores['subID'].append(subID)
#    
#df_plot = pd.DataFrame.from_dict(subj_scores)
#
##df_plot = df_data.melt(id_vars=index)
##ax = sns.barplot(x='ROI', y='value', data=df_plot)
#
#df_plot.plot(x='subID', y=["Item", "Bundle", "Bundle2"], kind="bar")
#plt.xticks(rotation=0)
#plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
#plt.title('Logistic Regression on Choice Task', fontsize=18)
#plt.ylabel('Accuracy', fontsize=18)
#plt.xlabel('Subject', fontsize=18)
#plt.show()