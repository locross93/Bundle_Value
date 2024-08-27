#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:47:11 2021

@author: logancross
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
import pandas as pd

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

subs2test = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']

#subs2test = ['101']

#linear model intercept
intercept = True

df_data = pd.DataFrame({'Bundle_Value':[0], 'Val_Item1':[0], 'Val_Item2':[0], 'Subj': [0]})
subj_scores = []

for subID in subs2test:
    #load item data
    bdm_item_value_orig = []
    bdm_item_orig = []
    for day in range(1,4):
        subID_temp = subID+'-'+str(day)
        sub_logs_temp = log_folder+'bdm_items_sub_'+subID_temp+'.mat'
        sub_data_temp = loadmat(sub_logs_temp)
        bdm_item_value_orig.append(sub_data_temp['value'].reshape(-1))
        bdm_item_orig.append(sub_data_temp['item'].reshape(-1))
    bdm_item_value_orig = np.ravel(bdm_item_value_orig)
    bdm_item_orig = np.ravel(bdm_item_orig)
    response_ind = np.where(bdm_item_value_orig != 100)[0]
    bdm_item_value = bdm_item_value_orig[response_ind]
    bdm_item = bdm_item_orig[response_ind]
    bdm_category = np.zeros([len(bdm_item)]).astype(int)
    trinket_inds = np.where(bdm_item > 71)[0]
    bdm_category[trinket_inds] = 1
    
    #load bundle data
    bdm_bundle_value_orig = []
    bdm_bundle_orig = []
    for day in range(1,4):
        subID_temp = subID+'-'+str(day)
        sub_logs_temp = log_folder+'bdm_bundle_sub_'+subID_temp+'.mat'
        sub_data_temp = loadmat(sub_logs_temp)
        bdm_bundle_value_orig.append(sub_data_temp['value'].reshape(-1))
        bdm_bundle_orig.append(sub_data_temp['item'])
    bdm_bundle_value_orig = np.ravel(bdm_bundle_value_orig)
    bdm_bundle_orig = np.vstack(bdm_bundle_orig)
    response_ind = np.where(bdm_bundle_value_orig != 100)[0]
    bdm_bundle_value = bdm_bundle_value_orig[response_ind]
    bdm_bundle_items = bdm_bundle_orig[response_ind]
    bdm_bundle_category = bdm_bundle_items > 71
    bdm_bundle_category = bdm_bundle_category.astype(int)
    
    #Assuming that first column is left item and 2nd column is right item.
    #Linear regression across left (x1) and right (x2) item
    #Bundle value=B1*x1+B2*x2+C
    bundle_item_values = np.zeros([bdm_bundle_items.shape[0], bdm_bundle_items.shape[1]]).astype(int)
    for j in range(2):
        for i in range(bdm_bundle_items.shape[0]):
            assert np.max(bdm_item==bdm_bundle_items[i,j]) == True, 'Bundle item not in ind item WTP'            
            temp = bdm_item_value[bdm_item==bdm_bundle_items[i,j]]
            #for items valued every day, take the value from the same day
            if len(temp) > 1:
                assert len(temp) == 3
                if i < 210:
                    bundle_item_values[i,j]=temp[0]
                elif i < 420:
                    bundle_item_values[i,j]=temp[1]
                else:
                    bundle_item_values[i,j]=temp[2]
            else:
                bundle_item_values[i,j]=temp[0]
            
    temp_df = pd.DataFrame({'Bundle_Value':bdm_bundle_value, 'Val_Item1':bundle_item_values[:,0], 'Val_Item2':bundle_item_values[:,1], 'Subj': [int(subID) for j in range(len(bdm_bundle_value))]})
    model_indiv = ols(formula = 'Bundle_Value ~ Val_Item1 + Val_Item2', data = temp_df).fit()
    subj_scores.append(model_indiv.rsquared)
    
    df_data = pd.concat([df_data, temp_df],ignore_index=True)
df_data = df_data.drop([0])

md = smf.mixedlm('Bundle_Value ~ Val_Item1 + Val_Item2', df_data, groups=df_data['Subj'], re_formula='~ Val_Item1 + Val_Item2')
#df_X = df_data[['Val_Item1','Val_Item2']]
#df_X = sm.add_constant(df_X)
#md = sm.MixedLM(df_data['Bundle_Value'], df_X, df_data['Subj'], df_X)
mdf = md.fit()
print(mdf.summary())

model1 = ols(formula = 'Bundle_Value ~ Val_Item1 + Val_Item2', data = df_data).fit()
print model1.summary()

plt.bar(np.arange(1,len(subj_scores)+1), subj_scores, color='g')
plt.xticks(np.arange(1,len(subj_scores)+1))
plt.xlabel('Subject', fontsize=16)
plt.ylabel('R-squared', fontsize=16)
plt.title('Linear Model Fit to Individual Subject', fontsize=16)

#plot bundle value vs sum of individual values
sum_of_values = np.sum(df_data[['Val_Item1','Val_Item2']], axis=1)
bundle_value_allsubjs = np.array(df_data['Bundle_Value'])
linear_add_r2 = r2_score(bundle_value_allsubjs, sum_of_values)

fig, ax = plt.subplots()
ax.scatter(sum_of_values, bundle_value_allsubjs)
ax.plot(sum_of_values, sum_of_values, color='r')
plt.xlim([0,20])
plt.xticks(np.arange(0, 21, 2))
plt.xlabel('Value of Sum of Individual Item Values')
plt.ylim([0,20])
plt.yticks(np.arange(0, 21, 2))
plt.ylabel('Bundle Value')
plt.title('Bundle Value vs. Linear Sum')
plt.text(0.8,0.1,'R^2: '+'%.3f'%linear_add_r2, horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
plt.show()