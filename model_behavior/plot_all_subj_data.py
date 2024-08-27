#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 21:58:21 2021

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

all_sub_item_value = []
all_sub_item_category = []
all_sub_bundle_value = []
all_sub_bundle_category = []
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
    
    all_sub_item_value.append(bdm_item_value)
    all_sub_item_category.append(bdm_category)
    
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
    
    all_sub_bundle_value.append(bdm_bundle_value)
    all_sub_bundle_category.append(bdm_bundle_category)
    
all_sub_item_value = np.concatenate((all_sub_item_value))
all_sub_item_category = np.concatenate((all_sub_item_category))

food_value = all_sub_item_value[np.where(all_sub_item_category == 0)[0]]
trinket_value = all_sub_item_value[np.where(all_sub_item_category == 1)[0]]
temp_values = np.column_stack((food_value, trinket_value))
plt.hist(temp_values, bins=[bin for bin in np.arange(21)], density=True, edgecolor='white', label=['Food','Trinket'])
plt.xticks(np.arange(0, 21))
plt.ylabel('Probability', fontsize=16)
plt.xlabel('Value', fontsize=16)
plt.title('WTP Bids - Individual Items', fontsize=16)
plt.legend()
plt.show()

all_sub_bundle_value = np.concatenate((all_sub_bundle_value))
all_sub_bundle_category = np.concatenate((all_sub_bundle_category))

#0: food 1: mixed 2: trinket
bundle_type = np.array([np.sum(items) for items in all_sub_bundle_category])
food_bundle = all_sub_bundle_value[np.where(bundle_type == 0)[0]]
trinket_bundle = all_sub_bundle_value[np.where(bundle_type == 2)[0]]
mixed_bundle = all_sub_bundle_value[np.where(bundle_type == 1)[0]]
temp_bundle_values = [food_bundle, trinket_bundle, mixed_bundle]
plt.hist(temp_bundle_values, bins=[bin for bin in np.arange(21)], density=True, edgecolor='white', label=['Food Bundle','Trinket Bundle','Mixed Bundle'])
#plt.hist(food_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Food Bundle')
#plt.hist(trinket_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Trinket Bundle')
#plt.hist(mixed_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Mixed Bundle')
plt.xticks(np.arange(0, 21))
plt.ylabel('Probability', fontsize=16)
plt.xlabel('Value', fontsize=16)
plt.title('WTP Bids - Bundles', fontsize=16)
plt.legend()
plt.show()