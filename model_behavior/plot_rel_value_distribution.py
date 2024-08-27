#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:37:33 2022

@author: logancross
"""

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from statsmodels.formula.api import ols
import pandas as pd
import scipy

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

#can analyze one day or across all day
subID = '103'

#linear model intercept
intercept = True

save = False

if len(subID) == 5:
    #load item data
    sub_logs = log_folder+'bdm_items_sub_'+subID+'.mat'
    sub_data = loadmat(sub_logs)
    bdm_item_value_orig = sub_data['value'].reshape(-1)
    bdm_item_orig = sub_data['item'].reshape(-1)
    
    #load bundle data
    sub_logs_bun = log_folder+'bdm_bundle_sub_'+subID+'.mat'
    sub_data_bun = loadmat(sub_logs_bun)
    bdm_bundle_value_orig = sub_data_bun['value'].reshape(-1)
    bdm_bundle_orig = sub_data_bun['item']
    
    #find item values in bundle for regression
    bdm_bundle_item_values = np.zeros([len(bdm_bundle_orig), 2])
    for j in range(len(bdm_bundle_orig)):
        temp_bundle = bdm_bundle_orig[j,:]
        left_item_ind = np.where(bdm_item_orig == temp_bundle[0])
        bdm_bundle_item_values[j,0] = bdm_item_value_orig[left_item_ind]
        right_item_ind = np.where(bdm_item_orig == temp_bundle[1])
        bdm_bundle_item_values[j,1] = bdm_item_value_orig[right_item_ind]
elif len(subID) == 3:
    bdm_item_value_orig = []
    bdm_item_orig = []
    bdm_bundle_value_orig = []
    bdm_bundle_orig = []
    bdm_bundle_item_values = []
    for day in range(1,4):
        subID_temp = subID+'-'+str(day)
        #load item data
        sub_logs_temp = log_folder+'bdm_items_sub_'+subID_temp+'.mat'
        sub_data_temp = loadmat(sub_logs_temp)
        item_value = sub_data_temp['value'].reshape(-1)
        single_item = sub_data_temp['item'].reshape(-1)
        
        #load bundle data
        sub_logs_temp = log_folder+'bdm_bundle_sub_'+subID_temp+'.mat'
        sub_data_temp = loadmat(sub_logs_temp)
        bundle_value = sub_data_temp['value'].reshape(-1)
        bundle_items = sub_data_temp['item']
        
        #find item values in bundle for regression
        bundle_item_values_temp = np.zeros([len(bundle_value), 2])
        for j in range(len(bundle_value)):
            temp_bundle = bundle_items[j,:]
            left_item_ind = np.where(single_item == temp_bundle[0])
            bundle_item_values_temp[j,0] = item_value[left_item_ind]
            right_item_ind = np.where(single_item == temp_bundle[1])
            bundle_item_values_temp[j,1] = item_value[right_item_ind]
        
        #append to list for all days
        bdm_item_value_orig.append(item_value)
        bdm_item_orig.append(single_item)
        bdm_bundle_value_orig.append(bundle_value)
        bdm_bundle_orig.append(bundle_items)
        bdm_bundle_item_values.append(bundle_item_values_temp)
     
    #collapse across days
    bdm_item_value_orig = np.ravel(bdm_item_value_orig)
    bdm_item_orig = np.ravel(bdm_item_orig)
    bdm_bundle_value_orig = np.ravel(bdm_bundle_value_orig)
    bdm_bundle_orig = np.vstack(bdm_bundle_orig)
    bdm_bundle_item_values = np.vstack(bdm_bundle_item_values)
    
response_ind = np.where(bdm_item_value_orig != 100)[0]
bdm_item_value = bdm_item_value_orig[response_ind]
bdm_item = bdm_item_orig[response_ind]
bdm_item_category = np.zeros([len(bdm_item)]).astype(int)
trinket_inds = np.where(bdm_item >= 71)[0]
bdm_item_category[trinket_inds] = 1

response_ind = np.where(bdm_bundle_value_orig != 100)[0]
bdm_bundle_value = bdm_bundle_value_orig[response_ind]
bdm_bundle_items = bdm_bundle_orig[response_ind]
bdm_bundle_item_values = bdm_bundle_item_values[response_ind,:]
bdm_bundle_category = bdm_bundle_items > 71
bdm_bundle_category = bdm_bundle_category.astype(int)

all_values = [bdm_item_value, bdm_bundle_value]
plt.hist(all_values, bins=[bin for bin in np.arange(21)], density=True, edgecolor='white', color=['tomato','dodgerblue'], label=['Single Item','Bundle'])
plt.xticks(np.arange(0, 21))
plt.ylabel('Probability', fontsize=16)
plt.xlabel('WTP Bid ($)', fontsize=16)
plt.title('Absolute Value', fontsize=20)
plt.legend()
if save:
    plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/abs_value_dist_example', dpi=500)
plt.show()

z_values = [scipy.stats.zscore(bdm_item_value), scipy.stats.zscore(bdm_bundle_value)]
plt.hist(z_values, bins=15, density=True, edgecolor='white', color=['tomato','dodgerblue'], label=['Single Item','Bundle'])
plt.ylabel('Probability', fontsize=16)
plt.xlabel('Value z-scored by condition', fontsize=16)
plt.title('Relative Value', fontsize=20)
plt.legend()
if save:
    plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/rel_value_dist_example', dpi=500)
plt.show()

dvnorm_item_values = (bdm_item_value.astype(float) - np.mean(bdm_item_value))/ np.mean(bdm_item_value)
dvnorm_bundle_values = (bdm_bundle_value.astype(float) - np.mean(bdm_item_value))/ np.mean(bdm_bundle_value)
dvnorm_values = [dvnorm_item_values, dvnorm_bundle_values]
plt.hist(dvnorm_values, density=True, edgecolor='white', color=['tomato','dodgerblue'], label=['Single Item','Bundle'])
plt.ylabel('Probability', fontsize=16)
plt.xlabel('Divisively Normalized Value', fontsize=16)
plt.title('Divisively Normalized Value', fontsize=20)
plt.legend()
plt.show()
