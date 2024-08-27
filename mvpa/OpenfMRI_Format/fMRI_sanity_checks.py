#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 16:44:32 2020

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

log_folder = '/Users/logancross/Documents/Bundle_Value/stim_presentation/Bundles_fMRI/logs/'

#can analyze one day or across all day
subID = '101'

#linear model intercept
intercept = True

if len(subID) == 5:
    sub_logs = log_folder+'bdm_items_sub_'+subID+'.mat'
    sub_data = loadmat(sub_logs)
    bdm_item_value_orig = sub_data['value'].reshape(-1)
    response_ind = np.where(bdm_item_value_orig != 100)[0]
    bdm_item_value = bdm_item_value_orig[response_ind]
    bdm_item_orig = sub_data['item'].reshape(-1)
    bdm_item = bdm_item_orig[response_ind]
    bdm_category = np.zeros([len(bdm_item)]).astype(int)
    trinket_inds = np.where(bdm_item >= 71)[0]
    bdm_category[trinket_inds] = 1
elif len(subID) == 3:
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
    
category_dict = {'0': 'food',
                 '1': 'trinket'
                 }
category_strings = [category_dict[str(cat)] for cat in bdm_category]

food_value = bdm_item_value[np.where(bdm_category == 0)[0]]
trinket_value = bdm_item_value[np.where(bdm_category == 1)[0]]
temp_values = np.column_stack((food_value, trinket_value))
plt.hist(temp_values, bins=[bin for bin in np.arange(21)], edgecolor='white', label=['Food','Trinket'])
plt.xticks(np.arange(0, 21))
plt.ylabel('Count')
plt.xlabel('Value')
plt.title('WTP Bids - Individual Items')
plt.legend()
plt.show()

#load bundle data
if len(subID) == 5:
    sub_logs_bun = log_folder+'bdm_bundle_sub_'+subID+'.mat'
    sub_data_bun = loadmat(sub_logs_bun)
    bdm_bundle_value_orig = sub_data_bun['value'].reshape(-1)
    bdm_bundle_orig = sub_data_bun['item']
elif len(subID) == 3:
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

#0: food 1: mixed 2: trinket
bundle_type = np.array([np.sum(items) for items in bdm_bundle_category])
food_bundle = bdm_bundle_value[np.where(bundle_type == 0)[0]]
trinket_bundle = bdm_bundle_value[np.where(bundle_type == 2)[0]]
mixed_bundle = bdm_bundle_value[np.where(bundle_type == 1)[0]]
temp_bundle_values = [food_bundle, trinket_bundle, mixed_bundle]
plt.hist(temp_bundle_values, bins=[bin for bin in np.arange(21)], edgecolor='white', label=['Food Bundle','Trinket Bundle','Mixed Bundle'])
#plt.hist(food_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Food Bundle')
#plt.hist(trinket_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Trinket Bundle')
#plt.hist(mixed_bundle, bins=[bin for bin in np.arange(21)], edgecolor='white', label='Mixed Bundle')
plt.xticks(np.arange(0, 21))
plt.ylabel('Count')
plt.xlabel('Value')
plt.title('WTP Bids - Bundles')
plt.legend()
plt.show()

#Assuming that first column is left item and 2nd column is right item.
#Linear regression across left (x1) and right (x2) item
#Bundle value=B1*x1+B2*x2+C
temp_list1 = []
temp_list2 = []
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

#fit model
if intercept:
    bundle_X = sm.add_constant(bundle_item_values)
else:
    bundle_X = bundle_item_values
    
data = pd.DataFrame({"Bundle_Value":bdm_bundle_value, "Val_Item1":bundle_item_values[:,0], "Val_Item2":bundle_item_values[:,1]})
model = ols(formula = 'Bundle_Value ~ Val_Item1 + Val_Item2', data = data).fit()
#model = ols(formula = 'Bundle_Value ~ Val_Item1 + Val_Item2 + np.power(Val_Item1,2) + np.power(Val_Item2,2)', data = data).fit()
#model = ols(formula = 'Bundle_Value ~ Val_Item1 + Val_Item2 + Val_Item1*Val_Item2', data = data).fit()
print model.summary()
r2 = model.rsquared
betas = model.params 
predicted_values = model.fittedvalues

#model = sm.OLS(bdm_bundle_value, bundle_X)   
#results = model.fit()
#print results.summary()
#r2 = results.rsquared
#betas = results.params 
#predicted_values = results.fittedvalues

fig, ax = plt.subplots()
ax.scatter(predicted_values, bdm_bundle_value)
ax.plot(bdm_bundle_value, bdm_bundle_value, color='r')
plt.xlabel('Predicted value from LM')
plt.ylabel('Actual value')
plt.title('Left vs Right Regression - Sub'+subID)
plt.text(0.8,0.15,'R2 value: '+'%.3f'%r2+
         '\nBeta1 (intercept): '+'%.3f'%betas[0]+'\nBeta2 (Left): '+'%.3f'%betas[1]+
         '\nBeta2 (Right): '+'%.3f'%betas[2], horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
plt.show()

#plot histograms of actual data and predicted data
temp_values = np.column_stack((bdm_bundle_value, predicted_values))
plt.hist(temp_values, bins=[bin for bin in np.arange(21)], edgecolor='white', label=['Data','Predictions'])
plt.xticks(np.arange(0, 21))
plt.ylabel('Count')
plt.xlabel('Value')
plt.legend()
plt.title('Predicted Bundle Bids from Model - Sub'+subID)
plt.show()
    
#Linear regression across food (x1) and trinket item (x2) for mixed bundles
#Bundle value=B1*x1+B2*x2+C
mixed_inds = np.where(bundle_type == 1)[0]
mixedbundle_value = bdm_bundle_value[mixed_inds]
mixed_items = np.sort(bdm_bundle_items[mixed_inds,:])
mixedbundle_item_values = np.zeros([mixed_items.shape[0], mixed_items.shape[1]]).astype(int)
for j in range(2):
    for i in range(mixed_items.shape[0]):
        assert np.max(bdm_item==mixed_items[i,j]) == True, 'Bundle item not in ind item WTP'
        temp = bdm_item_value[bdm_item==mixed_items[i,j]]
        #for items valued every day, take the value from the same day
        if len(temp) > 1:
            assert len(temp) == 3
            if i < 210:
                mixedbundle_item_values[i,j]=temp[0]
            elif i < 420:
                mixedbundle_item_values[i,j]=temp[1]
            else:
                mixedbundle_item_values[i,j]=temp[2]
        
#fit model
if intercept:
    mix_bundle_X = sm.add_constant(mixedbundle_item_values)
else:
    mix_bundle_X = mixedbundle_item_values
model2 = sm.OLS(mixedbundle_value, mix_bundle_X) 
results2 = model2.fit()
print results2.summary()
mix_r2 = results2.rsquared
mix_betas = results2.params 
mix_predicted_values = results2.fittedvalues

fig, ax = plt.subplots()
ax.scatter(mix_predicted_values, mixedbundle_value)
ax.plot(mixedbundle_value, mixedbundle_value, color='r')
plt.xlabel('Predicted value from LM')
plt.ylabel('Actual value')
plt.title('Food vs Trinket Regression - Sub'+subID)
plt.text(0.8,0.15,'R2 value: '+'%.3f'%mix_r2+
         '\nBeta1 (intercept): '+'%.3f'%mix_betas[0]+'\nBeta2 (Food): '+'%.3f'%mix_betas[1]+
         '\nBeta2 (Trinket): '+'%.3f'%mix_betas[2], horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
plt.show()

#plot histograms of actual data and predicted data
temp_values = np.column_stack((mixedbundle_value, mix_predicted_values))
plt.hist(temp_values, bins=[bin for bin in np.arange(21)], edgecolor='white', label=['Data','Predictions'])
plt.xticks(np.arange(0, 21))
plt.ylabel('Count')
plt.xlabel('Value')
plt.legend()
plt.title('Predicted Bundle Bids from Model - Sub'+subID)
plt.show()

#plot bundle value vs sum of individual values
sum_of_values = np.sum(bundle_item_values, axis=1)
linear_add_r2 = r2_score(bdm_bundle_value, sum_of_values)

fig, ax = plt.subplots()
ax.scatter(sum_of_values, bdm_bundle_value)
ax.plot(sum_of_values, sum_of_values, color='r')
plt.xlim([0,20])
plt.xticks(np.arange(0, 21, 2))
plt.xlabel('Value of Sum of Individual Item Values')
plt.ylim([0,20])
plt.yticks(np.arange(0, 21, 2))
plt.ylabel('Bundle Value')
plt.title('Bundle Value vs. Linear Sum - Sub'+subID)
plt.text(0.8,0.15,'R^2: '+'%.3f'%linear_add_r2, horizontalalignment='center',
         verticalalignment='center', transform=ax.transAxes)
plt.show()