#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 16:29:10 2022

@author: logancross
"""

# plot WTP values vs their predicted value for each category separately - train on bundle
fit_intercept = False
plt.scatter(b2b_real_bundle_vals, b2b_pred_bundle_vals, c='b', label="Bundle")
plt.scatter(b2s_real_item_vals, b2s_pred_item_vals, c='r', label="Single Item")
# plot line of best fit within category
#item_slope, item_b, r, p, se = linregress(s2s_real_item_vals, s2s_pred_item_vals)
#bundle_slope, bundle_b, r, p, se = linregress(s2b_real_bundle_vals, s2b_pred_bundle_vals)
reg_item = LinearRegression(fit_intercept=fit_intercept).fit(b2s_real_item_vals.reshape(-1, 1), b2s_pred_item_vals.reshape(-1, 1))
item_slope = reg_item.coef_[0][0]
if fit_intercept:
    item_b = reg_item.intercept_[0]
else:
    item_b = 0
reg_bundle = LinearRegression(fit_intercept=fit_intercept).fit(b2b_real_bundle_vals.reshape(-1, 1), b2b_pred_bundle_vals.reshape(-1, 1))
bundle_slope = reg_bundle.coef_[0][0]
if fit_intercept:
    bundle_b = reg_bundle.intercept_[0]
else:
    bundle_b = 0
x_item = np.arange(0,np.max(b2s_real_item_vals)+1)
x_bundle = np.arange(0,np.max(b2b_real_bundle_vals)+1)
plt.plot(x_item, (item_slope*x_item+item_b), 'r-')
plt.plot(x_bundle, (bundle_slope*x_bundle+bundle_b), 'b-')
plt.xlabel('WTP Bid')
plt.ylabel('Predicted WTP Bid')
plt.title(mask_label+' Cross Validated Predictions - Train on Bundle Sub'+subj)
plt.legend(bbox_to_anchor=(1.3, 1.03))
plt.show()



unique_bundle_vals = np.unique(b2b_real_bundle_vals)
avg_b2b_pred_bundle_vals = np.zeros([len(unique_bundle_vals)])

for i,bval in enumerate(unique_bundle_vals):
    temp_inds = np.where(b2b_real_bundle_vals == bval)[0]
    avg_b2b_pred_bundle_vals[i] = np.mean(b2b_pred_bundle_vals[temp_inds])
    
unique_item_vals = np.unique(b2s_real_item_vals)
avg_b2s_pred_item_vals = np.zeros([len(unique_item_vals)])

for i,ival in enumerate(unique_item_vals):
    temp_inds = np.where(b2s_real_item_vals == ival)[0]
    avg_b2s_pred_item_vals[i] = np.mean(b2s_pred_item_vals[temp_inds])
    
plt.scatter(unique_bundle_vals, avg_b2b_pred_bundle_vals, c='b', label="Bundle")
plt.scatter(unique_item_vals, avg_b2s_pred_item_vals, c='r', label="Single Item")
plt.xlabel('WTP Bid')
plt.ylabel('Predicted WTP Bid')
plt.title(mask_label+' Cross Validated Predictions - Train on Bundle Sub'+subj)
plt.legend(bbox_to_anchor=(1.3, 1.03))
plt.show()