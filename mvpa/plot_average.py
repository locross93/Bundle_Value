#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 13:45:08 2024

@author: ryanwebb
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from lmfit.model import load_modelresult

#bundle_path = '/Users/ryanwebb/Documents/GitHub/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

subj_list = ['101', '102', '103', '104','107','108','109','110','111','112','113','114']

all_data = []

for subj in subj_list:
    subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(subj))
    # Define the path to the JSON file
    results_file = os.path.join(subj_dir, 'rsa_norm_results_11_07_24.pkl')
    
    # Open and load the JSON file
    with open(results_file, 'rb') as f:
        #results_dict = json.load(f)
        results_dict = pickle.load(f)
        
    # Loop through each brain region and model to extract the adj_r2
    #for region, models in results_dict.items():
    #    for model, values in models.items():
            
    for model in results_dict.keys():
        for mask, results in results_dict[model].items():
            # Append the results to the all_data list
            
            all_data.append({
                'Subject': subj,
                'Mask': mask,
                'Model': model,
                'b0': results['b0'].n,
                'b1': results['b1'].n,
                'sigma': results['sigma'].n,
                'w1': results['w1'].n,
                'w_v': results['w_v'].n,
                'w_avg': results['w_avg'].n,
                'b0_s': results['b0'].s,
                'b1_s': results['b1'].s,
                'sigma_s': results['sigma'].s,
                'w1_s': results['w1'].s,
                'w_v_s': results['w_v'].s,
                'w_avg_s': results['w_avg'].s,
                'b1+w_v': (results['b1']+results['w_v']).n,
                'b1+w_v_s': (results['b1']+results['w_v']).s,
                'Adjusted R2': results['adj_r2'],
                'bic': results['bic']
            })
            
            # Load parameter fits and statistics
            #model_results_dict[model][mask] = load_modelresult(os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', model, mask,'.sav'])))
            
df = pd.DataFrame(all_data)        
     
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Mask', y='Adjusted R2', hue='Model', data=df, palette='Set2', ax=ax)

# Customize the plot
ax.set_title('RSA Normalized Codes'.format(subj), fontsize=16)
ax.set_xlabel('ROI', fontsize=12)
ax.set_ylabel('RSA Adj R2', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

df.assign(bic_rel=df.bic)
for subj in subj_list:
    for mask in df.Mask.unique():
        bic0 = df.loc[(df.Subject==subj) & (df.Mask==mask) & (df.Model=='Null'),'bic']
        df.loc[(df.Subject==subj) & (df.Mask==mask),'bic_rel']=df.loc[(df.Subject==subj) & (df.Mask==mask),'bic'].subtract(bic0, fill_value=bic0)

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Mask', y='bic_rel', estimator='sum', errorbar=None, hue='Model', data=df, palette='Set2', ax=ax)

# Customize the plot
ax.set_title('RSA Normalized Codes Relative to Null'.format(subj), fontsize=16)
ax.set_xlabel('ROI', fontsize=12)
ax.set_ylabel('Change in BIC', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()


#sigma
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
#plt.errorbar(x='Subject', y='sigma',yerr='sigma_s', data=df)
sns.scatterplot(x='Subject', y='sigma', hue='Model', data=df, palette='Set2', ax=ax)

# Customize the plot
ax.set_title('RSA Normalization Parameters'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('sigma', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()



#b Absolute across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b1', hue='Mask', data=df.loc[(df.Model=='Absolute')], palette='Set2', ax=ax)
#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
# plt.bar(x='Subject', height ='b', data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])
# plt.errorbar(x='Subject', y ='b', yerr = 'b_s', color="r", fmt="o", data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])


# Customize the plot
ax.set_title('b1 in Absolute'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b1', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

#b1 Absolute + Bundle across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b1', hue='Mask', data=df.loc[(df.Model=='Absolute + Bundle')], palette='Set2', ax=ax)
#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
# plt.bar(x='Subject', height ='b', data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])
# plt.errorbar(x='Subject', y ='b', yerr = 'b_s', color="r", fmt="o", data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])


# Customize the plot
ax.set_title('b1 in Absolute + Bundle'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b1', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

#b1 Absolute + Bundle across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b0', hue='Mask', data=df.loc[(df.Model=='Absolute + Bundle')], palette='Set2', ax=ax)
#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
# plt.bar(x='Subject', height ='b', data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])
# plt.errorbar(x='Subject', y ='b', yerr = 'b_s', color="r", fmt="o", data=df.loc[(df.Model=='Absolute') & (df.Mask=='vmPFC')])


# Customize the plot
ax.set_title('b0 in Absolute + Bundle'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b0', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()




# #b Absolute across subjets
# # Set up the plot
# fig, ax = plt.subplots(figsize=(12, 6))
# #hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
# ax=sns.barplot(x='Subject', y='b', errorbar='b_s', hue='Mask', data=df.loc[(df.Model=='Absolute')], palette='Set2', ax=ax)

# #ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
# #ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# # Customize the plot
# ax.set_title('b in Absolute'.format(subj), fontsize=16)
# ax.set_xlabel('Subject', fontsize=12)
# ax.set_ylabel('b', fontsize=12)

# # Adjust legend
# plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# # Adjust layout and display
# plt.tight_layout()
# plt.show()


#w1 Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b0', hue='Mask', data=df.loc[(df.Model=='Interaction Full(w)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('b0 in Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b0', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()


#b Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b1', hue='Mask', data=df.loc[(df.Model=='Interaction Full(w)')], palette='Set2', ax=ax)

# Customize the plot
ax.set_title('b1 in Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b1', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()


#wv Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='w_v', hue='Mask', data=df.loc[(df.Model=='Interaction Full(w)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('w_v in Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('w_v', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

#b+wv Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b1+w_v', hue='Mask', data=df.loc[(df.Model=='Interaction Full(w)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('b1+wv in Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b1+wv', fontsize=12)

abs(df.loc[(df.Model=='Interaction Full(w)')]['b1+w_v'] / df.loc[(df.Model=='Interaction Full(w)')]['b1+w_v_s'] )>1.96
#ax.bar_label(ax.containers[0],labels={1,1,1,1,1,1,1,1,1})
# for i in ax.containers:
#     breakpoint()
#     ax.bar_label(i,labels=[2,1,1])

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

(df['b1+w_v'] / df['b1+w_v_s'] )>1.96


#w1 Divisive By Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='w1', hue='Mask', data=df.loc[(df.Model=='Divisive by Cat Interaction (Diff Spec)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('w_1 in Divisive by Cat Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('w_1', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

#w1 Divisive By Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b0', hue='Mask', data=df.loc[(df.Model=='Divisive by Cat Interaction (Diff Spec)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('b0 in Divisive by Cat Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b0', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()


#w1 Divisive By Interaction across subjets
# Set up the plot
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Subject', y='b1', hue='Mask', data=df.loc[(df.Model=='Divisive by Cat Interaction (Diff Spec)')], palette='Set2', ax=ax)
#plt.bar(x='Subject', height ='w1', data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])
#plt.errorbar(x='Subject', y ='w1', yerr = 'w1_s', color="r", fmt="o", data=df.loc[(df.Model=='Interaction Average(w)') & (df.Mask=='vmPFC')])

#ax.fill_between(x, lower1, upper1, color='b', alpha=0.2)
#ax.fill_between(x, lower2, upper2, color='r', alpha=0.2)

# Customize the plot
ax.set_title('b1 in Divisive by Cat Interaction'.format(subj), fontsize=16)
ax.set_xlabel('Subject', fontsize=12)
ax.set_ylabel('b1', fontsize=12)

# Adjust legend
plt.legend(title='Mask', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()
