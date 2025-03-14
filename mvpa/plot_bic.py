# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 16:37:16 2025

@author: locro
"""

import os
import json
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats 
from lmfit.model import load_modelresult

#bundle_path = '/Users/ryanwebb/Documents/GitHub/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

#subj_list = ['101', '102', '103', '104','107','108','109','110','111','112','113','114']
subj_list = ['101', '102', '103', '104', '105', '106', '107','108','109','110','111','112','113','114']

all_data = []

for subj in subj_list:
    subj_dir = os.path.join(bundle_path, 'mvpa', 'analyses', 'sub'+str(subj))
    # Define the path to the JSON file
    #results_file = os.path.join(subj_dir, 'rsa_norm_results_2_07_24.pkl')
    results_file = os.path.join(subj_dir, 'rsa_norm_results_3_13_25.pkl')
    
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
                'a2': results['a2'].n,
                'b1': results['b1'].n,
                'b2': results['b2'].n,
                'sigma': results['sigma'].n,
                'w1': results['w1'].n,
                'w_v': results['w_v'].n,
                'w_avg': results['w_avg'].n,
                'a2_s': results['a2'].s,
                'b1_s': results['b1'].s,
                'b2_s': results['b2'].s,
                'sigma_s': results['sigma'].s,
                'w1_s': results['w1'].s,
                'w_v_s': results['w_v'].s,
                'w_avg_s': results['w_avg'].s,
                'b1+b2': (results['b1'].n+results['b2']).n,
                'b1+b2_s': (results['b1']+results['b2']).s,
                'Adjusted R2': results['adj_r2'],
                'bic': results['bic']
            })
            
            # Load parameter fits and statistics
            #model_results_dict[model][mask] = load_modelresult(os.path.join(subj_dir, "".join(['rsa_norm_results_11_07_24-', model, mask,'.sav'])))
            
    # Load the subtractive models data from CSV
    subtractive_file = os.path.join(subj_dir, 'rsa_subtractive.csv')
    if os.path.exists(subtractive_file):
        subtractive_df = pd.read_csv(subtractive_file)
        
        # Print column names to debug
        print(f"Subject {subj} - CSV columns: {list(subtractive_df.columns)}")
        
        # Now we can see the file has 'ROI' column and column names like 'Subtract Mean_adjr2'
        if 'ROI' in subtractive_df.columns:
            for mask in subtractive_df['ROI'].unique():
                mask_data = subtractive_df[subtractive_df['ROI'] == mask]
                
                # Add each subtractive model to all_data
                for model_name in ['Subtract Mean', 'Subtract Median', 'Z-score']:
                    # The columns are named as model_adjr2 and model_bic (with spaces preserved)
                    adjr2_col = f'{model_name}_adjr2'
                    bic_col = f'{model_name}_bic'
                    
                    if adjr2_col in mask_data.columns and bic_col in mask_data.columns:
                        all_data.append({
                            'Subject': subj,
                            'Mask': mask,  # Use the ROI value as the Mask
                            'Model': model_name,
                            'Adjusted R2': mask_data[adjr2_col].values[0],
                            'bic': mask_data[bic_col].values[0],
                            # Setting other values to NaN
                            'a2': np.nan, 'b1': np.nan, 'b2': np.nan, 'sigma': np.nan,
                            'w1': np.nan, 'w_v': np.nan, 'w_avg': np.nan,
                            'a2_s': np.nan, 'b1_s': np.nan, 'b2_s': np.nan, 'sigma_s': np.nan,
                            'w1_s': np.nan, 'w_v_s': np.nan, 'w_avg_s': np.nan,
                            'b1+b2': np.nan, 'b1+b2_s': np.nan
                        })
        else:
            print(f"Warning: No ROI column found for subject {subj}.")

df = pd.DataFrame(all_data) 

# Filter dataframe to only include the requested ROIs
roi_filter = ['vmPFC', 'OFCmed', 'dmPFC']
filtered_df = df[df['Mask'].isin(roi_filter)]

# Set up the plot for Adjusted R2
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Mask', y='Adjusted R2', hue='Model', data=filtered_df, palette='Set2', ax=ax)

# Customize the plot
ax.set_title('RSA Normalized Codes'.format(subj), fontsize=16)
ax.set_xlabel('ROI', fontsize=12)
ax.set_ylabel('RSA Adj R2', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Fix the bic_rel calculation
# First, create the bic_rel column with the same values as bic
df['bic_rel'] = df['bic']

for subj in subj_list:
    for mask in df.Mask.unique():
        # Check if there's a Null model for this subject and mask
        null_bics = df.loc[(df.Subject==subj) & (df.Mask==mask) & (df.Model=='Null'),'bic']
        
        if not null_bics.empty:
            # Get the first (and hopefully only) Null model BIC value
            bic0 = null_bics.iloc[0]
            
            # Subtract the Null model BIC from each model's BIC for this subject and mask
            mask_indices = (df.Subject==subj) & (df.Mask==mask)
            df.loc[mask_indices, 'bic_rel'] = df.loc[mask_indices, 'bic'] - bic0
        else:
            # If there's no Null model, print a warning and keep the original BIC values
            print(f"Warning: No Null model found for subject {subj}, mask {mask}")

# Filter dataframe again after bic_rel calculation
filtered_df = df[df['Mask'].isin(roi_filter)]

# Set up the plot for BIC
fig, ax = plt.subplots(figsize=(12, 6))
#hue_order = ['Divisive by Cat', 'Sigma and w', 'Absolute']
sns.barplot(x='Mask', y='bic_rel', estimator='sum', errorbar=None, 
            hue='Model', data=filtered_df.loc[(filtered_df.Model != 'Null')], 
            palette='Set2', ax=ax)

# Customize the plot
ax.set_title('RSA Value Code Compared to Null Model'.format(subj), fontsize=16)
ax.set_xlabel('ROI', fontsize=12)
ax.set_ylabel('Change in BIC', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()

# Add plot for raw BIC values averaged across subjects
# Filter dataframe to only include the requested ROIs and exclude Null model
filtered_df_no_null = df[(df['Mask'].isin(roi_filter)) & (df['Model'] != 'Null2')]

# Group by ROI and Model to calculate mean BIC across subjects
grouped_bic = filtered_df_no_null.groupby(['Mask', 'Model'])['bic'].mean().reset_index()

# Create the plot
fig, ax = plt.subplots(figsize=(12, 6))
sns.barplot(x='Mask', y='bic', hue='Model', data=grouped_bic, palette='Set2', ax=ax)

# Customize the plot
ax.set_title('Raw BIC Values by ROI and Model (Averaged Across Subjects)', fontsize=16)
ax.set_xlabel('ROI', fontsize=12)
ax.set_ylabel('BIC Value', fontsize=12)

# Adjust legend
plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout and display
plt.tight_layout()
plt.show()