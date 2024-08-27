# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 19:54:31 2023

@author: locro
"""

from scipy.stats import ttest_rel

# Get unique mask values
unique_masks = subj_df['Mask'].unique()

# Perform the paired sample t-test for each mask
for mask in unique_masks:
    # Filter results for the current mask
    mask_results = subj_df[subj_df['Mask'] == mask]

    # Get the accuracy values for each condition
    s2abs_accuracies = mask_results[mask_results['Decoding Type'] == 'S2Abs']['Accuracy'].values
    s2rel_accuracies = mask_results[mask_results['Decoding Type'] == 'S2Rel']['Accuracy'].values
    b2abs_accuracies = mask_results[mask_results['Decoding Type'] == 'B2Abs']['Accuracy'].values
    b2rel_accuracies = mask_results[mask_results['Decoding Type'] == 'B2Rel']['Accuracy'].values
    
    # Perform the t-tests
    t_stat_s2, p_value_s2 = ttest_rel(s2abs_accuracies, s2rel_accuracies)
    print mask, p_value_s2