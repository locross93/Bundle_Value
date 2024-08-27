#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 08:19:27 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
import os
os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import seaborn as sns
import pandas as pd

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

subj_list = ['101','102','103','104','105','107','108','109','110','111','113','114']
subj_names = ['sub 1','sub 2','sub 3','sub 4','sub 5', 'sub 7','sub 8','sub 9', 'sub 10','sub 11','sub 13', 'sub 14']

subj_list = ['104','105','107','108','109','110','111','113','114']
subj_names = ['sub 4','sub 5', 'sub 7','sub 8','sub 9', 'sub 10','sub 11','sub 13', 'sub 14']

mask_loop = ['sup_frontal_gyr', 'acc', 'paracingulate', 'frontal_pole', 'm_OFC', 'l_OFC', 'posterior_OFC']
mask_names = ['Sup Frontal Gyr', 'ACC', 'Paracingulate', 'Frontal pole', 'mOFC/vmPFC', 'lOFC', 'post. OFC']

analysis_name_list = ['ind_item','bundle']
analysis_name_legend = ['Single Item','Bundle']

#analysis_name_list = ['abs_value','rel_value']
#analysis_name_legend = ['Absolute Value','Relative Value']

df_data = pd.DataFrame({'Mask':[0], 'Model':[0], 'Accuracy':[0], 'Subj': [0]})

analysis_count = 0
for analysis_name in analysis_name_list:

    temp_mvpa_scores = np.zeros([len(subj_list), len(mask_loop)])
    
    for subj in subj_list:
        #temp_scores = np.load(bundle_path+'mvpa/analyses/sub'+str(subj)+'/quick_reg_scores_'+analysis_name+'.npy')
        temp_scores = np.load(bundle_path+'mvpa/analyses/sub'+str(subj)+'/aal_reg_scores_'+analysis_name+'.npy')
        mask_count = 0
        for score in temp_scores:
            df_data = df_data.append({'Mask': mask_names[mask_count], 'Model': analysis_name_legend[analysis_count], 'Accuracy': score, 'Subj': subj}, ignore_index=True)
            mask_count+=1
    analysis_count+=1
df_data = df_data.drop([0])

ax = sns.barplot(x="Mask", y="Accuracy", hue="Model", data=df_data, ci=68)
plt.xticks(rotation=90)
plt.ylabel('Prediction Accuracy (r)', fontsize=16)
plt.xlabel('ROI', fontsize=16)
    
#f = sns.heatmap(mvpa_scores, annot=True, annot_kws={"size": 7},  
#                    xticklabels=mask_names, yticklabels=subj_names, vmin=0.0, vmax=0.15, cbar_kws={'label': 'Prediction Accuracy (r)'})
#plt.yticks(rotation=0) 
#plt.show()
#
##plot the average
#avg_mvpa_scores = np.mean(mvpa_scores, axis=0)
#sem_mvpa_scores = scipy.stats.sem(mvpa_scores)
#
#plt.bar(np.arange(len(mask_names)), avg_mvpa_scores, yerr=sem_mvpa_scores, edgecolor='black')
#plt.xticks(np.arange(len(mask_names)), mask_names, rotation=90)
#plt.ylabel('Prediction Accuracy (r)', fontsize=16)
#plt.show()