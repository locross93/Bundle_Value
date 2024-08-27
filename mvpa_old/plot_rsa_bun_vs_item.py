#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 15:47:22 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import pandas as pd

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

scores_by_roi = np.zeros([len(subj_list), len(mask_names), 2])

subj = subj_list[0]
scores_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_bundle_vs_item2.csv')
scores_df = scores_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
scores_df['Subj'] = [subj for i in range(len(scores_df))]
#scores_by_roi[0,:,:] = scores_df.loc[:,['ivalue','bvalue']].values
scores_by_roi[0,:,:] = scores_df.loc[:,['Single Item','Bundle']].values

subj_count = 1
for subj in subj_list[1:]:
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_bundle_vs_item2.csv')
    temp_df = temp_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
    temp_df['Subj'] = [subj for i in range(len(temp_df))]
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    #scores_by_roi[subj_count,:,:] = temp_df.loc[:,['ivalue','bvalue']].values
    scores_by_roi[subj_count,:,:] = temp_df.loc[:,['Single Item','Bundle']].values
    subj_count+=1
    
avg_scores = np.mean(scores_by_roi, axis=0)
sem_scores = scipy.stats.sem(scores_by_roi, axis=0)

scores_df = scores_df.rename(columns={"ivalue": "Single Item", "bvalue": "Bundle"})
tidy_df = scores_df.melt(id_vars='ROI',value_vars=['Single Item','Bundle'])

f = sns.barplot(x='ROI', y='value', hue='variable', data=tidy_df, ci=68)
sns.despine()
f.set_xticklabels(f.get_xticklabels(), rotation=90)
plt.ylabel('RSA Correlation')