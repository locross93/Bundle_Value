# -*- coding: utf-8 -*-
"""
Created on Sun May 16 16:30:19 2021

@author: locro
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

analysis_name = 'rsa_val_bin_trialtype_btwnday'

subj = subj_list[0]
scores_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_name+'.csv')

subj_count = 1
for subj in subj_list[1:]:
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_val_bin_trialtype_btwnday.csv')
    temp_df['Subj'] = [subj for i in range(len(temp_df))]
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    subj_count+=1
    
#avg_scores = np.mean(scores_by_roi, axis=0)
#sem_scores = scipy.stats.sem(scores_by_roi, axis=0)

#tidy_df = scores_df.melt(id_vars='ROI',value_vars=['ivalue','bvalue'])

f = sns.barplot(x='ROI', y='value', hue='variable', data=scores_df, ci=68)
sns.despine()
f.set_xticklabels(f.get_xticklabels(), rotation=90)
plt.ylabel('RSA Correlation')