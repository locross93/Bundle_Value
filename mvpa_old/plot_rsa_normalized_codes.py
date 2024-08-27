#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 20:38:09 2022

@author: logancross
"""

#from mvpa2.suite import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import itertools

# statistical annotation
def annotate_stats(rect1, rect2):
    x1, x2 = rect1.xy[0]+ (rect1.get_width() / 2), rect2.xy[0] + (rect2.get_width() / 2)
    y1, y2 = rect1.get_height(), rect2.get_height()
    y, h, col = np.max([y1,y2]) + 0.03, 0.002, 'k'
    #y, h, col = 0.14, 0.002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

#bundle_path = '/Users/logancross/Documents/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

save = False

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

models = ['Absolute','Relative','WTP - Ref','Subtract Mean','Divide Mean']

scores_by_roi = np.zeros([len(subj_list), len(mask_names), len(models)])

subj = subj_list[0]
scores_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_normalized_codes.csv')
scores_df = scores_df.drop(columns=['ROI'])
scores_df = scores_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
scores_df['Subj'] = [subj for i in range(len(scores_df))]
scores_by_roi[0,:,:] = scores_df.loc[:,models].values

subj_count = 1
for subj in subj_list[1:]:
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_normalized_codes.csv')
    temp_df = temp_df.drop(columns=['ROI'])
    temp_df = temp_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
    temp_df['Subj'] = [subj for i in range(len(temp_df))]
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    scores_by_roi[subj_count,:,:] = temp_df.loc[:,models].values
    subj_count+=1

#scores_df = scores_df[['ROI']+models]
tidy_df = scores_df.melt(id_vars='ROI',value_vars=models)

zscore_inds = np.where(tidy_df['variable'] == 'Relative')[0].tolist()
tidy_df.loc[zscore_inds,'variable'] = 'Z-Score'
models[1] = 'Z-Score'

model_combos = list(itertools.combinations(models, 2))

pairs = []
for mask in mask_names:
    for combo in model_combos:
        temp_pair = ((mask, combo[0]),(mask, combo[1]))
        pairs.append(temp_pair)

    
ax = sns.barplot(x='ROI', y='value', hue='variable', data=tidy_df, ci=None)
sns.despine()

annot = Annotator(ax, pairs, data=tidy_df, x='ROI', y='value', hue='variable')
#annot.configure(test='Wilcoxon', verbose=2)
#annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', loc='outside', verbose=False)
annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=False)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes', fontsize=18)
plt.ylabel('RSA Correlation', fontsize=18)
plt.xlabel('ROI', fontsize=18)
plt.show()


# divide_inds = np.where(tidy_df['variable'] == 'Divide Mean')[0].tolist()
# two_cat_inds = zscore_inds + divide_inds
# df_two_cat = tidy_df.loc[two_cat_inds,:]

# two_cat_pairs = []
# for mask in mask_names:
#     temp_pair = ((mask, 'Z-Score'),(mask, 'Divide Mean'))
#     two_cat_pairs.append(temp_pair)

# colors = [sns.color_palette()[1], sns.color_palette()[4]]
# ax = sns.barplot(x='ROI', y='value', hue='variable', data=df_two_cat, ci=68, palette=colors)
# sns.despine()

# annot = Annotator(ax, two_cat_pairs, data=df_two_cat, x='ROI', y='value', hue='variable')
# #annot.configure(test='Wilcoxon', verbose=2)
# annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
# annot.apply_test()
# annot.annotate()

# plt.xticks(rotation=90)
# plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
# plt.title('RSA Normalized Codes', fontsize=18)
# plt.ylabel('RSA Correlation', fontsize=18)
# plt.xlabel('ROI', fontsize=18)
# plt.show()