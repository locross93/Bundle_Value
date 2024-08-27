#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 16:17:08 2022

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
subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

#mask_names = ['vmPFC','OFCmed','dmPFC']


file_column_keys = [('rsa_advanced_frac_model', 'Divisive'), ('rsa_mc_rw_sweeps', 'MC'), ('rsa_mc_rw_sweeps', 'MC by day'),
                    ('rsa_mc_rw_sweeps', 'RW Update'), ('rsa_mc_rw_sweeps', 'RW Update by day')]

file_column_keys = [('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
                    ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC_adjr2'), 
                    ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'),
                    ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update_adjr2'), 
                    ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2')]

# file_column_keys = [('rsa_mc_rw_by_day', 'RW Update by day_corr'),
#                     ('rsa_normalized_codes', 'Relative'),
#                     ('rsa_normalized_codes', 'Absolute'),
#                     ('rsa_normalized_codes', 'Subtract Mean'),
#                     ('rsa_normalized_codes', 'WTP - Ref')]

# file_column_keys = [('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2')]

models = [temp_tuple[1] for temp_tuple in file_column_keys]

# file_column_keys = [('rsa_simple_frac_model_adjr2', 'Simple Fractional Model_adjr2'),
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2')]

# file_column_keys = [('rsa_simple_frac_model_adjr2', 'Simple Fractional Model_bic'),
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_bic')]

# models = [temp_tuple[1] for temp_tuple in file_column_keys]

# def make_subj_df(file_column_keys, subj):
#     for i,key in enumerate(file_column_keys):
#         temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+key+'.csv')
#         #column_name = file_column_keys[key]
#         if i == 0:
#             column_name = 'Simple_Frac'
#             temp_df = temp_df.rename(columns={'Divisive': column_name})
#             subj_df = pd.DataFrame(temp_df.loc[:,['ROI',column_name]])
#         else:
#             column_name = 'Advanced_Frac'
#             temp_df = temp_df.rename(columns={'Divisive': column_name})
#             subj_df = subj_df.join(temp_df[column_name])
            
#     subj_df['Subj'] = [subj for i in range(len(subj_df))]
    
#     return subj_df

def make_subj_df(file_column_keys, subj):
    count = 0
    for file,model in file_column_keys:
        temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+file+'.csv')
        column_name = model
        if count == 0:
            subj_df = pd.DataFrame(temp_df.loc[:,['ROI',column_name]])
        else:
            subj_df = subj_df.join(temp_df[column_name])
        count+=1
            
    subj_df['Subj'] = [subj for i in range(len(subj_df))]
    
    return subj_df

for i,subj in enumerate(subj_list):
    if i == 0:
        scores_df = make_subj_df(file_column_keys, subj)
    else:
        subj_df = make_subj_df(file_column_keys, subj)
        scores_df = pd.concat([scores_df, subj_df], ignore_index=True)

#scores_df = scores_df[['ROI']+models]
tidy_df = scores_df.melt(id_vars='ROI',value_vars=models)

model_combos = list(itertools.combinations(models, 2))

pairs = []
for mask in mask_names:
    for combo in model_combos:
        temp_pair = ((mask, combo[0]),(mask, combo[1]))
        pairs.append(temp_pair)

    
#ax = sns.barplot(x='ROI', y='value', hue='variable', data=tidy_df, ci=68)
ax = sns.barplot(x='ROI', y='value', hue='variable', data=tidy_df, ci=None)
sns.despine()
#new_labels = ['RW Update by day','Weighted RW Update by day']
#plt.legend(bbox_to_anchor=(1.6, 1), labels=new_labels)
plt.legend(bbox_to_anchor=(0.95, 1))

annot = Annotator(ax, pairs, data=tidy_df, x='ROI', y='value', hue='variable')
annot.configure(test='Wilcoxon', verbose=2)
#annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', verbose=2)
#annot.configure(test='t-test_paired', verbose=2)
annot.apply_test()
annot.annotate()

plt.xticks(rotation=90)
#plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
plt.title('RSA Normalized Codes', fontsize=18)
plt.ylabel('Adj R2', fontsize=18)
plt.xlabel('ROI', fontsize=18)
plt.show()

# # statistical annotation
# # test with nonparametric stats - wilcoxon signed rank
# cat_diff_list = []
# uncorr_pvals = []
# for mask_num,mask in enumerate(mask_names):
#     mask_abs_scores = scores_df[scores_df['ROI'] == mask]['Simple Fractional Model_adjr2'].values
#     mask_rel_scores = scores_df[scores_df['ROI'] == mask]['Advanced Fractional Model_adjr2'].values
#     cat_diff = np.mean(mask_abs_scores - mask_rel_scores)
#     wilcoxon_p = stats.wilcoxon(mask_abs_scores, mask_rel_scores)[1]
#     cat_diff_list.append(cat_diff)
#     uncorr_pvals.append(wilcoxon_p)


#divide_inds = np.where(tidy_df['variable'] == 'Divide Mean')[0].tolist()
#two_cat_inds = zscore_inds + divide_inds
#df_two_cat = tidy_df.loc[two_cat_inds,:]
#
#two_cat_pairs = []
#for mask in mask_names:
#    temp_pair = ((mask, 'Z-Score'),(mask, 'Divide Mean'))
#    two_cat_pairs.append(temp_pair)
#
#colors = [sns.color_palette()[1], sns.color_palette()[4]]
#ax = sns.barplot(x='ROI', y='value', hue='variable', data=df_two_cat, ci=68, palette=colors)
#sns.despine()
#
#annot = Annotator(ax, two_cat_pairs, data=df_two_cat, x='ROI', y='value', hue='variable')
##annot.configure(test='Wilcoxon', verbose=2)
#annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
#annot.apply_test()
#annot.annotate()
#
#plt.xticks(rotation=90)
#plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
#plt.title('RSA Normalized Codes', fontsize=18)
#plt.ylabel('RSA Correlation', fontsize=18)
#plt.xlabel('ROI', fontsize=18)
#plt.show()