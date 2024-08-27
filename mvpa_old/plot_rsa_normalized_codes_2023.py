# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:06:17 2023

@author: locro
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import scipy
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
import itertools

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

bundle_path = '/Users/locro/Documents/Bundle_Value/'

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['104','105','106','107','108','109','110','111','112','113','114']

annotate = True
save = True
#save_file = 'rsa_div_vs_sub_group'
save_file = 'rsa_norm_model_codes_group_no_legend'

#sn_palette = 'tab10'
sn_palette = 'Set2'

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

mask_names = ['vmPFC','OFCmed','dmPFC']
#mask_names = ['vmPFC']

# file_column_keys = [('rsa_normalized_codes', 'Divide Mean'),
#                     ('rsa_normalized_codes', 'Relative'),
#                     ('rsa_normalized_codes', 'Absolute'),
#                     ('rsa_normalized_codes', 'Subtract Mean'),
#                     ('rsa_normalized_codes', 'WTP - Ref')]

# model_labels = ['Divisive', 'Z-Score', 'Absolute', 'Subtractive', 'WTP - Ref']

# file_column_keys = [('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'),
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2')]

# model_labels = ['Advanced Fractional Model', 'MC', 'MC by day', 'RW Update', 'RW Update by day']

# file_column_keys = [('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2'),
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update_adjr2'), 
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'),
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC_adjr2'),
#                     ('rsa_abs_value_adjr2', 'Absolute_adjr2')]

# model_labels = ['RW Update by day', 'RW Update', 'Advanced Fractional Model', 'MC by day', 'MC', 'Absolute Value']

# file_column_keys = [('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2'),
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'),
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
#                     ('rsa_range_norm_divisive', 'Range_adjr2'),
#                     ('rsa_abs_value_adjr2', 'Absolute_adjr2')]

# model_labels = ['DN Recency Weighted', 'DN Linear', 'Divisive Norm Static', 'Range Norm', 'Absolute Value']

file_column_keys = [('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2'),
                    ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_adjr2'),
                    ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'), 
                    ('rsa_abs_value_adjr2', 'Absolute_adjr2')]

model_labels = ['DN Recency Weighted', 'DN Linear', 'Divisive Norm Static', 'Absolute Value']

#y_label = 'RSA Correlation (r)'
y_label = 'RSA Adjusted $R^2$'

# file_column_keys = [('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_corr'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC_corr'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'MC by day_corr'),
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update_corr'), 
#                     ('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_corr')]

# model_labels = ['Advanced Fractional Model', 'MC', 'MC by day', 'RW Update', 'RW Update by day']

# file_column_keys = [('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_adjr2'),
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_adjr2'),
#                     ('rsa_normalized_codes', 'Divide Mean'),
#                     ('rsa_normalized_codes', 'Relative'),
#                     ('rsa_normalized_codes', 'Absolute'),
#                     ('rsa_normalized_codes', 'WTP - Ref')]

# model_labels = ['RW Update by day', 'Advanced Fractional Model']

# file_column_keys = [('rsa_div_rw_mc_norm_sweeps_adjr2', 'RW Update by day_corr'),
#                     ('rsa_advanced_frac_model_adjr2', 'Advanced Fractional Model_corr'),
#                     ('rsa_normalized_codes', 'Divide Mean'),
#                     ('rsa_normalized_codes', 'Relative'),
#                     ('rsa_normalized_codes', 'Absolute'),
#                     ('rsa_normalized_codes', 'WTP - Ref')]

# model_labels = ['RW Update by day', 'Advanced Fractional Model', 'Divide Mean', 'Z-Score', 'Absolute', 'WTP - Ref']


# file_column_keys and model_labels need to be the same length
assert len(file_column_keys) == len(model_labels)

models = [temp_tuple[1] for temp_tuple in file_column_keys]

for i,subj in enumerate(subj_list):
    if i == 0:
        scores_df = make_subj_df(file_column_keys, subj)
    else:
        subj_df = make_subj_df(file_column_keys, subj)
        scores_df = pd.concat([scores_df, subj_df], ignore_index=True)
        
tidy_df = scores_df.melt(id_vars='ROI',value_vars=models)
        
#reorder inds/rows based on mask_names order
reorder_inds = []
for mask in mask_names:
    mask_inds = np.where(tidy_df['ROI'] == mask)[0].tolist()
    reorder_inds = reorder_inds + mask_inds
    
df_plot = tidy_df.loc[reorder_inds,:]
df_plot = df_plot.reset_index()

# change labels of models
for model_num in range(len(models)):
    temp_model = models[model_num]
    model_inds = np.where(df_plot['variable'] == temp_model)[0].tolist()
    model_label = model_labels[model_num]
    df_plot.loc[model_inds,'variable'] = model_label

model_combos = list(itertools.combinations(model_labels, 2))

if len(mask_names) > 1:
    pairs = []
    for mask in mask_names:
        for combo in model_combos:
            temp_pair = ((mask, combo[0]),(mask, combo[1]))
            pairs.append(temp_pair)
        
    ax = sns.barplot(x='ROI', y='value', data=df_plot, hue='variable', palette=sn_palette, ci=68)
    #x_order = ['RW Update by day', 'RW Update', 'Advanced Fractional Model', 'MC by day', 'MC']
    #ax = sns.barplot(x='ROI', y='value', data=df_plot, hue='variable', hue_order=x_order, palette=sn_palette, ci=68)
    annot = Annotator(ax, pairs, data=df_plot, x='ROI', y='value', hue='variable', hide_non_significant=True)
else:
    pairs = []
    for combo in model_combos:
        temp_pair = ((combo[0]),(combo[1]))
        pairs.append(temp_pair)
    avg_corr_by_model = []
    for model in model_labels:
        temp_avg_corr = np.mean(df_plot[df_plot['variable'] == model]['value'].values)
        avg_corr_by_model.append(temp_avg_corr)
    sort_inds = np.argsort(-np.array(avg_corr_by_model))
    x_order = [model_labels[i] for i in sort_inds]
    
    #ax = sns.barplot(x='variable', y='value', order=x_order, data=df_plot, ci=68)
    ax = sns.barplot(x='variable', y='value', data=df_plot, ci=68)
    annot = Annotator(ax, pairs, data=df_plot, x='variable', y='value')

if annotate:
    #annot.configure(test='Wilcoxon', verbose=2)
    annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
    annot.apply_test()
    annot.annotate()

sns.despine()
#plt.ylabel('RSA Correlation (r)', fontsize=18)
plt.ylabel(y_label, fontsize=18)
plt.xlabel('Model', fontsize=16)
if len(mask_names) > 1:
    legend = plt.legend(bbox_to_anchor=(1.0, 1),borderaxespad=0)
    #legend.remove()
    plt.title('RSA Normalized Codes', fontsize=18)
else:
    #plt.xticks(np.arange(len(models)), model_labels, rotation=0)
    plt.xticks(rotation=45)
    plt.title('RSA Normalized Codes in '+mask_names[0], fontsize=18)
if save:
    plt.savefig(bundle_path+'figures/'+save_file, dpi=500, bbox_inches='tight')
plt.show()