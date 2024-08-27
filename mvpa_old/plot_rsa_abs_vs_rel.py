#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  8 13:58:49 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# statistical annotation
def annotate_stats(rect1, rect2):
    x1, x2 = rect1.xy[0]+ (rect1.get_width() / 2), rect2.xy[0] + (rect2.get_width() / 2)
    y1, y2 = rect1.get_height(), rect2.get_height()
    y, h, col = np.max([y1,y2]) + 0.03, 0.002, 'k'
    #y, h, col = 0.14, 0.002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

save = False

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
#subj_list = ['104']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG',
              'V1','Fusiform']

scores_by_roi = np.zeros([len(subj_list), len(mask_names), 2])

subj = subj_list[0]
scores_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_abs_vs_rel.csv')
scores_df = scores_df.drop(columns=['ROI'])
scores_df = scores_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
scores_df['Subj'] = [subj for i in range(len(scores_df))]
scores_by_roi[0,:,:] = scores_df.loc[:,['Relative','Absolute']].values

subj_count = 1
for subj in subj_list[1:]:
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/rsa_abs_vs_rel.csv')
    if int(subj) < 104:
        temp_df = temp_df.drop(columns=['ROI'])
    temp_df = temp_df.rename({'Unnamed: 0': 'ROI'}, axis=1)
    temp_df['Subj'] = [subj for i in range(len(temp_df))]
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    scores_by_roi[subj_count,:,:] = temp_df.loc[:,['Relative','Absolute']].values
    subj_count+=1
    
avg_scores = np.mean(scores_by_roi, axis=0)
sem_scores = scipy.stats.sem(scores_by_roi, axis=0)

scores_df = scores_df[['ROI','Absolute','Relative','Subj']]
tidy_df = scores_df.melt(id_vars='ROI',value_vars=['Absolute','Relative'])

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

mask_names = ['vmPFC','OFCmed','dmPFC']

#reorder inds/rows based on mask_names order
reorder_inds = []
for mask in mask_names:
    mask_inds = np.where(tidy_df['ROI'] == mask)[0].tolist()
    reorder_inds = reorder_inds + mask_inds
    
df_plot = tidy_df.loc[reorder_inds,:]
ax = sns.barplot(x='ROI', y='value', hue='variable', data=df_plot, ci=68)
sns.despine()
#plt.xticks(rotation=90)
plt.xticks([0, 1, 2], ['vmPFC','mOFC','dmPFC'])
plt.legend(bbox_to_anchor=(1.3, 1),borderaxespad=0)
plt.title('RSA Absolute vs. Relative Value', fontsize=18)
plt.ylabel('RSA Correlation', fontsize=18)
plt.xlabel('ROI', fontsize=18)
        
# statistical annotation
# test with nonparametric stats - wilcoxon signed rank
cat_diff_list = []
uncorr_pvals = []
for mask_num,mask in enumerate(mask_names):
    mask_abs_scores = scores_df[scores_df['ROI'] == mask]['Absolute'].values
    mask_rel_scores = scores_df[scores_df['ROI'] == mask]['Relative'].values
    cat_diff = np.mean(mask_abs_scores - mask_rel_scores)
    wilcoxon_p = stats.wilcoxon(mask_abs_scores, mask_rel_scores)[1]
    cat_diff_list.append(cat_diff)
    uncorr_pvals.append(wilcoxon_p)
    
# correct for multiple comparisons
sig_bools, corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(uncorr_pvals, alpha=0.05, method='fdr_bh')
for mask_num,sig in enumerate(sig_bools):
    mask = mask_names[mask_num]
    fdr_p = corr_pvals[mask_num]
    if sig:
        print mask,' Significant ',cat_diff_list[mask_num],fdr_p
        annotate_stats(ax.patches[mask_num],  ax.patches[12+mask_num])
    else:
        print mask,' Not Significant ',cat_diff_list[mask_num],fdr_p
if save:
    plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/rsa_abs_rel', dpi=500, bbox_inches='tight')
plt.show()