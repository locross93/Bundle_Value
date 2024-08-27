# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 13:21:26 2023

@author: locro
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import statsmodels
from statsmodels.stats.multitest import multipletests
from statannotations.Annotator import Annotator
from matplotlib.legend import Legend
from matplotlib.text import Text
import numpy as np
import scipy.stats as stats

# statistical annotation
def annotate_stats(rect1):
    x1 = rect1.xy[0]+ (rect1.get_width() / 2)
    y1 = rect1.get_height() + 0.04
    plt.text(x1, y1, "*", ha='center', va='bottom', color='k')
    
def test_subj_significance(subj, df_plot_train, mask_names, trained_on):
    uncorr_pvals_2s = []
    uncorr_pvals_2b = []
    pair_uncorr_pvals = []
    sig_dict = {}
    
    for mask_num,mask in enumerate(mask_names):
        mask_score_2s = df_plot_train.loc[(df_plot_train['Mask'] == mask) 
                                            & (df_plot_train['Decoding Type'] == trained_on+'2S') 
                                            & (df_plot_train['Subj'] == subj)]['Accuracy'].values
        wilp_2s = stats.wilcoxon(mask_score_2s)[1]
        uncorr_pvals_2s.append(wilp_2s)
        mask_score_2b = df_plot_train.loc[(df_plot_train['Mask'] == mask) 
                                            & (df_plot_train['Decoding Type'] == trained_on+'2B')
                                            & (df_plot_train['Subj'] == subj)]['Accuracy'].values
        wilp_2b = stats.wilcoxon(mask_score_2b)[1]
        uncorr_pvals_2b.append(wilp_2b)
        paired_pval = stats.wilcoxon(mask_score_2s, mask_score_2b)[1]
        pair_uncorr_pvals.append(paired_pval)
        
    # multiple comparisons correction
    # test on single items
    sig_bools_2s, corr_pvals_2s, alphacSidak, alphacBonf = multipletests(uncorr_pvals_2s, alpha=0.05, method='fdr_bh')
    sig_dict['2s'] = sig_bools_2s
    
    # test on bundles
    sig_bools_2b, corr_pvals_2b, alphacSidak, alphacBonf = multipletests(uncorr_pvals_2b, alpha=0.05, method='fdr_bh')
    sig_dict['2b'] = sig_bools_2b
    
    # paired differences
    sig_bools_pair, pair_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pair_uncorr_pvals, alpha=0.05, method='fdr_bh') 
    sig_dict['pair'] = sig_bools_pair
    
    return sig_dict

#bundle_path = '/Users/logancross/Documents/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

#analysis_file = 'xdecode_rois.csv'
analysis_file = 'xdecode_rois_2023.csv'

save = True

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']

mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','112','113','114']
subj = subj_list[0]
scores_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_file)
scores_df = scores_df.drop(columns=['Unnamed: 0'])

subj_count = 1
for subj in subj_list[1:]:
    temp_df = pd.read_csv(bundle_path+'mvpa/analyses/sub'+str(subj)+'/'+analysis_file)
    temp_df = temp_df.drop(columns=['Unnamed: 0'])
    scores_df = pd.concat([scores_df, temp_df], ignore_index=True)
    
df_plot = pd.melt(scores_df, id_vars=["Subj","Mask"], var_name="Decoding Type", value_name="Accuracy")

df_plot['Subj'] = df_plot['Subj'].replace({101: '001', 102: '002', 103: '003', 
       104: '004', 105: '005', 106: '006', 107: '007', 108: '008', 109: '009', 
       110: '010', 111: '011', 112: '012', 113: '013', 114: '014'})

subnums = df_plot['Subj'].unique()

#####################################
# Train on Single Items plot
df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2S') + (df_plot['Decoding Type'] == 'S2B')]
df_plot_strain = df_plot_strain.reset_index(drop=True)

color={'S2S': 'tomato','S2B': 'dodgerblue'}

for subj in subnums:
    df_plot_subj = df_plot_strain[(df_plot_strain['Subj'] == subj)]

    ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_subj, palette=color, ci=68)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    plt.xticks(rotation=90)
    h, l = ax.get_legend_handles_labels()
    labels=['Single Item','Bundle']
    ax.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Accuracy (Pearson r)', fontsize=14)
    plt.xlabel('ROI', fontsize=16)
    plt.title('Sub'+str(subj)+': Cross Decoding Value - Train on Single Items')
    
    # are individual bars significant?
    sig_dict = test_subj_significance(subj, df_plot_strain, mask_names, trained_on='S')
    for test_on,c in enumerate(ax.containers):
        if test_on == 0:
            sig_bools = sig_dict['2s']
        elif test_on == 1:
            sig_bools = sig_dict['2b']
        
        labels = ['*' for w in c]

        # loop over the patches and the labels
        mask_num = 0
        for patch, label in zip(c.patches, labels):
            if sig_bools[mask_num]:
                # get the patch coordinates and height
                x = patch.get_x()
                y = patch.get_y()
                h = patch.get_height()
                # annotate the patch with the label at a y coordinate 0.04 higher than the height
                x_annot = x + 0.2
                y_annot = np.min([y + h + 0.07, (ax.get_ylim()[1]-0.02)])
                ax.annotate(label, (x_annot, y_annot), ha='center')
            mask_num+=1
            
    pairs = []
    for mask in mask_names:
        temp_pair = ((mask, 'S2S'),(mask, 'S2B'))
        pairs.append(temp_pair)
        
    # are paired differences significant?
    hue_order = ['S2S', 'S2B']
    order = mask_names
    subj_df = df_plot_strain.loc[df_plot_strain['Subj'] == subj]
    annot = Annotator(ax, pairs, data=subj_df, x='Mask', y='Accuracy', hue='Decoding Type', order=order, hue_order=hue_order, hide_non_significant=True)
    annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=0)
    #annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
    annot.apply_test()
    annot.annotate()
    if save:
        plt.savefig('/Users/locro/Documents/Bundle_Value/figures/individual_subj_plots/sub'+subj+'_xdecode_roi_strain.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    
#####################################
# Train on Bundles plot
df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2S') + (df_plot['Decoding Type'] == 'B2B')]
df_plot_btrain = df_plot_btrain.reset_index(drop=True)

color={'B2S': 'tomato','B2B': 'dodgerblue'}
hue_order = ['B2S','B2B']

for subj in subnums:
    df_plot_subj = df_plot_btrain[(df_plot_btrain['Subj'] == subj)]
    
    ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_subj, hue_order=hue_order, palette=color, ci=68)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    plt.xticks(rotation=90)
    h, l = ax.get_legend_handles_labels()
    labels=['Single Item','Bundle']
    ax.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
    plt.ylabel('Accuracy (Pearson r)', fontsize=14)
    plt.xlabel('ROI', fontsize=16)
    plt.title('Sub'+str(subj)+': Cross Decoding Value - Train on Bundles')
    
    # are individual bars significant?
    sig_dict = test_subj_significance(subj, df_plot_btrain, mask_names, trained_on='B')
    for test_on,c in enumerate(ax.containers):
        if test_on == 0:
            sig_bools = sig_dict['2s']
        elif test_on == 1:
            sig_bools = sig_dict['2b']
        
        labels = ['*' for w in c]
        
        # loop over the patches and the labels
        mask_num = 0
        for patch, label in zip(c.patches, labels):
            if sig_bools[mask_num]:
                # get the patch coordinates and height
                x = patch.get_x()
                y = patch.get_y()
                h = patch.get_height()
                # annotate the patch with the label at a y coordinate 0.04 higher than the height
                x_annot = x + 0.2
                y_annot = np.min([y + h + 0.07, (ax.get_ylim()[1]-0.02)])
                ax.annotate(label, (x_annot, y_annot), ha='center')
            mask_num+=1
            
    pairs = []
    for mask in mask_names:
        temp_pair = ((mask, 'B2S'),(mask, 'B2B'))
        pairs.append(temp_pair)
        
    hue_order = ['B2S', 'B2B']
    order = mask_names
    subj_df = df_plot_btrain.loc[df_plot_btrain['Subj'] == subj]
    annot = Annotator(ax, pairs, data=subj_df, x='Mask', y='Accuracy', hue='Decoding Type', order=order, hue_order=hue_order, hide_non_significant=True)
    annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=0)
    #annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
    annot.apply_test()
    annot.annotate()
    if save:
        plt.savefig('/Users/locro/Documents/Bundle_Value/figures/individual_subj_plots/sub'+subj+'_xdecode_roi_btrain.png', dpi=300, bbox_inches='tight')
    plt.show()
    

