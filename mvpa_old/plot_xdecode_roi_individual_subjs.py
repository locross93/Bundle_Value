# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 17:15:26 2023

@author: locro
"""

#from mvpa2.suite import *
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

save = False

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

#####################################
# Train on Single Items plot
df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2S') + (df_plot['Decoding Type'] == 'S2B')]
df_plot_strain = df_plot_strain.reset_index(drop=True)

color={'S2S': 'tomato','S2B': 'dodgerblue'}
xtick_labels = [Text(0, 0, 'rACC'), Text(1, 0, 'dACC'), Text(2, 0, 'vlPFC'), Text(3, 0, 'vmPFC'), Text(4, 0, 'OFCant'), Text(5, 0, 'OFClat'), Text(6, 0, 'OFCmed'), Text(7, 0, 'OFCpost'), Text(8, 0, 'dmPFC'), Text(9, 0, 'dlPFC'), Text(10, 0, 'MFG'), Text(11, 0, 'IFG')]

g = sns.catplot(x="Mask", y="Accuracy", hue="Decoding Type", col="Subj", data=df_plot_strain,
kind="bar", ci=68, palette=color, legend=False, col_wrap=5, aspect=1.5)

# add a global title
g.fig.suptitle('Cross Decoding Value - Train on Single Items', fontsize=34)

# adjust the spacing between the title and the subplots
g.fig.subplots_adjust(top=0.9)

for sub_num,ax in enumerate(g.axes):
    ax.set_title(ax.get_title(), fontdict={'size': 20})
    ax.tick_params(labelbottom=True)
    
    subj = str((sub_num+1)).zfill(3)
    sig_dict = test_subj_significance(subj, df_plot_strain, mask_names, trained_on='S')
    
    pairs = []
    for mask in mask_names:
        temp_pair = ((mask, 'S2S'),(mask, 'S2B'))
        pairs.append(temp_pair)
        
    hue_order = ['S2S', 'S2B']
    order = mask_names
    subj_df = df_plot_strain.loc[df_plot_strain['Subj'] == subj]
    annot = Annotator(ax, pairs, data=subj_df, x='Mask', y='Accuracy', hue='Decoding Type', order=order, hue_order=hue_order, hide_non_significant=True)
    annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=0)
    #annot.configure(test='Wilcoxon', comparisons_correction='fdr_bh', text_format='star', verbose=2)
    annot.apply_test()
    annot.annotate()
    
    # loop over the axes containers
    for test_on,c in enumerate(ax.containers):
        if test_on == 0:
            sig_bools = sig_dict['2s']
        elif test_on == 1:
            sig_bools = sig_dict['2b']
        
        labels = ['*' for w in c]
        #labels = [str(num*test_on) for num in range(len(labels))]

        # loop over the patches and the labels
        mask_num = 0
        for patch, label in zip(c.patches, labels):
            if sig_bools[mask_num]:
                # get the patch coordinates and height
                x = patch.get_x()
                y = patch.get_y()
                h = patch.get_height()
                # annotate the patch with the label at a y coordinate 0.04 higher than the height
                ax.annotate(label, (x + 0.2, y + h + 0.1), ha='center')
            mask_num+=1
  
lg = plt.legend(['Single Item', 'Bundle'], prop={'size':26}, 
           bbox_to_anchor=(1.04, 1), loc="upper left")
lg.set_title('Test On', prop={'size': 30})
g.set_ylabels('Accuracy (Pearson r)', fontdict={'size': 20})
g.set_xlabels('ROI', fontdict={'size': 20})

h, l = ax.get_legend_handles_labels()
labels=['Single Item','Bundle']
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
if save:
    plt.savefig('/Users/locro/Documents/Bundle_Value/figures/xdecode_roi_individual_subj_strain.png', dpi=300)
    
#####################################
# Train on Bundles plot
df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2S') + (df_plot['Decoding Type'] == 'B2B')]
df_plot_btrain = df_plot_btrain.reset_index(drop=True)

color={'B2S': 'tomato','B2B': 'dodgerblue'}

g = sns.catplot(x="Mask", y="Accuracy", hue="Decoding Type", col="Subj", data=df_plot_btrain,
kind="bar", ci=68, palette=color, legend=False, col_wrap=5, aspect=1.5)

# add a global title
g.fig.suptitle('Cross Decoding Value - Train on Bundles', fontsize=34)

# adjust the spacing between the title and the subplots
g.fig.subplots_adjust(top=0.9)

for sub_num,ax in enumerate(g.axes):
    ax.set_title(ax.get_title(), fontdict={'size': 20})
    ax.tick_params(labelbottom=True)
    
    subj = str((sub_num+1)).zfill(3)
    sig_dict = test_subj_significance(subj, df_plot_btrain, mask_names, trained_on='B')
    
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
    
    # loop over the axes containers
    for test_on,c in enumerate(ax.containers):
        if test_on == 0:
            sig_bools = sig_dict['2s']
        elif test_on == 1:
            sig_bools = sig_dict['2b']
        
        labels = ['*' for w in c]
        #labels = [str(num*test_on) for num in range(len(labels))]

        # loop over the patches and the labels
        mask_num = 0
        for patch, label in zip(c.patches, labels):
            if sig_bools[mask_num]:
                # get the patch coordinates and height
                x = patch.get_x()
                y = patch.get_y()
                h = patch.get_height()
                # annotate the patch with the label at a y coordinate 0.04 higher than the height
                ax.annotate(label, (x + 0.2, y + h + 0.1), ha='center')
            mask_num+=1
  
lg = plt.legend(['Single Item', 'Bundle'], prop={'size':26}, 
           bbox_to_anchor=(1.04, 1), loc="upper left")
lg.set_title('Test On', prop={'size': 30})
g.set_ylabels('Accuracy (Pearson r)', fontdict={'size': 20})
g.set_xlabels('ROI', fontdict={'size': 20})

h, l = ax.get_legend_handles_labels()
labels=['Single Item','Bundle']
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
if save:
    plt.savefig('/Users/locro/Documents/Bundle_Value/figures/xdecode_roi_individual_subj_btrain.png', dpi=300)




# # TURN THIS INTO A FUNCTION TO CALL IN ax.containers loop to label appropriate bars
# subj = '004'

# def test_subj_significance(subj, df_plot_train, mask_names, trained_on):
#     uncorr_pvals_2s = []
#     uncorr_pvals_2b = []
#     pair_uncorr_pvals = []
#     sig_dict = {}
    
#     for mask_num,mask in enumerate(mask_names):
#         mask_score_2s = df_plot_train.loc[(df_plot_train['Mask'] == mask) 
#                                             & (df_plot_train['Decoding Type'] == trained_on+'2S') 
#                                             & (df_plot_train['Subj'] == subj)]['Accuracy'].values
#         wilp_2s = stats.wilcoxon(mask_score_2s)[1]
#         uncorr_pvals_2s.append(wilp_2s)
#         mask_score_2b = df_plot_train.loc[(df_plot_train['Mask'] == mask) 
#                                             & (df_plot_train['Decoding Type'] == trained_on+'2B')
#                                             & (df_plot_train['Subj'] == subj)]['Accuracy'].values
#         wilp_2b = stats.wilcoxon(mask_score_2b)[1]
#         uncorr_pvals_2b.append(wilp_2b)
#         paired_pval = stats.wilcoxon(mask_score_2s, mask_score_2b)[1]
#         pair_uncorr_pvals.append(paired_pval)
        
#     # multiple comparisons correction
#     # test on single items
#     sig_bools_2s, corr_pvals_2s, alphacSidak, alphacBonf = multipletests(uncorr_pvals_2s, alpha=0.05, method='fdr_bh')
#     sig_dict['2s'] = sig_bools_2s
    
#     # test on bundles
#     sig_bools_2b, corr_pvals_2b, alphacSidak, alphacBonf = multipletests(uncorr_pvals_2b, alpha=0.05, method='fdr_bh')
#     sig_dict['2b'] = sig_bools_2b
    
#     # paired differences
#     sig_bools_pair, pair_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pair_uncorr_pvals, alpha=0.05, method='fdr_bh') 
#     sig_dict['pair'] = sig_bools_pair
    
#     return sig_dict
    
    
    
# s2s_uncorr_pvals = []
# s2b_uncorr_pvals = []
# pair_uncorr_pvals = []
# for mask_num,mask in enumerate(mask_names):
#     mask_score_s2s = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) 
#                                         & (df_plot_strain['Decoding Type'] == 'S2S') 
#                                         & (df_plot_strain['Subj'] == subj)]['Accuracy'].values
#     wilp_s2s = stats.wilcoxon(mask_score_s2s)[1]
#     s2s_uncorr_pvals.append(wilp_s2s)
#     mask_score_s2b = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) 
#                                         & (df_plot_strain['Decoding Type'] == 'S2B')
#                                         & (df_plot_strain['Subj'] == subj)]['Accuracy'].values
#     wilp_s2b = stats.wilcoxon(mask_score_s2b)[1]
#     s2b_uncorr_pvals.append(wilp_s2b)
#     paired_pval = stats.wilcoxon(mask_score_s2s, mask_score_s2b)[1]
#     pair_uncorr_pvals.append(paired_pval)

# # s2s
# sig_bools, s2s_corr_pvals, alphacSidak, alphacBonf = multipletests(s2s_uncorr_pvals, alpha=0.05, method='fdr_bh')
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = s2s_corr_pvals[mask_num]   
#     if sig:
#         print(mask,' Significant S Train S Test',fdr_p)
#         annotate_stats(ax.patches[mask_num])
#     else:
#         print(mask,' Not Significant S Train S Test',fdr_p)
# # s2b
# sig_bools, s2b_corr_pvals, alphacSidak, alphacBonf = multipletests(s2b_uncorr_pvals, alpha=0.05, method='fdr_bh')       
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = s2b_corr_pvals[mask_num]   
#     if sig:
#         print(mask,' Significant S Train B Test',fdr_p)
#         annotate_stats(ax.patches[mask_num+12])
#     else:
#         print(mask,' Not Significant S Train B Test',fdr_p)









# ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=68)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
# #ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
# plt.xticks(rotation=90)
# plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
# plt.title('Cross Decoding All Subjects')
# plt.show()

# #####################################
# # Train on Single Items plot
# df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2S') + (df_plot['Decoding Type'] == 'S2B')]
# df_plot_strain = df_plot_strain.reset_index(drop=True)

# color={'S2S': 'tomato','S2B': 'dodgerblue'}

# #ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, palette=color, ci=68) 
# ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, palette=color, ci=None)

# ## Create a custom color palette for subjects
# #unique_subjects = df_plot_strain['Subj'].unique()
# #subject_colors = sns.color_palette("husl", len(unique_subjects))
# #subject_palette = dict(zip(unique_subjects, subject_colors))
# #
# ## Plot the points for each subject
# #ax2 = sns.stripplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain,
# #                   dodge=True, marker='o', edgecolor='black', linewidth=1, alpha=0.7,
# #                   palette=subject_palette)        

# # Plot the points for each subject
# ax2 = sns.stripplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, 
#                    dodge=True, jitter=True, marker='o', edgecolor='black', linewidth=1, alpha=0.7)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
# #ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
# plt.xticks(rotation=90)
# plt.ylim([0, 0.19])

# # Remove the default legend
# ax.get_legend().remove()

# # Create custom legend for bars
# legend_elements = [plt.Line2D([0], [0], color='tomato', lw=4, label='Single Item'),
#                    plt.Line2D([0], [0], color='dodgerblue', lw=4, label='Bundle')]

# legend = Legend(ax, legend_elements, ['Single Item', 'Bundle'], title='Test On', bbox_to_anchor=(1.04, 1), loc="upper left")
# ax.add_artist(legend)

# # Create custom legend for subjects
# unique_subjects = df_plot_strain['Subj'].unique()
# legend_elements2 = [plt.Line2D([0], [0], marker='o', color='w', label=subj,
#                                markerfacecolor=ax2.get_children()[i].get_facecolor()[0]) for i, subj in enumerate(unique_subjects)]

# legend2 = Legend(ax, legend_elements2, unique_subjects, title='Subjects', bbox_to_anchor=(1.04, 0.7), loc="upper left")
# ax.add_artist(legend2)

# #h, l = ax.get_legend_handles_labels()
# #labels=['Single Item','Bundle']
# #ax.legend(h[:2], labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
# #ax.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")

# plt.ylabel('Accuracy (Pearson r)', fontsize=14)
# plt.xlabel('ROI', fontsize=16)
# plt.title('Cross Decoding Value - Train on Single Items')

# # statistical annotation
# # test with nonparametric stats - wilcoxon signed rank
# s2s_uncorr_pvals = []
# s2b_uncorr_pvals = []
# pair_uncorr_pvals = []
# for mask_num,mask in enumerate(mask_names):
#     mask_score_s2s = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2S')]['Accuracy'].values
#     wilp_s2s = stats.wilcoxon(mask_score_s2s)[1]
#     s2s_uncorr_pvals.append(wilp_s2s)
#     mask_score_s2b = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2B')]['Accuracy'].values
#     wilp_s2b = stats.wilcoxon(mask_score_s2b)[1]
#     s2b_uncorr_pvals.append(wilp_s2b)
#     paired_pval = stats.wilcoxon(mask_score_s2s, mask_score_s2b)[1]
#     pair_uncorr_pvals.append(paired_pval)
    
# # correct for multiple comparisons
# # s2s
# print '\nS2S'
# sig_bools, s2s_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(s2s_uncorr_pvals, alpha=0.05, method='fdr_bh')
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = s2s_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant S Train S Test',fdr_p
#         annotate_stats(ax.patches[mask_num])
#     else:
#         print mask,' Not Significant S Train S Test',fdr_p
# # s2b
# print '\nS2B'
# sig_bools, s2b_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(s2b_uncorr_pvals, alpha=0.05, method='fdr_bh')       
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = s2b_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant S Train B Test',fdr_p
#         annotate_stats(ax.patches[mask_num+12])
#     else:
#         print mask,' Not Significant S Train B Test',fdr_p      
        
# # paired comparison
# print '\nPairs'
# sig_bools, pair_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pair_uncorr_pvals, alpha=0.05, method='fdr_bh') 
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = pair_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant Paired Diff',fdr_p
#         #annotate_stats(ax.patches[mask_num+12])
#     else:
#         print mask,' Not Significant Paired Diff,fdr_p'     
# if save:
#     plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/xdecode_roi_strain', dpi=500, bbox_inches='tight')
# plt.show()
# #####################################


# #####################################
# # Train on Bundles plot
# df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2S') + (df_plot['Decoding Type'] == 'B2B')]
# df_plot_btrain = df_plot_btrain.reset_index(drop=True)

# color={'B2S': 'tomato','B2B': 'dodgerblue'}
# hue_order = ['B2S','B2B']

# ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_btrain, hue_order=hue_order, palette=color, ci=68)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
# #ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
# plt.xticks(rotation=90)
# #plt.ylim([0, 0.22])
# h, l = ax.get_legend_handles_labels()
# labels=['Single Item','Bundle']
# ax.legend(h, labels, title='Test On', bbox_to_anchor=(1.04,1), loc="upper left")
# plt.ylabel('Accuracy (Pearson r)', fontsize=14)
# plt.xlabel('ROI', fontsize=16)
# plt.title('Cross Decoding Value - Train on Bundles')

# # statistical annotation
# # test with nonparametric stats - wilcoxon signed rank
# b2s_uncorr_pvals = []
# b2b_uncorr_pvals = []
# pair_uncorr_pvals = []
# for mask_num,mask in enumerate(mask_names):
#     mask_score_b2s = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2S')]['Accuracy'].values
#     wilp_b2s = stats.wilcoxon(mask_score_b2s)[1]
#     b2s_uncorr_pvals.append(wilp_b2s)
#     mask_score_b2b = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2B')]['Accuracy'].values
#     wilp_b2b = stats.wilcoxon(mask_score_b2b)[1]
#     b2b_uncorr_pvals.append(wilp_b2b)
#     paired_pval = stats.wilcoxon(mask_score_b2s, mask_score_b2b)[1]
#     pair_uncorr_pvals.append(paired_pval)
    
# # correct for multiple comparisons
# # b2s
# print '\nB2S'
# sig_bools, b2s_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(b2s_uncorr_pvals, alpha=0.05, method='fdr_bh')
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = b2s_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant B Train S Test',fdr_p
#         annotate_stats(ax.patches[mask_num])
#     else:
#         print mask,' Not Significant B Train S Test',fdr_p
# # b2b
# print '\nB2B'
# sig_bools, b2b_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(b2b_uncorr_pvals, alpha=0.05, method='fdr_bh')       
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = b2b_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant B Train B Test',fdr_p
#         annotate_stats(ax.patches[mask_num+12])
#     else:
#         print mask,' Not Significant B Train B Test',fdr_p      
        
# # paired comparison
# print '\nPairs'
# sig_bools, pair_corr_pvals, alphacSidak, alphacBonf = statsmodels.stats.multitest.multipletests(pair_uncorr_pvals, alpha=0.05, method='fdr_bh') 
# for mask_num,sig in enumerate(sig_bools):
#     mask = mask_names[mask_num]
#     fdr_p = pair_corr_pvals[mask_num]   
#     if sig:
#         print mask,' Significant Paired Diff',fdr_p
#         #annotate_stats(ax.patches[mask_num+12])
#     else:
#         print mask,' Not Significant Paired Diff',fdr_p  
# if save:
#     plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/xdecode_roi_btrain', dpi=500, bbox_inches='tight')
# plt.show()
# #####################################