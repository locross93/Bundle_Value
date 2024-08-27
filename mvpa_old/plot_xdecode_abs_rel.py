#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 10:50:00 2021

@author: logancross
"""

from mvpa2.suite import *
import matplotlib.pyplot as plt
import sys
#sys.path.insert(0, "/Users/logancross/Documents/Bundle_Value/mvpa/")
sys.path.insert(0, "/Users/locro/Documents/Bundle_Value/mvpa/")
import os
#os.chdir("/Users/logancross/Documents/Bundle_Value/mvpa/")
os.chdir("/Users/locro/Documents/Bundle_Value/mvpa/")
import mvpa_utils 
import seaborn as sns
import pandas as pd
import statsmodels

# statistical annotation
def annotate_stats(rect1, rect2):
    x1, x2 = rect1.xy[0]+ (rect1.get_width() / 2), rect2.xy[0] + (rect2.get_width() / 2)
    y1, y2 = rect1.get_height(), rect2.get_height()
    y, h, col = np.max([y1,y2]) + 0.035, 0.002, 'k'
    #y, h, col = 0.14, 0.002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

#bundle_path = '/Users/logancross/Documents/Bundle_Value/'
bundle_path = '/Users/locro/Documents/Bundle_Value/'

save = False

subj_list = ['101','102','103','104','105','106','107','108','109','110','111','113','114']
subj_names = ['sub 1','sub 2','sub 3','sub 4','sub 5','sub 6', 'sub 7','sub 8','sub 9', 'sub 10','sub 11','sub 13', 'sub 14']


mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

df_data = pd.DataFrame({'Subj':[0], 'Mask':[0], 'S2Abs':[0], 'S2Rel': [0], 'B2Abs':[0], 'B2Rel': [0]})
for subj in subj_list:
    #temp_scores = np.load(bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_abs_rel_scores.npy')    
    temp_scores = np.load(bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_abs_rel_scores.npy', allow_pickle=True)
    temp_scores = temp_scores.item()
    df_temp = pd.DataFrame(temp_scores)
    df_temp = df_temp.T
    df_temp.columns = ['Subj', 'Mask', 'S2Abs', 'S2Rel', 'B2Abs', 'B2Rel']
    df_data = pd.concat([df_data, df_temp], ignore_index = True, sort = False)
df_data = df_data.drop([0])
df_plot = pd.melt(df_data, id_vars=["Subj","Mask"], var_name="Decoding Type", value_name="Accuracy")

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Abs vs Rel All Subjects')
plt.show()


df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2Abs') + (df_plot['Decoding Type'] == 'S2Rel')]
df_plot_strain = df_plot_strain.reset_index(drop=True)

reorder_inds = []
for mask in mask_names:
    mask_inds = np.where(df_plot_strain['Mask'] == mask)[0].tolist()
    reorder_inds = reorder_inds + mask_inds
    
df_plot_strain = df_plot_strain.loc[reorder_inds,:]

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, ci=68)
plt.xticks(rotation=90)
l = plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
l.get_texts()[0].set_text('Absolute Value')
l.get_texts()[1].set_text('Relative Value')
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.title('Cross Decoding - Train on Single Item', fontsize=16)

# statistical annotation
# test with nonparametric stats - wilcoxon signed rank
cat_diff_list = []
uncorr_pvals = []
for mask_num,mask in enumerate(mask_names):
    mask_abs_scores = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2Abs')]['Accuracy'].values
    mask_rel_scores = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2Rel')]['Accuracy'].values
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
        print mask,' Significant S Train',cat_diff_list[mask_num],fdr_p
        annotate_stats(ax.patches[mask_num],  ax.patches[12+mask_num])
    else:
        print mask,' Not Significant S Train',cat_diff_list[mask_num],fdr_p
if save:
    plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/xdecode_abs_rel_strain', dpi=500, bbox_inches='tight')
plt.show()

df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2Abs') + (df_plot['Decoding Type'] == 'B2Rel')]
df_plot_btrain = df_plot_btrain.reset_index(drop=True)

reorder_inds = []
for mask in mask_names:
    mask_inds = np.where(df_plot_btrain['Mask'] == mask)[0].tolist()
    reorder_inds = reorder_inds + mask_inds
    
df_plot_btrain = df_plot_btrain.loc[reorder_inds,:]

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_btrain, ci=68)
plt.xticks(rotation=90)
l = plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
l.get_texts()[0].set_text('Absolute Value')
l.get_texts()[1].set_text('Relative Value')
plt.ylabel('Accuracy (Pearson r)', fontsize=14)
plt.xlabel('ROI', fontsize=16)
plt.title('Cross Decoding - Train on Bundles', fontsize=16)

# statistical annotation
# test with nonparametric stats - wilcoxon signed rank
cat_diff_list = []
uncorr_pvals = []
for mask_num,mask in enumerate(mask_names):
    mask_abs_scores = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2Abs')]['Accuracy'].values
    mask_rel_scores = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2Rel')]['Accuracy'].values
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
        print mask,' Significant B Train',cat_diff_list[mask_num],fdr_p
        annotate_stats(ax.patches[mask_num],  ax.patches[12+mask_num])
    else:
        print mask,' Not Significant B Train',cat_diff_list[mask_num],fdr_p
if save:
    plt.savefig('/Users/logancross/Documents/Bundle_Value/figures/xdecode_abs_rel_btrain', dpi=500, bbox_inches='tight')
plt.show()



    

#mask_loop = ['ACC_pre','ACC_sup',
#             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
#             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']
#
#mask_names = ['rACC','dACC',
#              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
#              'dmPFC','dlPFC','MFG','IFG']
#    
#df_plot = np.load(bundle_path+'mvpa/analyses/Group/Cross_decoding/xdecode_abs_rel_scores.npy')
#df_plot = pd.DataFrame(df_plot, columns = ['Subj', 'Mask','Decoding Type','Accuracy'])
#
#sub104_perm_dict = np.load(bundle_path+'mvpa/analyses/sub104/perm_tests/perms_xdecode_abs_rel_scores.npy')
#sub104_perm_dict = sub104_perm_dict.item()
#
#sub104_df_plot = df_plot[(df_plot['Subj'] == '104')]
#
#df_plot_strain = sub104_df_plot[(sub104_df_plot['Decoding Type'] == 'S2Abs') + (sub104_df_plot['Decoding Type'] == 'S2Rel')]
#
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, ci=None)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding Train on Single Item Sub104')
#plt.show()
#
#df_plot_btrain = sub104_df_plot[(sub104_df_plot['Decoding Type'] == 'B2Abs') + (sub104_df_plot['Decoding Type'] == 'B2Rel')]
#
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_btrain, ci=None)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding Train on Bundles Sub104')
#plt.show()
#
#num_perms = 1000
#for p in range(num_perms):
#    sub104_perm_dict['ACC_pre']
#    
#s_abs_rel = sub104_perm_dict['ACC_pre'][0] - sub104_perm_dict['ACC_pre'][1]
#
#for mask_num, mask in enumerate(mask_loop):
#    mask_name = mask_names[mask_num]
#    abs_score = sub104_df_plot[(sub104_df_plot['Mask'] == mask_name) & (sub104_df_plot['Decoding Type'] == 'S2Abs')]['Accuracy'].values[0]
#    rel_score = sub104_df_plot[(sub104_df_plot['Mask'] == mask_name) & (sub104_df_plot['Decoding Type'] == 'S2Rel')]['Accuracy'].values[0]
#    abs_v_rel_diff = abs_score - rel_score
#    perm_diff_dist = sub104_perm_dict[mask][0] - sub104_perm_dict[mask][1]
#    max_null_diff = np.max(np.abs(perm_diff_dist))
#    null_thresh = np.percentile(perm_diff_dist, 95)
#    print mask, abs_v_rel_diff, null_thresh
#    
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding Abs vs Rel All Subjects')
#plt.show()
#
#
#df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2Abs') + (df_plot['Decoding Type'] == 'S2Rel')]
#
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_strain, ci=None)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding Train on Single Item All Subjects')
#plt.show()
#
#df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2Abs') + (df_plot['Decoding Type'] == 'B2Rel')]
#
#ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot_btrain, ci=None)
#plt.xticks(rotation=90)
#plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
#plt.title('Cross Decoding Train on Bundles All Subjects')
#plt.show()
#
## test with nonparametric stats - wilcoxon signed rank
#for mask in mask_names:
#    mask_abs_scores = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2Abs')]['Accuracy'].values
#    mask_rel_scores = df_plot_strain.loc[(df_plot_strain['Mask'] == mask) & (df_plot_strain['Decoding Type'] == 'S2Rel')]['Accuracy'].values
#    cat_diff = np.mean(mask_abs_scores - mask_rel_scores)
#    wilcoxon_p = stats.wilcoxon(mask_abs_scores, mask_rel_scores)[1]
#    if wilcoxon_p < 0.05:
#        print mask,' Significant S Train',cat_diff,wilcoxon_p
#    else:
#        print mask,' Not Significant S Train',cat_diff,wilcoxon_p
#        
#for mask in mask_names:
#    mask_abs_scores = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2Abs')]['Accuracy'].values
#    mask_rel_scores = df_plot_btrain.loc[(df_plot_btrain['Mask'] == mask) & (df_plot_btrain['Decoding Type'] == 'B2Rel')]['Accuracy'].values
#    cat_diff = np.mean(mask_abs_scores - mask_rel_scores)
#    wilcoxon_p = stats.wilcoxon(mask_abs_scores, mask_rel_scores)[1]
#    if wilcoxon_p < 0.05:
#        print mask,' Significant B Train',cat_diff,wilcoxon_p
#    else:
#        print mask,' Not Significant B Train',cat_diff,wilcoxon_p