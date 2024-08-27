#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 14:43:39 2022

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
import statsmodels

# statistical annotation
def annotate_stats(rect1, rect2):
    x1, x2 = rect1.xy[0]+ (rect1.get_width() / 2), rect2.xy[0] + (rect2.get_width() / 2)
    y1, y2 = rect1.get_height(), rect2.get_height()
    y, h, col = np.max([y1,y2]) + 0.035, 0.002, 'k'
    #y, h, col = 0.14, 0.002, 'k'
    plt.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=1.5, c=col)
    plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color=col)

bundle_path = '/Users/logancross/Documents/Bundle_Value/'

save = False

subj_list = ['104','105','106','107','108','109','110','111','113','114']


mask_loop = ['ACC_pre','ACC_sup',
             'Frontal_Inf_Orb_2','Frontal_Med_Orb','OFCant','OFClat','OFCmed','OFCpost' ,
             'Frontal_Sup_Medial','Frontal_Sup_2','Frontal_Mid_2','Frontal_Inf_Tri']

mask_names = ['rACC','dACC',
              'vlPFC','vmPFC','OFCant','OFClat','OFCmed','OFCpost',
              'dmPFC','dlPFC','MFG','IFG']

df_data = pd.DataFrame({'Subj':[0], 'Mask':[0], 'S2Abs':[0], 'S2Rel': [0],'S2Sbn': [0], 'B2Abs':[0], 'B2Rel': [0], 'B2Sbn': [0],})
for subj in subj_list:
    temp_scores = np.load(bundle_path+'mvpa/analyses/sub'+str(subj)+'/xdecode_norm_codes_scores.npy')    
    temp_scores = temp_scores.item()
    df_temp = pd.DataFrame(temp_scores)
    df_temp = df_temp.T
    df_temp.columns = ['Subj', 'Mask','S2Abs','S2Rel','S2Sbn','B2Abs','B2Rel','B2Sbn']
    df_data = pd.concat([df_data, df_temp], ignore_index = True, sort = False)
df_data = df_data.drop([0])
df_plot = pd.melt(df_data, id_vars=["Subj","Mask"], var_name="Decoding Type", value_name="Accuracy")

ax = sns.barplot(x="Mask", y="Accuracy", hue="Decoding Type", data=df_plot, ci=None)
plt.xticks(rotation=90)
plt.legend(title='Decoding Type', bbox_to_anchor=(1.04,1), loc="upper left")
plt.title('Cross Decoding Abs vs Rel All Subjects')
plt.show()


#####################################
# Train on Single Items plot
df_plot_strain = df_plot[(df_plot['Decoding Type'] == 'S2Abs') + (df_plot['Decoding Type'] == 'S2Rel') + (df_plot['Decoding Type'] == 'S2Sbn')]
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


#####################################


#####################################
# Train on Bundles plot
df_plot_btrain = df_plot[(df_plot['Decoding Type'] == 'B2Abs') + (df_plot['Decoding Type'] == 'B2Rel') + (df_plot['Decoding Type'] == 'B2Sbn')]
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

#####################################